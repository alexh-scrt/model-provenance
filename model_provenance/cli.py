"""cli.py: Typer-based CLI entrypoint for model_provenance.

Exposes the following command groups and sub-commands:

- ``verify``  — Fingerprint a model, check hashes against the known-good
  database, scan for suspicious files, and check license compliance.  Prints
  a Rich verdict table and exits with a non-zero status code on FAIL.

- ``report``  — Full provenance report (same pipeline as ``verify``) but
  always produces complete structured output in Rich, JSON, or YAML format.

- ``db init``    — Initialise the local SQLite database and seed it from the
  bundled YAML file.
- ``db add``     — Add a known-good hash record to the database.
- ``db list``    — List all model/revision pairs in the database.
- ``db query``   — Query all known-good hashes for a specific model.
- ``db remove``  — Remove a specific hash record from the database.

All commands are implemented with graceful error handling and Rich-formatted
error messages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from typing_extensions import Annotated

from model_provenance import __version__
from model_provenance.checker import Verdict, check_manifest
from model_provenance.db import HashDatabase, get_default_db
from model_provenance.fetcher import fetch_model_listing
from model_provenance.fingerprint import build_manifest_from_directory
from model_provenance.license_check import check_license_from_card
from model_provenance.reporter import (
    ProvenanceReport,
    assemble_report,
    render_report,
    render_rich_to_console,
    write_report_to_file,
)
from model_provenance.scanner import ModelScanner

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Rich console
# ---------------------------------------------------------------------------

console = Console(stderr=False, highlight=False, markup=True)
err_console = Console(stderr=True, highlight=False, markup=True)

# ---------------------------------------------------------------------------
# Typer application tree
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="model-provenance",
    help=(
        "Audit AI models for supply chain integrity: cryptographic fingerprinting, "
        "tamper detection, suspicious file scanning, and provenance reporting."
    ),
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

db_app = typer.Typer(
    name="db",
    help="Manage the local known-good hash database.",
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
app.add_typer(db_app, name="db")


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"model-provenance {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose debug logging.",
        ),
    ] = False,
) -> None:
    """model-provenance: Audit AI model supply chain integrity."""
    if verbose:
        logging.getLogger("model_provenance").setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Internal pipeline helper
# ---------------------------------------------------------------------------


def _run_audit_pipeline(
    model_source: str,
    revision: str,
    local: bool,
    token: Optional[str],
    skip_scan: bool,
    skip_license: bool,
    strict: bool,
    db: HashDatabase,
) -> ProvenanceReport:
    """Run the full audit pipeline and return a :class:`ProvenanceReport`.

    Args:
        model_source: Model ID (Hub) or directory path (local).
        revision: Git revision for Hub models.
        local: Force local directory mode.
        token: Hugging Face API token.
        skip_scan: Skip suspicious file scanning.
        skip_license: Skip license compliance checking.
        strict: Unknown files trigger FAIL verdict.
        db: An open :class:`~model_provenance.db.HashDatabase` instance.

    Returns:
        A fully assembled :class:`ProvenanceReport`.
    """
    errors: list[str] = []
    manifest = None
    check_result = None
    scan_report = None
    license_report = None
    source_label = "local" if local else "hub"
    effective_revision = revision

    # ----------------------------------------------------------------
    # Step 1: Fetch file listing and model card
    # ----------------------------------------------------------------
    try:
        listing = fetch_model_listing(
            model_source=model_source,
            revision=revision,
            token=token,
            local=local,
        )
        source_label = listing.source
        effective_revision = listing.revision
        if listing.fetch_error:
            errors.append(f"Fetch warning: {listing.fetch_error}")
    except NotADirectoryError as exc:
        errors.append(f"Invalid path: {exc}")
        return assemble_report(
            model_id=model_source,
            revision=revision,
            source=source_label,
            errors=errors,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to fetch model listing: {exc}")
        listing = None

    # ----------------------------------------------------------------
    # Step 2: Fingerprint (local only — Hub files are not downloaded)
    # ----------------------------------------------------------------
    source_path = Path(str(model_source))
    if local or source_path.exists():
        try:
            manifest = build_manifest_from_directory(
                directory=source_path,
                model_id=model_source,
                revision=revision,
            )
            effective_revision = manifest.revision
        except NotADirectoryError as exc:
            errors.append(f"Fingerprint error: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Fingerprint error: {exc}")
    else:
        # For Hub models we cannot compute local hashes without downloading.
        # We log a note but continue with listing-only data.
        logger.info(
            "Hub model '%s' — skipping local fingerprinting (no download). "
            "Hash checking will rely on the known-good database only.",
            model_source,
        )

    # ----------------------------------------------------------------
    # Step 3: Hash check against known-good database
    # ----------------------------------------------------------------
    if manifest is not None:
        try:
            check_result = check_manifest(
                manifest=manifest,
                db=db,
                revision=effective_revision,
                strict=strict,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Hash check error: {exc}")

    # ----------------------------------------------------------------
    # Step 4: Suspicious file scan
    # ----------------------------------------------------------------
    if not skip_scan:
        if local or source_path.exists():
            try:
                scanner = ModelScanner()
                scan_report = scanner.scan_directory(
                    directory=source_path,
                    model_id=model_source,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Scan error: {exc}")
        else:
            logger.info(
                "Hub model '%s' — skipping file scan (no local files).",
                model_source,
            )

    # ----------------------------------------------------------------
    # Step 5: License compliance check
    # ----------------------------------------------------------------
    if not skip_license and listing is not None:
        try:
            license_report = check_license_from_card(listing.card)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"License check error: {exc}")

    # ----------------------------------------------------------------
    # Step 6: Assemble
    # ----------------------------------------------------------------
    return assemble_report(
        model_id=model_source,
        revision=effective_revision,
        source=source_label,
        manifest=manifest,
        check_result=check_result,
        scan_report=scan_report,
        license_report=license_report,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# ``verify`` command
# ---------------------------------------------------------------------------


@app.command("verify")
def verify_command(
    model_source: Annotated[
        str,
        typer.Argument(
            help=(
                "Hugging Face Hub model ID (e.g. 'bert-base-uncased') or path "
                "to a local model directory."
            ),
        ),
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="Git revision (branch, tag, or commit SHA) for Hub models.",
        ),
    ] = "main",
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            "-l",
            help="Treat the model source as a local filesystem directory.",
        ),
    ] = False,
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            "-t",
            help="Hugging Face API token for private repositories.",
            envvar="HF_TOKEN",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'rich' (default), 'json', or 'yaml'.",
        ),
    ] = "rich",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Save the report to this file path instead of printing to stdout.",
        ),
    ] = None,
    skip_scan: Annotated[
        bool,
        typer.Option(
            "--no-scan",
            help="Skip the suspicious file scanner.",
        ),
    ] = False,
    skip_license: Annotated[
        bool,
        typer.Option(
            "--no-license",
            help="Skip the license compliance checker.",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Unknown files (not in DB) cause a FAIL verdict instead of WARN.",
        ),
    ] = False,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db",
            help="Path to the SQLite known-good hash database (default: ~/.model-provenance/hashes.db).",
        ),
    ] = None,
    seed_db: Annotated[
        bool,
        typer.Option(
            "--seed-db",
            help="Seed the database from the bundled YAML before running.",
        ),
    ] = False,
) -> None:
    """Verify a model's supply chain integrity.

    Fingerprints model files, checks hashes against the known-good database,
    scans for suspicious content, and checks license compliance.

    Exits with code 0 on PASS, 1 on WARN, 2 on FAIL.
    """
    fmt_lower = fmt.strip().lower()
    if fmt_lower not in ("rich", "json", "yaml"):
        err_console.print(
            f"[red]Error:[/red] Unknown format '{fmt}'. Choose from: rich, json, yaml."
        )
        raise typer.Exit(code=2)

    # Open database.
    try:
        db = HashDatabase(db_path) if db_path is not None else get_default_db()
        db.init_schema()
        if seed_db:
            inserted = db.seed_from_yaml()
            console.print(
                f"[dim]Seeded {inserted} new record(s) into the database.[/dim]"
            )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    # Run pipeline.
    try:
        report = _run_audit_pipeline(
            model_source=model_source,
            revision=revision,
            local=local,
            token=token,
            skip_scan=skip_scan,
            skip_license=skip_license,
            strict=strict,
            db=db,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Audit pipeline error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    finally:
        db.close()

    # Render / write output.
    try:
        if output is not None:
            write_report_to_file(report, output, fmt=fmt_lower)
            console.print(
                f"[green]Report written to:[/green] {output}"
            )
            # Also print a brief verdict summary to the terminal.
            _print_verdict_summary(report)
        else:
            render_report(report, fmt=fmt_lower)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Output error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    # Exit code based on verdict.
    _exit_for_verdict(report.verdict)


# ---------------------------------------------------------------------------
# ``report`` command
# ---------------------------------------------------------------------------


@app.command("report")
def report_command(
    model_source: Annotated[
        str,
        typer.Argument(
            help=(
                "Hugging Face Hub model ID (e.g. 'bert-base-uncased') or path "
                "to a local model directory."
            ),
        ),
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="Git revision (branch, tag, or commit SHA) for Hub models.",
        ),
    ] = "main",
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            "-l",
            help="Treat the model source as a local filesystem directory.",
        ),
    ] = False,
    token: Annotated[
        Optional[str],
        typer.Option(
            "--token",
            "-t",
            help="Hugging Face API token for private repositories.",
            envvar="HF_TOKEN",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'rich' (default), 'json', or 'yaml'.",
        ),
    ] = "rich",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Save the report to this file path instead of printing to stdout.",
        ),
    ] = None,
    skip_scan: Annotated[
        bool,
        typer.Option(
            "--no-scan",
            help="Skip the suspicious file scanner.",
        ),
    ] = False,
    skip_license: Annotated[
        bool,
        typer.Option(
            "--no-license",
            help="Skip the license compliance checker.",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Unknown files (not in DB) cause a FAIL verdict instead of WARN.",
        ),
    ] = False,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db",
            help="Path to the SQLite known-good hash database.",
        ),
    ] = None,
    seed_db: Annotated[
        bool,
        typer.Option(
            "--seed-db",
            help="Seed the database from the bundled YAML before running.",
        ),
    ] = False,
) -> None:
    """Generate a full structured provenance report for a model.

    Runs the complete audit pipeline (fingerprinting, hash checking, file
    scanning, license compliance) and produces a detailed report.

    Exits with code 0 on PASS, 1 on WARN, 2 on FAIL.
    """
    fmt_lower = fmt.strip().lower()
    if fmt_lower not in ("rich", "json", "yaml"):
        err_console.print(
            f"[red]Error:[/red] Unknown format '{fmt}'. Choose from: rich, json, yaml."
        )
        raise typer.Exit(code=2)

    # Open database.
    try:
        db = HashDatabase(db_path) if db_path is not None else get_default_db()
        db.init_schema()
        if seed_db:
            inserted = db.seed_from_yaml()
            console.print(
                f"[dim]Seeded {inserted} new record(s) into the database.[/dim]"
            )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    # Run pipeline.
    try:
        report = _run_audit_pipeline(
            model_source=model_source,
            revision=revision,
            local=local,
            token=token,
            skip_scan=skip_scan,
            skip_license=skip_license,
            strict=strict,
            db=db,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Audit pipeline error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    finally:
        db.close()

    # Render / write output.
    try:
        if output is not None:
            write_report_to_file(report, output, fmt=fmt_lower)
            console.print(f"[green]Report written to:[/green] {output}")
            _print_verdict_summary(report)
        else:
            render_report(report, fmt=fmt_lower)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Output error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    _exit_for_verdict(report.verdict)


# ---------------------------------------------------------------------------
# ``db`` sub-commands
# ---------------------------------------------------------------------------


@db_app.command("init")
def db_init(
    yaml_path: Annotated[
        Optional[Path],
        typer.Option(
            "--yaml",
            "-y",
            help="Path to a custom known-good YAML seed file (default: bundled data).",
        ),
    ] = None,
    db_path: Annotated[
        Optional[Path],
        typer.Option(
            "--db",
            help="Path to the SQLite database file.",
        ),
    ] = None,
) -> None:
    """Initialise and seed the local known-good hash database.

    Creates the SQLite database (if not already present) and seeds it from
    the bundled YAML file (or a custom file if --yaml is provided).
    """
    try:
        db = HashDatabase(db_path) if db_path is not None else HashDatabase()
        db.init_schema()
        inserted = db.seed_from_yaml(yaml_path)
        total = db.count()
        console.print(
            f"[green]✅ Database initialised.[/green] "
            f"Inserted {inserted} new record(s). Total records: {total}."
        )
        db.close()
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database init error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@db_app.command("add")
def db_add(
    model_id: Annotated[
        str,
        typer.Argument(help="Hugging Face model ID or any model identifier."),
    ],
    file_path: Annotated[
        str,
        typer.Argument(help="Relative file path within the model repository."),
    ],
    sha256: Annotated[
        str,
        typer.Argument(help="SHA-256 hex digest of the file."),
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="Git revision (branch, tag, or commit SHA).",
        ),
    ] = "main",
    source: Annotated[
        str,
        typer.Option(
            "--source",
            "-s",
            help="Provenance source label (official, community, computed, etc.).",
        ),
    ] = "computed",
    notes: Annotated[
        Optional[str],
        typer.Option(
            "--notes",
            "-n",
            help="Optional notes for this record.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite an existing record with the same key.",
        ),
    ] = False,
    db_path: Annotated[
        Optional[Path],
        typer.Option("--db", help="Path to the SQLite database file."),
    ] = None,
) -> None:
    """Add a known-good hash record to the database."""
    if len(sha256.strip()) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha256.strip()):
        err_console.print(
            "[red]Error:[/red] sha256 must be a 64-character hexadecimal string."
        )
        raise typer.Exit(code=1)

    try:
        db = HashDatabase(db_path) if db_path is not None else get_default_db()
        db.init_schema()
        inserted = db.add_hash(
            model_id=model_id,
            file_path=file_path,
            sha256=sha256,
            revision=revision,
            source=source,
            notes=notes,
            overwrite=overwrite,
        )
        if inserted:
            console.print(
                f"[green]✅ Added:[/green] "
                f"{model_id}@{revision}/{file_path} → {sha256[:16]}…"
            )
        else:
            console.print(
                f"[yellow]⚠️  Record already exists[/yellow] (use --overwrite to update): "
                f"{model_id}@{revision}/{file_path}"
            )
        db.close()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@db_app.command("list")
def db_list(
    db_path: Annotated[
        Optional[Path],
        typer.Option("--db", help="Path to the SQLite database file."),
    ] = None,
) -> None:
    """List all model/revision pairs stored in the database."""
    try:
        db = HashDatabase(db_path) if db_path is not None else get_default_db()
        db.init_schema()
        models = db.list_models()
        total = db.count()
        db.close()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not models:
        console.print(
            "[yellow]No records found.[/yellow] "
            "Run [bold]model-provenance db init[/bold] to seed the database."
        )
        return

    table = Table(
        title=f"Known-Good Hash Database ({total} total records)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Model ID", style="cyan", no_wrap=False)
    table.add_column("Revision", style="dim", width=20)

    for model_id, revision in models:
        table.add_row(model_id, revision)

    console.print(table)


@db_app.command("query")
def db_query(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID to query."),
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="Git revision to query.",
        ),
    ] = "main",
    db_path: Annotated[
        Optional[Path],
        typer.Option("--db", help="Path to the SQLite database file."),
    ] = None,
) -> None:
    """Query all known-good hashes for a specific model and revision."""
    try:
        db = HashDatabase(db_path) if db_path is not None else get_default_db()
        db.init_schema()
        records = db.get_all_hashes_for_model(model_id, revision=revision)
        db.close()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not records:
        console.print(
            f"[yellow]No records found[/yellow] for [cyan]{model_id}[/cyan] "
            f"at revision [cyan]{revision}[/cyan]."
        )
        return

    table = Table(
        title=f"Known-Good Hashes: {model_id}@{revision}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("File Path", style="cyan", no_wrap=False, max_width=50)
    table.add_column("SHA-256 (prefix)", style="dim", width=20)
    table.add_column("Source", style="dim", width=12)
    table.add_column("Notes", style="dim", no_wrap=False, max_width=30)

    for record in records:
        table.add_row(
            record.file_path,
            record.sha256[:16] + "…",
            record.source,
            record.notes or "",
        )

    console.print(table)
    console.print(f"  [dim]{len(records)} record(s) found.[/dim]")


@db_app.command("remove")
def db_remove(
    model_id: Annotated[
        str,
        typer.Argument(help="Model ID of the record to remove."),
    ],
    file_path: Annotated[
        str,
        typer.Argument(help="File path of the record to remove."),
    ],
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "-r",
            help="Git revision of the record to remove.",
        ),
    ] = "main",
    db_path: Annotated[
        Optional[Path],
        typer.Option("--db", help="Path to the SQLite database file."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Remove a specific hash record from the database."""
    if not yes:
        confirmed = typer.confirm(
            f"Remove record for {model_id}@{revision}/{file_path}?"
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit()

    try:
        db = HashDatabase(db_path) if db_path is not None else get_default_db()
        db.init_schema()
        deleted = db.delete_hash(
            model_id=model_id,
            revision=revision,
            file_path=file_path,
        )
        db.close()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Database error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if deleted:
        console.print(
            f"[green]✅ Removed:[/green] {model_id}@{revision}/{file_path}"
        )
    else:
        console.print(
            f"[yellow]⚠️  Record not found:[/yellow] {model_id}@{revision}/{file_path}"
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _print_verdict_summary(report: ProvenanceReport) -> None:
    """Print a brief single-line verdict summary to the terminal.

    Args:
        report: The :class:`ProvenanceReport` to summarise.
    """
    verdict_icons = {
        Verdict.PASS: "[bold green]✅ PASS[/bold green]",
        Verdict.WARN: "[bold yellow]⚠️  WARN[/bold yellow]",
        Verdict.FAIL: "[bold red]❌ FAIL[/bold red]",
    }
    icon = verdict_icons.get(report.verdict, report.verdict.value.upper())
    console.print(
        f"Verdict: {icon} — "
        f"[cyan]{report.model_id}[/cyan]@[cyan]{report.revision}[/cyan]"
    )


def _exit_for_verdict(verdict: Verdict) -> None:
    """Exit the process with an appropriate exit code based on the verdict.

    Exit codes:
        0 — PASS
        1 — WARN
        2 — FAIL

    Args:
        verdict: The overall :class:`~model_provenance.checker.Verdict`.
    """
    if verdict == Verdict.PASS:
        raise typer.Exit(code=0)
    elif verdict == Verdict.WARN:
        raise typer.Exit(code=1)
    else:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Entry point (for direct invocation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app()
