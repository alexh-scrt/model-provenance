"""license_check.py: License parsing and regulatory compliance checking.

This module parses model card license metadata and emits structured warnings
for restricted, non-commercial, or proprietary licenses. It also annotates
findings with relevant regulatory compliance notes for the EU AI Act and the
NIST AI Risk Management Framework (RMF).

The primary entry point is :func:`check_license`, which accepts a license
identifier string (SPDX or free-form) and returns a :class:`LicenseReport`
containing restriction flags, compliance notes, and remediation suggestions.

The module also exports :func:`check_license_from_card` which works directly
with :class:`~model_provenance.fetcher.ModelCardInfo` objects returned by the
fetcher module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LicenseRestrictionLevel(str, Enum):
    """How restricted a license is for typical production AI use.

    Values:
        PERMISSIVE: Open-source permissive license (MIT, Apache-2.0, BSD, etc.).
            No significant restrictions on commercial or production use.
        COPYLEFT: Share-alike / copyleft license (GPL, LGPL, CC-BY-SA, etc.).
            Derivative works may need to be released under the same terms.
        NON_COMMERCIAL: License prohibits commercial use (CC-BY-NC, etc.).
        CONDITIONAL: License imposes use-case or behavioural conditions
            (RAIL, OpenRAIL, BigCode OpenRAIL, Llama Community License, etc.).
        PROPRIETARY: Fully proprietary or custom license with unknown terms.
        UNKNOWN: License field is absent or unrecognised.
    """

    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    NON_COMMERCIAL = "non_commercial"
    CONDITIONAL = "conditional"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class ComplianceFramework(str, Enum):
    """Regulatory / standards framework referenced in a compliance note.

    Values:
        EU_AI_ACT: European Union Artificial Intelligence Act.
        NIST_RMF: NIST AI Risk Management Framework.
        GDPR: General Data Protection Regulation (EU).
        GENERAL: General best-practice guidance not tied to a specific framework.
    """

    EU_AI_ACT = "EU AI Act"
    NIST_RMF = "NIST AI RMF"
    GDPR = "GDPR"
    GENERAL = "General"


# ---------------------------------------------------------------------------
# Compliance note
# ---------------------------------------------------------------------------


@dataclass
class ComplianceNote:
    """A single regulatory compliance observation tied to a license.

    Attributes:
        framework: The :class:`ComplianceFramework` this note applies to.
        severity: ``'info'``, ``'warning'``, or ``'critical'``.
        title: Short human-readable title.
        detail: Extended explanation of the compliance issue.
        remediation: Suggested action to address the issue.
    """

    framework: ComplianceFramework
    severity: str  # 'info' | 'warning' | 'critical'
    title: str
    detail: str
    remediation: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "framework": self.framework.value,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "remediation": self.remediation,
        }


# ---------------------------------------------------------------------------
# License report
# ---------------------------------------------------------------------------


@dataclass
class LicenseReport:
    """Structured result of a license compliance check.

    Attributes:
        spdx_id: The normalised license identifier (SPDX where possible,
            otherwise the raw string from the model card, or ``'unknown'``).
        raw_license: The original raw license string from the model card,
            or ``None`` if absent.
        restriction_level: The assessed :class:`LicenseRestrictionLevel`.
        is_restricted: ``True`` if the license imposes meaningful restrictions
            (i.e. anything other than ``PERMISSIVE``).
        is_osi_approved: ``True`` if the license is an OSI-approved open-source
            license.
        allows_commercial_use: ``True`` if commercial use is explicitly
            permitted (or not restricted).
        allows_redistribution: ``True`` if redistribution is explicitly
            permitted.
        requires_attribution: ``True`` if the license requires attribution.
        compliance_notes: List of :class:`ComplianceNote` objects with
            regulatory observations.
        summary: Human-readable one-line summary of the license status.
        remediation_notes: List of actionable remediation strings.
    """

    spdx_id: str
    raw_license: str | None
    restriction_level: LicenseRestrictionLevel
    is_restricted: bool
    is_osi_approved: bool
    allows_commercial_use: bool
    allows_redistribution: bool
    requires_attribution: bool
    compliance_notes: list[ComplianceNote] = field(default_factory=list)
    summary: str = ""
    remediation_notes: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Return ``True`` if any compliance note has severity ``'warning'`` or ``'critical'``."""
        return any(
            n.severity in ("warning", "critical") for n in self.compliance_notes
        )

    @property
    def has_critical(self) -> bool:
        """Return ``True`` if any compliance note has severity ``'critical'``."""
        return any(n.severity == "critical" for n in self.compliance_notes)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "spdx_id": self.spdx_id,
            "raw_license": self.raw_license,
            "restriction_level": self.restriction_level.value,
            "is_restricted": self.is_restricted,
            "is_osi_approved": self.is_osi_approved,
            "allows_commercial_use": self.allows_commercial_use,
            "allows_redistribution": self.allows_redistribution,
            "requires_attribution": self.requires_attribution,
            "has_warnings": self.has_warnings,
            "has_critical": self.has_critical,
            "summary": self.summary,
            "compliance_notes": [n.to_dict() for n in self.compliance_notes],
            "remediation_notes": self.remediation_notes,
        }


# ---------------------------------------------------------------------------
# License knowledge base
# ---------------------------------------------------------------------------

# Each entry maps a normalised SPDX-like ID to a tuple of:
#   (restriction_level, is_osi, allows_commercial, allows_redistribution, requires_attribution)
_LICENSE_DB: dict[
    str,
    tuple[LicenseRestrictionLevel, bool, bool, bool, bool],
] = {
    # ---- Permissive OSI-approved ----------------------------------------
    "mit": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "apache-2.0": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "bsd-2-clause": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "bsd-3-clause": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "isc": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "unlicense": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, False),
    "cc0-1.0": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, False),
    "wtfpl": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, False),
    "bsl-1.0": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "zlib": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "pddl": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, False),
    "odc-by": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    "cdla-permissive-1.0": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    "cdla-permissive-2.0": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    "ecl-2.0": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "eupl-1.1": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "eupl-1.2": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "ms-pl": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "ms-rl": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "bigscience-bloom-rail-1.0": (LicenseRestrictionLevel.CONDITIONAL, False, False, True, True),
    # ---- Creative Commons permissive -----------------------------------
    "cc-by-4.0": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    "cc-by-3.0": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    "cc-by-2.0": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    "cc-by-2.5": (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True),
    # ---- Copyleft OSI-approved -----------------------------------------
    "gpl-2.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "gpl-3.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "gpl-2.0-only": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "gpl-3.0-only": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "lgpl-2.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "lgpl-2.1": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "lgpl-3.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "agpl-3.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "mpl-2.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "cddl-1.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "eupl-1.1-copyleft": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "osl-3.0": (LicenseRestrictionLevel.COPYLEFT, True, True, True, True),
    "cc-by-sa-4.0": (LicenseRestrictionLevel.COPYLEFT, False, True, True, True),
    "cc-by-sa-3.0": (LicenseRestrictionLevel.COPYLEFT, False, True, True, True),
    "odbl": (LicenseRestrictionLevel.COPYLEFT, False, True, True, True),
    "cdla-sharing-1.0": (LicenseRestrictionLevel.COPYLEFT, False, True, True, True),
    # ---- Non-commercial -------------------------------------------------
    "cc-by-nc-4.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, True, True),
    "cc-by-nc-3.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, True, True),
    "cc-by-nc-2.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, True, True),
    "cc-by-nc-sa-4.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, True, True),
    "cc-by-nc-sa-3.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, True, True),
    "cc-by-nc-nd-4.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, False, True),
    "cc-by-nc-nd-3.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, False, True),
    "cc-by-nd-4.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, True, False, True),
    "cc-by-nd-3.0": (LicenseRestrictionLevel.NON_COMMERCIAL, False, True, False, True),
    # ---- Conditional / RAIL-based ---------------------------------------
    "openrail": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "openrail++": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "bigcode-openrail-m": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "creativeml-openrail-m": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "llama2": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "llama3": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "llama-3": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "gemma": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "falcon-180b-license": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    "deepfloyd-if-license": (LicenseRestrictionLevel.CONDITIONAL, False, False, True, True),
    "community-license": (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True),
    # ---- Other restricted / proprietary ---------------------------------
    "other": (LicenseRestrictionLevel.PROPRIETARY, False, False, False, False),
    "proprietary": (LicenseRestrictionLevel.PROPRIETARY, False, False, False, False),
    "custom": (LicenseRestrictionLevel.PROPRIETARY, False, False, False, False),
    "commercial": (LicenseRestrictionLevel.PROPRIETARY, False, True, False, True),
    "afl-3.0": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
    "artistic-2.0": (LicenseRestrictionLevel.PERMISSIVE, True, True, True, True),
}

# Alias map for common non-canonical spellings.
_LICENSE_ALIASES: dict[str, str] = {
    "apache2": "apache-2.0",
    "apache 2.0": "apache-2.0",
    "apache2.0": "apache-2.0",
    "apache license 2.0": "apache-2.0",
    "apache software license 2.0": "apache-2.0",
    "gplv2": "gpl-2.0",
    "gplv3": "gpl-3.0",
    "lgplv2": "lgpl-2.0",
    "lgplv3": "lgpl-3.0",
    "agplv3": "agpl-3.0",
    "mit license": "mit",
    "the mit license": "mit",
    "bsd": "bsd-3-clause",
    "bsd license": "bsd-3-clause",
    "cc-by": "cc-by-4.0",
    "cc-by-nc": "cc-by-nc-4.0",
    "cc-by-sa": "cc-by-sa-4.0",
    "cc-by-nd": "cc-by-nd-4.0",
    "cc-by-nc-sa": "cc-by-nc-sa-4.0",
    "cc-by-nc-nd": "cc-by-nc-nd-4.0",
    "cc0": "cc0-1.0",
    "public domain": "cc0-1.0",
    "openrail-m": "openrail",
    "open-rail": "openrail",
    "open rail": "openrail",
    "creativeml openrail-m": "creativeml-openrail-m",
    "bigcode openrail-m": "bigcode-openrail-m",
    "llama 2": "llama2",
    "llama-2": "llama2",
    "llama 3": "llama3",
    "meta llama 3": "llama3",
    "gemma terms": "gemma",
    "gemma license": "gemma",
}


# ---------------------------------------------------------------------------
# Compliance note generators
# ---------------------------------------------------------------------------


def _notes_for_permissive(spdx_id: str) -> list[ComplianceNote]:
    """Generate compliance notes for permissive licenses."""
    return [
        ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="info",
            title="Permissive license — low EU AI Act risk",
            detail=(
                f"License '{spdx_id}' is a permissive open-source license. "
                "It does not impose restrictions that conflict with the EU AI Act. "
                "Standard transparency and documentation obligations still apply "
                "for high-risk AI systems."
            ),
            remediation="No license-specific action required.",
        ),
        ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="info",
            title="Permissive license — low NIST RMF risk",
            detail=(
                f"License '{spdx_id}' is permissive and presents low license-related "
                "risk under NIST AI RMF. Document the license in your model inventory "
                "as part of the GOVERN and MAP functions."
            ),
            remediation="Document the license in your AI system inventory.",
        ),
    ]


def _notes_for_copyleft(spdx_id: str) -> list[ComplianceNote]:
    """Generate compliance notes for copyleft licenses."""
    return [
        ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="warning",
            title="Copyleft license — share-alike obligations may apply",
            detail=(
                f"License '{spdx_id}' is a copyleft/share-alike license. "
                "If you modify or integrate this model into a larger work, "
                "you may be required to release derivative works under the "
                "same license. Consult legal counsel before commercial "
                "deployment, especially for high-risk AI systems under the EU AI Act."
            ),
            remediation=(
                "Review copyleft obligations with legal counsel. "
                "Ensure derivative works comply with the license terms."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="warning",
            title="Copyleft license — document share-alike obligations",
            detail=(
                f"License '{spdx_id}' requires careful documentation under "
                "NIST AI RMF GOVERN function. Share-alike obligations must be "
                "tracked and communicated to downstream users."
            ),
            remediation=(
                "Document the copyleft obligations in your AI system inventory "
                "and inform downstream integrators."
            ),
        ),
    ]


def _notes_for_non_commercial(spdx_id: str) -> list[ComplianceNote]:
    """Generate compliance notes for non-commercial licenses."""
    return [
        ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="critical",
            title="Non-commercial license — EU commercial AI deployment prohibited",
            detail=(
                f"License '{spdx_id}' prohibits commercial use. "
                "Deploying this model in a commercial EU AI system likely violates "
                "the license terms. Under the EU AI Act, using a model in violation "
                "of its license terms constitutes non-compliance and may expose the "
                "deployer to legal liability."
            ),
            remediation=(
                "Do NOT use this model for commercial purposes without obtaining "
                "a commercial license. Seek a commercially-licensed alternative "
                "or contact the model author for a commercial license."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="critical",
            title="Non-commercial license — high NIST RMF risk",
            detail=(
                f"License '{spdx_id}' restricts commercial use. "
                "Under NIST AI RMF, the GOVERN function requires that all third-party "
                "components are used in compliance with their terms. "
                "Commercial use of this model violates these terms and creates "
                "significant legal and reputational risk."
            ),
            remediation=(
                "Document this restriction in your AI system risk register. "
                "Obtain a commercial license or replace the model before deployment."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.GENERAL,
            severity="critical",
            title="Non-commercial restriction — commercial deployment not allowed",
            detail=(
                f"License '{spdx_id}' explicitly restricts commercial use. "
                "Using this model in a revenue-generating product or service "
                "without an appropriate commercial license violates the terms."
            ),
            remediation=(
                "Contact the model author or licensor to obtain a commercial license, "
                "or choose an alternative model with a permissive license."
            ),
        ),
    ]


def _notes_for_conditional(spdx_id: str) -> list[ComplianceNote]:
    """Generate compliance notes for conditional/RAIL licenses."""
    notes: list[ComplianceNote] = [
        ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="warning",
            title="Conditional license — review use-case restrictions",
            detail=(
                f"License '{spdx_id}' is a conditional or RAIL-based license that "
                "imposes use-case restrictions (e.g. prohibited uses, behavioural "
                "conditions). Some restrictions may conflict with your intended "
                "application or with EU AI Act requirements for high-risk AI systems. "
                "Legal review is recommended."
            ),
            remediation=(
                "Read the full license terms carefully. Ensure your use case is "
                "not listed in the prohibited uses. Consult legal counsel if "
                "deploying in a regulated sector."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="warning",
            title="Conditional license — document use-case constraints",
            detail=(
                f"License '{spdx_id}' imposes conditions that must be tracked. "
                "Under NIST AI RMF GOVERN and MAP functions, all constraints on "
                "model use must be documented and communicated to operators and "
                "affected parties."
            ),
            remediation=(
                "Document the specific use-case restrictions in your AI system "
                "inventory. Ensure operators are aware of prohibited uses."
            ),
        ),
    ]

    # Add model-specific notes for known conditional licenses.
    llama_ids = {"llama2", "llama3", "llama-3"}
    if spdx_id in llama_ids:
        notes.append(
            ComplianceNote(
                framework=ComplianceFramework.EU_AI_ACT,
                severity="warning",
                title="Llama community license — monthly active user threshold",
                detail=(
                    f"The '{spdx_id}' license (Meta Llama Community License) "
                    "restricts use by services with more than 700 million monthly "
                    "active users. Additionally, it prohibits use to improve other "
                    "large language models not developed by Meta. "
                    "These restrictions are relevant for EU AI Act compliance "
                    "assessments of high-volume AI systems."
                ),
                remediation=(
                    "If your service exceeds the MAU threshold, contact Meta to "
                    "obtain a separate license. Document the MAU constraints in "
                    "your AI system risk register."
                ),
            )
        )

    if spdx_id == "gemma":
        notes.append(
            ComplianceNote(
                framework=ComplianceFramework.EU_AI_ACT,
                severity="warning",
                title="Google Gemma license — prohibited use restrictions",
                detail=(
                    "The Gemma Terms of Use prohibit using the model in ways that "
                    "violate applicable laws or Google's usage policies. "
                    "Review the terms for prohibited applications before EU deployment."
                ),
                remediation=(
                    "Review Google's Gemma prohibited use policy and ensure your "
                    "application complies before deployment."
                ),
            )
        )

    return notes


def _notes_for_proprietary(spdx_id: str) -> list[ComplianceNote]:
    """Generate compliance notes for proprietary/custom licenses."""
    return [
        ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="critical",
            title="Proprietary/custom license — legal review required",
            detail=(
                f"License '{spdx_id}' is proprietary or custom. The full terms are "
                "not publicly known or standardised. Deploying this model in an "
                "EU AI Act-regulated context without a thorough legal review of "
                "the license terms creates significant compliance risk. "
                "The EU AI Act requires that providers of high-risk AI systems "
                "maintain control over the components they use."
            ),
            remediation=(
                "Obtain and review the full license text before deployment. "
                "Engage legal counsel for an assessment of compliance with "
                "EU AI Act obligations. Consider requesting explicit written "
                "permission for your intended use case."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="critical",
            title="Proprietary/custom license — high NIST RMF risk",
            detail=(
                f"License '{spdx_id}' is proprietary or non-standard. "
                "Under NIST AI RMF GOVERN function, all third-party components "
                "must be used in accordance with their terms. Proprietary licenses "
                "often impose restrictions on modification, redistribution, and "
                "commercial use that must be explicitly assessed."
            ),
            remediation=(
                "Document the license terms in your AI system risk register. "
                "Perform a legal review and obtain explicit authorisation for "
                "your intended use before deployment."
            ),
        ),
    ]


def _notes_for_unknown() -> list[ComplianceNote]:
    """Generate compliance notes when the license is unknown or absent."""
    return [
        ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="warning",
            title="License unknown — cannot assess EU AI Act compliance",
            detail=(
                "No license information is available for this model. "
                "Without a known license, it is not possible to confirm that "
                "the model can be used in compliance with the EU AI Act. "
                "Absent a license, the model may be considered 'all rights reserved', "
                "which would prohibit commercial use and redistribution."
            ),
            remediation=(
                "Contact the model author to determine the applicable license. "
                "Do not deploy in production until the license terms are confirmed."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="warning",
            title="License unknown — cannot document compliance",
            detail=(
                "NIST AI RMF GOVERN function requires documentation of all "
                "third-party component licenses. An unknown license makes it "
                "impossible to complete this documentation requirement."
            ),
            remediation=(
                "Identify and document the license before including this model "
                "in a NIST AI RMF-compliant AI system."
            ),
        ),
        ComplianceNote(
            framework=ComplianceFramework.GENERAL,
            severity="warning",
            title="No license specified — treat as all rights reserved",
            detail=(
                "When no license is specified, copyright law generally treats "
                "the work as 'all rights reserved'. This means you may not have "
                "the right to use, modify, or redistribute the model."
            ),
            remediation=(
                "Seek written permission from the model author before any use, "
                "or choose a model with an explicit open-source license."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalise_license_id(raw: str) -> str:
    """Normalise a raw license string to a canonical lowercase SPDX-like ID.

    Applies the following transformations:
    1. Strip whitespace and convert to lowercase.
    2. Look up in the alias table.
    3. Return as-is if no alias matches (still lowercased).

    Args:
        raw: Raw license string from a model card or database.

    Returns:
        Normalised license identifier string.
    """
    stripped = raw.strip().lower()
    # Try exact alias match.
    if stripped in _LICENSE_ALIASES:
        return _LICENSE_ALIASES[stripped]
    # Try with common suffix cleanup (e.g. "apache-2.0-or-later" → "apache-2.0").
    for alias, canonical in _LICENSE_ALIASES.items():
        if stripped.startswith(alias):
            return canonical
    return stripped


def _lookup_license(normalised_id: str) -> tuple[
    LicenseRestrictionLevel, bool, bool, bool, bool
] | None:
    """Look up a normalised license ID in the knowledge base.

    Args:
        normalised_id: Lowercased, normalised license identifier.

    Returns:
        A 5-tuple ``(restriction_level, is_osi, allows_commercial,
        allows_redistribution, requires_attribution)`` or ``None`` if not
        found.
    """
    if normalised_id in _LICENSE_DB:
        return _LICENSE_DB[normalised_id]

    # Partial / prefix matching for variant spellings.
    for key, value in _LICENSE_DB.items():
        if normalised_id.startswith(key) or key.startswith(normalised_id):
            return value

    # Pattern-based fallback.
    if re.search(r"\bnon[- ]?commercial\b", normalised_id):
        return (LicenseRestrictionLevel.NON_COMMERCIAL, False, False, True, True)
    if re.search(r"\b(gpl|copyleft|share[- ]?alike)\b", normalised_id):
        return (LicenseRestrictionLevel.COPYLEFT, False, True, True, True)
    if re.search(r"\b(rail|openrail|open[- ]rail)\b", normalised_id):
        return (LicenseRestrictionLevel.CONDITIONAL, False, True, True, True)
    if re.search(r"\b(proprietary|commercial|all[- ]rights[- ]reserved)\b", normalised_id):
        return (LicenseRestrictionLevel.PROPRIETARY, False, False, False, False)
    if re.search(r"\b(mit|bsd|apache|isc|unlicense|public[- ]domain)\b", normalised_id):
        return (LicenseRestrictionLevel.PERMISSIVE, False, True, True, True)

    return None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def check_license(raw_license: str | None) -> LicenseReport:
    """Check a license string and generate a structured compliance report.

    This is the primary entry point for license compliance checking. It
    accepts a raw license string (SPDX identifier or free-form text) from
    a model card and returns a :class:`LicenseReport` containing:

    - Normalised SPDX-like identifier.
    - Restriction level classification.
    - Boolean flags for commercial use, redistribution, and attribution.
    - Compliance notes for EU AI Act and NIST RMF.
    - Actionable remediation suggestions.

    Args:
        raw_license: The raw license string from the model card, or ``None``
            if no license information is available.

    Returns:
        A fully populated :class:`LicenseReport`.
    """
    if raw_license is None or not str(raw_license).strip():
        return _build_unknown_report(raw_license)

    normalised = normalise_license_id(str(raw_license))
    db_entry = _lookup_license(normalised)

    if db_entry is None:
        # Unknown license — treat conservatively.
        logger.debug("License '%s' (normalised: '%s') not found in knowledge base.", raw_license, normalised)
        return _build_unknown_report(raw_license, spdx_id=normalised)

    restriction_level, is_osi, allows_commercial, allows_redistribution, requires_attribution = db_entry

    is_restricted = restriction_level != LicenseRestrictionLevel.PERMISSIVE

    # Generate compliance notes based on restriction level.
    notes: list[ComplianceNote] = []
    if restriction_level == LicenseRestrictionLevel.PERMISSIVE:
        notes = _notes_for_permissive(normalised)
    elif restriction_level == LicenseRestrictionLevel.COPYLEFT:
        notes = _notes_for_copyleft(normalised)
    elif restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL:
        notes = _notes_for_non_commercial(normalised)
    elif restriction_level == LicenseRestrictionLevel.CONDITIONAL:
        notes = _notes_for_conditional(normalised)
    elif restriction_level == LicenseRestrictionLevel.PROPRIETARY:
        notes = _notes_for_proprietary(normalised)
    else:
        notes = _notes_for_unknown()

    summary = _build_summary(normalised, restriction_level, allows_commercial)
    remediation = _build_remediation(restriction_level, normalised)

    return LicenseReport(
        spdx_id=normalised,
        raw_license=raw_license,
        restriction_level=restriction_level,
        is_restricted=is_restricted,
        is_osi_approved=is_osi,
        allows_commercial_use=allows_commercial,
        allows_redistribution=allows_redistribution,
        requires_attribution=requires_attribution,
        compliance_notes=notes,
        summary=summary,
        remediation_notes=remediation,
    )


def check_license_from_card(
    card_info: "ModelCardInfo",  # type: ignore[name-defined]  # noqa: F821
) -> LicenseReport:
    """Check the license of a model from its :class:`~model_provenance.fetcher.ModelCardInfo`.

    Convenience wrapper around :func:`check_license` that extracts the
    license field from a ``ModelCardInfo`` object and also inspects the
    model tags for a ``license:`` tag as a fallback.

    Args:
        card_info: A :class:`~model_provenance.fetcher.ModelCardInfo` object
            as returned by the fetcher module.

    Returns:
        A fully populated :class:`LicenseReport`.
    """
    license_str: str | None = getattr(card_info, "license", None)

    # Fallback: scan tags for a 'license:...' tag.
    if not license_str:
        tags: list[str] = getattr(card_info, "tags", []) or []
        for tag in tags:
            if tag.lower().startswith("license:"):
                license_str = tag[len("license:"):].strip()
                break

    return check_license(license_str)


def list_known_licenses() -> list[dict[str, object]]:
    """Return a list of all licenses in the knowledge base.

    Useful for tooling that needs to enumerate or display the supported
    license identifiers.

    Returns:
        A list of dictionaries, each with keys ``'spdx_id'``,
        ``'restriction_level'``, ``'is_osi_approved'``,
        ``'allows_commercial_use'``, ``'allows_redistribution'``,
        and ``'requires_attribution'``.
    """
    result: list[dict[str, object]] = []
    for spdx_id, (restriction, is_osi, commercial, redistrib, attribution) in sorted(
        _LICENSE_DB.items()
    ):
        result.append(
            {
                "spdx_id": spdx_id,
                "restriction_level": restriction.value,
                "is_osi_approved": is_osi,
                "allows_commercial_use": commercial,
                "allows_redistribution": redistrib,
                "requires_attribution": attribution,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_unknown_report(
    raw_license: str | None,
    spdx_id: str | None = None,
) -> LicenseReport:
    """Build a :class:`LicenseReport` for an unknown or absent license.

    Args:
        raw_license: The raw license string (may be ``None``).
        spdx_id: The normalised ID if one could be derived, else ``None``.

    Returns:
        A :class:`LicenseReport` with ``UNKNOWN`` restriction level.
    """
    effective_spdx = spdx_id if spdx_id else "unknown"
    notes = _notes_for_unknown()
    summary = (
        f"License unknown ('{raw_license}') — cannot verify compliance. "
        "Treat as all rights reserved."
        if raw_license
        else "No license specified — treat as all rights reserved."
    )
    remediation = [
        "Identify the license before deploying this model.",
        "Contact the model author to clarify the license terms.",
        "Consider replacing with a model that has an explicit open-source license.",
    ]
    return LicenseReport(
        spdx_id=effective_spdx,
        raw_license=raw_license,
        restriction_level=LicenseRestrictionLevel.UNKNOWN,
        is_restricted=True,
        is_osi_approved=False,
        allows_commercial_use=False,
        allows_redistribution=False,
        requires_attribution=False,
        compliance_notes=notes,
        summary=summary,
        remediation_notes=remediation,
    )


def _build_summary(
    spdx_id: str,
    restriction_level: LicenseRestrictionLevel,
    allows_commercial: bool,
) -> str:
    """Build a human-readable one-line summary for a license report.

    Args:
        spdx_id: Normalised license identifier.
        restriction_level: Assessed restriction level.
        allows_commercial: Whether commercial use is permitted.

    Returns:
        A single descriptive string.
    """
    level_label = restriction_level.value.replace("_", "-").title()
    commercial_str = "commercial use allowed" if allows_commercial else "commercial use NOT allowed"
    return f"{spdx_id} — {level_label} — {commercial_str}."


def _build_remediation(
    restriction_level: LicenseRestrictionLevel,
    spdx_id: str,
) -> list[str]:
    """Build a list of remediation notes for a given restriction level.

    Args:
        restriction_level: The assessed :class:`LicenseRestrictionLevel`.
        spdx_id: Normalised license identifier.

    Returns:
        List of actionable remediation strings.
    """
    if restriction_level == LicenseRestrictionLevel.PERMISSIVE:
        return [
            f"License '{spdx_id}' is permissive — no specific license-related "
            "action required for deployment."
        ]
    elif restriction_level == LicenseRestrictionLevel.COPYLEFT:
        return [
            f"Review copyleft obligations for '{spdx_id}' before releasing "
            "derivative works.",
            "Consult legal counsel if integrating this model into proprietary software.",
        ]
    elif restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL:
        return [
            f"License '{spdx_id}' prohibits commercial use. "
            "Obtain a commercial license before production deployment.",
            "Contact the model author or licensor for commercial licensing terms.",
            "Consider replacing with a permissively licensed alternative.",
        ]
    elif restriction_level == LicenseRestrictionLevel.CONDITIONAL:
        return [
            f"Review the specific use-case restrictions in the '{spdx_id}' license.",
            "Ensure your application is not listed in the prohibited uses section.",
            "Document compliance with conditional terms in your AI system risk register.",
        ]
    elif restriction_level == LicenseRestrictionLevel.PROPRIETARY:
        return [
            f"Obtain and review the full text of the '{spdx_id}' license.",
            "Engage legal counsel for a compliance assessment before deployment.",
            "Request explicit written permission for your intended use case if unclear.",
        ]
    else:
        return [
            "Identify the license before deploying this model.",
            "Contact the model author to clarify the license terms.",
        ]
