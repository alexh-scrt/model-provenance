"""Unit tests for model_provenance.license_check module.

Covers license parsing, restriction level classification, compliance flag
generation for EU AI Act and NIST RMF, known-license database lookups,
alias normalisation, and the check_license_from_card convenience function.
"""

from __future__ import annotations

import pytest

from model_provenance.license_check import (
    ComplianceFramework,
    ComplianceNote,
    LicenseReport,
    LicenseRestrictionLevel,
    _build_remediation,
    _build_summary,
    _build_unknown_report,
    _lookup_license,
    _notes_for_conditional,
    _notes_for_copyleft,
    _notes_for_non_commercial,
    _notes_for_permissive,
    _notes_for_proprietary,
    _notes_for_unknown,
    check_license,
    check_license_from_card,
    list_known_licenses,
    normalise_license_id,
)
from model_provenance.fetcher import ModelCardInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(license_str: str | None = None, tags: list[str] | None = None) -> ModelCardInfo:
    """Build a minimal ModelCardInfo for testing."""
    return ModelCardInfo(
        model_id="test/model",
        license=license_str,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# ComplianceNote
# ---------------------------------------------------------------------------


class TestComplianceNote:
    def test_to_dict_keys(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="warning",
            title="Test note",
            detail="Test detail",
            remediation="Take action.",
        )
        d = note.to_dict()
        assert set(d.keys()) == {"framework", "severity", "title", "detail", "remediation"}

    def test_to_dict_framework_is_string(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="info",
            title="t",
            detail="d",
        )
        assert note.to_dict()["framework"] == "EU AI Act"

    def test_to_dict_severity_preserved(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="critical",
            title="t",
            detail="d",
        )
        assert note.to_dict()["severity"] == "critical"

    def test_to_dict_remediation_default_empty(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.GENERAL,
            severity="info",
            title="t",
            detail="d",
        )
        assert note.to_dict()["remediation"] == ""

    def test_all_frameworks_have_string_values(self) -> None:
        for fw in ComplianceFramework:
            assert isinstance(fw.value, str)

    def test_all_restriction_levels_have_string_values(self) -> None:
        for level in LicenseRestrictionLevel:
            assert isinstance(level.value, str)


# ---------------------------------------------------------------------------
# LicenseReport
# ---------------------------------------------------------------------------


class TestLicenseReport:
    def _make(
        self,
        spdx_id: str = "mit",
        restriction_level: LicenseRestrictionLevel = LicenseRestrictionLevel.PERMISSIVE,
        notes: list[ComplianceNote] | None = None,
        is_restricted: bool = False,
        is_osi_approved: bool = True,
        allows_commercial: bool = True,
        allows_redistribution: bool = True,
        requires_attribution: bool = True,
    ) -> LicenseReport:
        return LicenseReport(
            spdx_id=spdx_id,
            raw_license=spdx_id,
            restriction_level=restriction_level,
            is_restricted=is_restricted,
            is_osi_approved=is_osi_approved,
            allows_commercial_use=allows_commercial,
            allows_redistribution=allows_redistribution,
            requires_attribution=requires_attribution,
            compliance_notes=notes or [],
            summary=f"{spdx_id} — test",
            remediation_notes=[],
        )

    def test_has_warnings_false_when_no_notes(self) -> None:
        report = self._make()
        assert not report.has_warnings

    def test_has_warnings_true_for_warning_note(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="warning",
            title="t",
            detail="d",
        )
        report = self._make(notes=[note])
        assert report.has_warnings

    def test_has_warnings_true_for_critical_note(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="critical",
            title="t",
            detail="d",
        )
        report = self._make(notes=[note])
        assert report.has_warnings

    def test_has_warnings_false_for_info_only(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.NIST_RMF,
            severity="info",
            title="t",
            detail="d",
        )
        report = self._make(notes=[note])
        assert not report.has_warnings

    def test_has_critical_false_when_no_notes(self) -> None:
        report = self._make()
        assert not report.has_critical

    def test_has_critical_true_for_critical_note(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="critical",
            title="t",
            detail="d",
        )
        report = self._make(notes=[note])
        assert report.has_critical

    def test_has_critical_false_for_warning_only(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="warning",
            title="t",
            detail="d",
        )
        report = self._make(notes=[note])
        assert not report.has_critical

    def test_to_dict_keys(self) -> None:
        report = self._make()
        d = report.to_dict()
        expected_keys = {
            "spdx_id",
            "raw_license",
            "restriction_level",
            "is_restricted",
            "is_osi_approved",
            "allows_commercial_use",
            "allows_redistribution",
            "requires_attribution",
            "has_warnings",
            "has_critical",
            "summary",
            "compliance_notes",
            "remediation_notes",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_restriction_level_is_string(self) -> None:
        report = self._make(restriction_level=LicenseRestrictionLevel.COPYLEFT)
        d = report.to_dict()
        assert d["restriction_level"] == "copyleft"

    def test_to_dict_compliance_notes_is_list(self) -> None:
        note = ComplianceNote(
            framework=ComplianceFramework.EU_AI_ACT,
            severity="info",
            title="t",
            detail="d",
        )
        report = self._make(notes=[note])
        d = report.to_dict()
        assert isinstance(d["compliance_notes"], list)
        assert len(d["compliance_notes"]) == 1

    def test_to_dict_booleans_correct(self) -> None:
        report = self._make(
            is_restricted=True,
            is_osi_approved=False,
            allows_commercial=False,
            allows_redistribution=True,
            requires_attribution=True,
        )
        d = report.to_dict()
        assert d["is_restricted"] is True
        assert d["is_osi_approved"] is False
        assert d["allows_commercial_use"] is False
        assert d["allows_redistribution"] is True
        assert d["requires_attribution"] is True

    def test_to_dict_spdx_id_preserved(self) -> None:
        report = self._make(spdx_id="apache-2.0")
        assert report.to_dict()["spdx_id"] == "apache-2.0"


# ---------------------------------------------------------------------------
# normalise_license_id
# ---------------------------------------------------------------------------


class TestNormaliseLicenseId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("MIT", "mit"),
            ("Apache-2.0", "apache-2.0"),
            ("apache2", "apache-2.0"),
            ("Apache 2.0", "apache-2.0"),
            ("Apache2.0", "apache-2.0"),
            ("GPL-3.0", "gpl-3.0"),
            ("gplv3", "gpl-3.0"),
            ("gplv2", "gpl-2.0"),
            ("MIT License", "mit"),
            ("The MIT License", "mit"),
            ("BSD", "bsd-3-clause"),
            ("cc-by", "cc-by-4.0"),
            ("cc-by-nc", "cc-by-nc-4.0"),
            ("cc-by-sa", "cc-by-sa-4.0"),
            ("cc0", "cc0-1.0"),
            ("public domain", "cc0-1.0"),
            ("openrail-m", "openrail"),
            ("llama 2", "llama2"),
            ("llama 3", "llama3"),
            ("Meta Llama 3", "llama3"),
            ("  mit  ", "mit"),  # whitespace stripped
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert normalise_license_id(raw) == expected

    def test_unknown_license_lowercased(self) -> None:
        result = normalise_license_id("Some-Custom-License-X")
        assert result == result.lower()

    def test_already_normalised_passthrough(self) -> None:
        assert normalise_license_id("apache-2.0") == "apache-2.0"
        assert normalise_license_id("mit") == "mit"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert normalise_license_id("  MIT  ") == "mit"

    def test_returns_string(self) -> None:
        result = normalise_license_id("MIT")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _lookup_license
# ---------------------------------------------------------------------------


class TestLookupLicense:
    def test_returns_tuple_for_known_license(self) -> None:
        result = _lookup_license("mit")
        assert result is not None
        assert len(result) == 5

    def test_mit_is_permissive(self) -> None:
        result = _lookup_license("mit")
        assert result is not None
        restriction_level, is_osi, allows_commercial, allows_redistribution, requires_attribution = result
        assert restriction_level == LicenseRestrictionLevel.PERMISSIVE

    def test_mit_is_osi_approved(self) -> None:
        result = _lookup_license("mit")
        assert result is not None
        _, is_osi, _, _, _ = result
        assert is_osi is True

    def test_apache_2_0_is_permissive(self) -> None:
        result = _lookup_license("apache-2.0")
        assert result is not None
        restriction_level, _, allows_commercial, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert allows_commercial is True

    def test_gpl_3_0_is_copyleft(self) -> None:
        result = _lookup_license("gpl-3.0")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.COPYLEFT

    def test_cc_by_nc_is_non_commercial(self) -> None:
        result = _lookup_license("cc-by-nc-4.0")
        assert result is not None
        restriction_level, _, allows_commercial, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL
        assert allows_commercial is False

    def test_openrail_is_conditional(self) -> None:
        result = _lookup_license("openrail")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.CONDITIONAL

    def test_proprietary_is_proprietary(self) -> None:
        result = _lookup_license("proprietary")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.PROPRIETARY

    def test_unknown_license_returns_none(self) -> None:
        result = _lookup_license("totally-made-up-license-xyz")
        assert result is None

    def test_non_commercial_pattern_match(self) -> None:
        result = _lookup_license("custom-non-commercial-license")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL

    def test_gpl_pattern_match(self) -> None:
        result = _lookup_license("some-gpl-license")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.COPYLEFT

    def test_rail_pattern_match(self) -> None:
        result = _lookup_license("custom-openrail-license")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.CONDITIONAL

    def test_mit_pattern_match(self) -> None:
        result = _lookup_license("mit-style-license")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.PERMISSIVE

    def test_llama2_is_conditional(self) -> None:
        result = _lookup_license("llama2")
        assert result is not None
        restriction_level, _, _, _, _ = result
        assert restriction_level == LicenseRestrictionLevel.CONDITIONAL

    def test_cc0_is_permissive_no_attribution(self) -> None:
        result = _lookup_license("cc0-1.0")
        assert result is not None
        restriction_level, _, allows_commercial, allows_redistribution, requires_attribution = result
        assert restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert allows_commercial is True
        assert requires_attribution is False


# ---------------------------------------------------------------------------
# check_license — permissive licenses
# ---------------------------------------------------------------------------


class TestCheckLicensePermissive:
    @pytest.mark.parametrize(
        "license_id",
        ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "cc0-1.0"],
    )
    def test_permissive_licenses(self, license_id: str) -> None:
        report = check_license(license_id)
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE

    def test_mit_is_not_restricted(self) -> None:
        report = check_license("mit")
        assert not report.is_restricted

    def test_apache_allows_commercial(self) -> None:
        report = check_license("apache-2.0")
        assert report.allows_commercial_use is True

    def test_apache_is_osi_approved(self) -> None:
        report = check_license("apache-2.0")
        assert report.is_osi_approved is True

    def test_mit_allows_redistribution(self) -> None:
        report = check_license("mit")
        assert report.allows_redistribution is True

    def test_mit_requires_attribution(self) -> None:
        report = check_license("mit")
        assert report.requires_attribution is True

    def test_permissive_has_no_critical_notes(self) -> None:
        report = check_license("mit")
        critical_notes = [n for n in report.compliance_notes if n.severity == "critical"]
        assert len(critical_notes) == 0

    def test_permissive_has_info_notes(self) -> None:
        report = check_license("apache-2.0")
        info_notes = [n for n in report.compliance_notes if n.severity == "info"]
        assert len(info_notes) > 0

    def test_permissive_compliance_notes_not_empty(self) -> None:
        report = check_license("mit")
        assert len(report.compliance_notes) > 0

    def test_permissive_eu_ai_act_note_present(self) -> None:
        report = check_license("apache-2.0")
        eu_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.EU_AI_ACT
        ]
        assert len(eu_notes) > 0

    def test_permissive_nist_rfm_note_present(self) -> None:
        report = check_license("mit")
        nist_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.NIST_RMF
        ]
        assert len(nist_notes) > 0

    def test_permissive_summary_contains_id(self) -> None:
        report = check_license("mit")
        assert "mit" in report.summary

    def test_permissive_remediation_notes_populated(self) -> None:
        report = check_license("apache-2.0")
        assert len(report.remediation_notes) > 0

    def test_raw_license_preserved(self) -> None:
        report = check_license("Apache-2.0")
        assert report.raw_license == "Apache-2.0"

    def test_cc_by_4_0_is_permissive(self) -> None:
        report = check_license("cc-by-4.0")
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert report.allows_commercial_use is True

    def test_unlicense_no_attribution_required(self) -> None:
        report = check_license("unlicense")
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert report.requires_attribution is False


# ---------------------------------------------------------------------------
# check_license — copyleft licenses
# ---------------------------------------------------------------------------


class TestCheckLicenseCopyleft:
    @pytest.mark.parametrize(
        "license_id",
        ["gpl-2.0", "gpl-3.0", "lgpl-2.1", "agpl-3.0", "mpl-2.0", "cc-by-sa-4.0"],
    )
    def test_copyleft_licenses(self, license_id: str) -> None:
        report = check_license(license_id)
        assert report.restriction_level == LicenseRestrictionLevel.COPYLEFT

    def test_gpl_3_is_restricted(self) -> None:
        report = check_license("gpl-3.0")
        assert report.is_restricted is True

    def test_gpl_3_allows_commercial(self) -> None:
        report = check_license("gpl-3.0")
        assert report.allows_commercial_use is True

    def test_gpl_3_is_osi_approved(self) -> None:
        report = check_license("gpl-3.0")
        assert report.is_osi_approved is True

    def test_copyleft_has_warning_notes(self) -> None:
        report = check_license("gpl-3.0")
        warning_notes = [n for n in report.compliance_notes if n.severity == "warning"]
        assert len(warning_notes) > 0

    def test_copyleft_no_critical_notes(self) -> None:
        report = check_license("gpl-2.0")
        critical_notes = [n for n in report.compliance_notes if n.severity == "critical"]
        assert len(critical_notes) == 0

    def test_copyleft_eu_ai_act_note_present(self) -> None:
        report = check_license("gpl-3.0")
        eu_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.EU_AI_ACT
        ]
        assert len(eu_notes) > 0

    def test_copyleft_nist_note_present(self) -> None:
        report = check_license("agpl-3.0")
        nist_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.NIST_RMF
        ]
        assert len(nist_notes) > 0

    def test_copyleft_remediation_populated(self) -> None:
        report = check_license("gpl-3.0")
        assert len(report.remediation_notes) > 0

    def test_copyleft_summary_contains_level(self) -> None:
        report = check_license("gpl-3.0")
        assert "copyleft" in report.summary.lower() or "Copyleft" in report.summary

    def test_cc_by_sa_copyleft(self) -> None:
        report = check_license("cc-by-sa-4.0")
        assert report.restriction_level == LicenseRestrictionLevel.COPYLEFT
        assert report.allows_commercial_use is True

    def test_lgpl_is_osi_approved(self) -> None:
        report = check_license("lgpl-2.1")
        assert report.is_osi_approved is True


# ---------------------------------------------------------------------------
# check_license — non-commercial licenses
# ---------------------------------------------------------------------------


class TestCheckLicenseNonCommercial:
    @pytest.mark.parametrize(
        "license_id",
        [
            "cc-by-nc-4.0",
            "cc-by-nc-3.0",
            "cc-by-nc-sa-4.0",
            "cc-by-nc-nd-4.0",
            "cc-by-nd-4.0",
        ],
    )
    def test_non_commercial_licenses(self, license_id: str) -> None:
        report = check_license(license_id)
        assert report.restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL

    def test_cc_by_nc_prohibits_commercial(self) -> None:
        report = check_license("cc-by-nc-4.0")
        assert report.allows_commercial_use is False

    def test_cc_by_nc_is_restricted(self) -> None:
        report = check_license("cc-by-nc-4.0")
        assert report.is_restricted is True

    def test_cc_by_nc_not_osi_approved(self) -> None:
        report = check_license("cc-by-nc-4.0")
        assert report.is_osi_approved is False

    def test_non_commercial_has_critical_notes(self) -> None:
        report = check_license("cc-by-nc-4.0")
        critical_notes = [n for n in report.compliance_notes if n.severity == "critical"]
        assert len(critical_notes) > 0

    def test_non_commercial_eu_ai_act_critical(self) -> None:
        report = check_license("cc-by-nc-4.0")
        eu_critical = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.EU_AI_ACT
            and n.severity == "critical"
        ]
        assert len(eu_critical) > 0

    def test_non_commercial_nist_critical(self) -> None:
        report = check_license("cc-by-nc-4.0")
        nist_critical = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.NIST_RMF
            and n.severity == "critical"
        ]
        assert len(nist_critical) > 0

    def test_non_commercial_general_note_present(self) -> None:
        report = check_license("cc-by-nc-sa-4.0")
        general_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.GENERAL
        ]
        assert len(general_notes) > 0

    def test_non_commercial_remediation_populated(self) -> None:
        report = check_license("cc-by-nc-4.0")
        assert len(report.remediation_notes) > 0
        assert any("commercial" in n.lower() for n in report.remediation_notes)

    def test_cc_by_nc_nd_prohibits_redistribution(self) -> None:
        report = check_license("cc-by-nc-nd-4.0")
        assert report.allows_redistribution is False

    def test_cc_by_nd_allows_commercial_but_no_redistribution(self) -> None:
        report = check_license("cc-by-nd-4.0")
        assert report.restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL
        assert report.allows_redistribution is False

    def test_non_commercial_summary_contains_not_allowed(self) -> None:
        report = check_license("cc-by-nc-4.0")
        assert "NOT" in report.summary or "not" in report.summary.lower()

    def test_has_critical_true_for_non_commercial(self) -> None:
        report = check_license("cc-by-nc-4.0")
        assert report.has_critical is True


# ---------------------------------------------------------------------------
# check_license — conditional/RAIL licenses
# ---------------------------------------------------------------------------


class TestCheckLicenseConditional:
    @pytest.mark.parametrize(
        "license_id",
        ["openrail", "openrail++", "creativeml-openrail-m", "llama2", "llama3", "gemma"],
    )
    def test_conditional_licenses(self, license_id: str) -> None:
        report = check_license(license_id)
        assert report.restriction_level == LicenseRestrictionLevel.CONDITIONAL

    def test_openrail_is_restricted(self) -> None:
        report = check_license("openrail")
        assert report.is_restricted is True

    def test_openrail_allows_commercial(self) -> None:
        report = check_license("openrail")
        assert report.allows_commercial_use is True

    def test_openrail_not_osi_approved(self) -> None:
        report = check_license("openrail")
        assert report.is_osi_approved is False

    def test_conditional_has_warning_notes(self) -> None:
        report = check_license("openrail")
        warning_notes = [n for n in report.compliance_notes if n.severity == "warning"]
        assert len(warning_notes) > 0

    def test_conditional_eu_ai_act_warning(self) -> None:
        report = check_license("llama2")
        eu_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.EU_AI_ACT
        ]
        assert len(eu_notes) > 0

    def test_llama2_has_mau_threshold_note(self) -> None:
        report = check_license("llama2")
        mau_notes = [
            n for n in report.compliance_notes
            if "700" in n.detail or "monthly" in n.detail.lower()
        ]
        assert len(mau_notes) > 0

    def test_llama3_has_mau_threshold_note(self) -> None:
        report = check_license("llama3")
        mau_notes = [
            n for n in report.compliance_notes
            if "700" in n.detail or "monthly" in n.detail.lower()
        ]
        assert len(mau_notes) > 0

    def test_gemma_has_prohibited_use_note(self) -> None:
        report = check_license("gemma")
        gemma_notes = [
            n for n in report.compliance_notes
            if "gemma" in n.detail.lower() or "prohibited" in n.detail.lower()
        ]
        assert len(gemma_notes) > 0

    def test_conditional_remediation_populated(self) -> None:
        report = check_license("openrail")
        assert len(report.remediation_notes) > 0

    def test_conditional_no_critical_notes_for_openrail(self) -> None:
        report = check_license("openrail")
        critical_notes = [n for n in report.compliance_notes if n.severity == "critical"]
        assert len(critical_notes) == 0

    def test_conditional_summary_contains_id(self) -> None:
        report = check_license("openrail")
        assert "openrail" in report.summary.lower()

    def test_creativeml_openrail_conditional(self) -> None:
        report = check_license("creativeml-openrail-m")
        assert report.restriction_level == LicenseRestrictionLevel.CONDITIONAL


# ---------------------------------------------------------------------------
# check_license — proprietary licenses
# ---------------------------------------------------------------------------


class TestCheckLicenseProprietary:
    @pytest.mark.parametrize(
        "license_id",
        ["proprietary", "other", "custom"],
    )
    def test_proprietary_licenses(self, license_id: str) -> None:
        report = check_license(license_id)
        assert report.restriction_level == LicenseRestrictionLevel.PROPRIETARY

    def test_proprietary_is_restricted(self) -> None:
        report = check_license("proprietary")
        assert report.is_restricted is True

    def test_proprietary_prohibits_commercial(self) -> None:
        report = check_license("proprietary")
        assert report.allows_commercial_use is False

    def test_proprietary_not_osi_approved(self) -> None:
        report = check_license("other")
        assert report.is_osi_approved is False

    def test_proprietary_has_critical_notes(self) -> None:
        report = check_license("proprietary")
        critical_notes = [n for n in report.compliance_notes if n.severity == "critical"]
        assert len(critical_notes) > 0

    def test_proprietary_eu_ai_act_critical(self) -> None:
        report = check_license("proprietary")
        eu_critical = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.EU_AI_ACT
            and n.severity == "critical"
        ]
        assert len(eu_critical) > 0

    def test_proprietary_nist_critical(self) -> None:
        report = check_license("custom")
        nist_critical = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.NIST_RMF
            and n.severity == "critical"
        ]
        assert len(nist_critical) > 0

    def test_proprietary_remediation_populated(self) -> None:
        report = check_license("proprietary")
        assert len(report.remediation_notes) > 0

    def test_has_critical_true_for_proprietary(self) -> None:
        report = check_license("proprietary")
        assert report.has_critical is True

    def test_proprietary_prohibits_redistribution(self) -> None:
        report = check_license("proprietary")
        assert report.allows_redistribution is False

    def test_other_is_proprietary_level(self) -> None:
        report = check_license("other")
        assert report.restriction_level == LicenseRestrictionLevel.PROPRIETARY


# ---------------------------------------------------------------------------
# check_license — unknown / absent licenses
# ---------------------------------------------------------------------------


class TestCheckLicenseUnknown:
    def test_none_license_returns_unknown(self) -> None:
        report = check_license(None)
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_empty_string_returns_unknown(self) -> None:
        report = check_license("")
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_whitespace_only_returns_unknown(self) -> None:
        report = check_license("   ")
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_unrecognised_license_returns_unknown(self) -> None:
        report = check_license("totally-unknown-xyz-license-v99")
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_unknown_is_restricted(self) -> None:
        report = check_license(None)
        assert report.is_restricted is True

    def test_unknown_prohibits_commercial(self) -> None:
        report = check_license(None)
        assert report.allows_commercial_use is False

    def test_unknown_not_osi_approved(self) -> None:
        report = check_license(None)
        assert report.is_osi_approved is False

    def test_unknown_has_warning_notes(self) -> None:
        report = check_license(None)
        warning_notes = [n for n in report.compliance_notes if n.severity == "warning"]
        assert len(warning_notes) > 0

    def test_unknown_eu_ai_act_note_present(self) -> None:
        report = check_license(None)
        eu_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.EU_AI_ACT
        ]
        assert len(eu_notes) > 0

    def test_unknown_nist_note_present(self) -> None:
        report = check_license(None)
        nist_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.NIST_RMF
        ]
        assert len(nist_notes) > 0

    def test_unknown_general_note_present(self) -> None:
        report = check_license(None)
        general_notes = [
            n for n in report.compliance_notes
            if n.framework == ComplianceFramework.GENERAL
        ]
        assert len(general_notes) > 0

    def test_unknown_spdx_id_when_none(self) -> None:
        report = check_license(None)
        assert report.spdx_id == "unknown"

    def test_unknown_raw_license_preserved(self) -> None:
        report = check_license(None)
        assert report.raw_license is None

    def test_raw_license_preserved_for_unrecognised(self) -> None:
        report = check_license("mystery-license-v3")
        assert report.raw_license == "mystery-license-v3"

    def test_unknown_remediation_populated(self) -> None:
        report = check_license(None)
        assert len(report.remediation_notes) > 0

    def test_unknown_summary_mentions_treat_as(self) -> None:
        report = check_license(None)
        assert "all rights reserved" in report.summary.lower()

    def test_unknown_has_warnings_true(self) -> None:
        report = check_license(None)
        assert report.has_warnings is True

    def test_unknown_prohibits_redistribution(self) -> None:
        report = check_license(None)
        assert report.allows_redistribution is False


# ---------------------------------------------------------------------------
# check_license — alias resolution via check_license
# ---------------------------------------------------------------------------


class TestCheckLicenseAliasResolution:
    @pytest.mark.parametrize(
        "alias,expected_level",
        [
            ("Apache 2.0", LicenseRestrictionLevel.PERMISSIVE),
            ("apache2", LicenseRestrictionLevel.PERMISSIVE),
            ("GPL v3", LicenseRestrictionLevel.UNKNOWN),  # Not in alias table; falls to unknown
            ("gplv3", LicenseRestrictionLevel.COPYLEFT),
            ("BSD License", LicenseRestrictionLevel.PERMISSIVE),
            ("cc-by-nc", LicenseRestrictionLevel.NON_COMMERCIAL),
            ("cc0", LicenseRestrictionLevel.PERMISSIVE),
            ("open-rail", LicenseRestrictionLevel.CONDITIONAL),
            ("llama 2", LicenseRestrictionLevel.CONDITIONAL),
        ],
    )
    def test_alias_resolution(self, alias: str, expected_level: LicenseRestrictionLevel) -> None:
        report = check_license(alias)
        assert report.restriction_level == expected_level

    def test_uppercase_mit_resolved(self) -> None:
        report = check_license("MIT")
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert report.spdx_id == "mit"

    def test_mixed_case_apache_resolved(self) -> None:
        report = check_license("Apache-2.0")
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert report.spdx_id == "apache-2.0"


# ---------------------------------------------------------------------------
# check_license_from_card
# ---------------------------------------------------------------------------


class TestCheckLicenseFromCard:
    def test_uses_license_field(self) -> None:
        card = _card(license_str="apache-2.0")
        report = check_license_from_card(card)
        assert report.spdx_id == "apache-2.0"
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE

    def test_none_license_returns_unknown(self) -> None:
        card = _card(license_str=None)
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_falls_back_to_license_tag(self) -> None:
        card = _card(license_str=None, tags=["pytorch", "license:mit"])
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE
        assert report.spdx_id == "mit"

    def test_license_field_takes_precedence_over_tag(self) -> None:
        # If license field is set, it should be used instead of tags.
        card = _card(license_str="apache-2.0", tags=["license:gpl-3.0"])
        report = check_license_from_card(card)
        assert report.spdx_id == "apache-2.0"

    def test_tag_license_prefix_stripped(self) -> None:
        card = _card(license_str=None, tags=["license:cc-by-nc-4.0"])
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.NON_COMMERCIAL

    def test_no_license_field_no_tag_returns_unknown(self) -> None:
        card = _card(license_str=None, tags=["pytorch", "text-generation"])
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_gpl_license_from_card(self) -> None:
        card = _card(license_str="gpl-3.0")
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.COPYLEFT

    def test_returns_license_report_instance(self) -> None:
        card = _card(license_str="mit")
        report = check_license_from_card(card)
        assert isinstance(report, LicenseReport)

    def test_empty_tags_list_no_fallback(self) -> None:
        card = _card(license_str=None, tags=[])
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_uppercase_license_tag_resolved(self) -> None:
        card = _card(license_str=None, tags=["license:MIT"])
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.PERMISSIVE

    def test_proprietary_license_from_card(self) -> None:
        card = _card(license_str="other")
        report = check_license_from_card(card)
        assert report.restriction_level == LicenseRestrictionLevel.PROPRIETARY

    def test_compliance_notes_generated_from_card(self) -> None:
        card = _card(license_str="cc-by-nc-4.0")
        report = check_license_from_card(card)
        assert len(report.compliance_notes) > 0


# ---------------------------------------------------------------------------
# list_known_licenses
# ---------------------------------------------------------------------------


class TestListKnownLicenses:
    def test_returns_non_empty_list(self) -> None:
        licenses = list_known_licenses()
        assert len(licenses) > 0

    def test_each_entry_has_required_keys(self) -> None:
        licenses = list_known_licenses()
        for entry in licenses:
            assert "spdx_id" in entry
            assert "restriction_level" in entry
            assert "is_osi_approved" in entry
            assert "allows_commercial_use" in entry
            assert "allows_redistribution" in entry
            assert "requires_attribution" in entry

    def test_mit_in_list(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert "mit" in ids

    def test_apache_in_list(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert "apache-2.0" in ids

    def test_gpl_in_list(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert "gpl-3.0" in ids

    def test_cc_by_nc_in_list(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert "cc-by-nc-4.0" in ids

    def test_restriction_level_is_string(self) -> None:
        licenses = list_known_licenses()
        for entry in licenses:
            assert isinstance(entry["restriction_level"], str)

    def test_booleans_are_bool(self) -> None:
        licenses = list_known_licenses()
        for entry in licenses:
            assert isinstance(entry["is_osi_approved"], bool)
            assert isinstance(entry["allows_commercial_use"], bool)
            assert isinstance(entry["allows_redistribution"], bool)
            assert isinstance(entry["requires_attribution"], bool)

    def test_list_sorted_by_spdx_id(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert ids == sorted(ids)

    def test_openrail_in_list(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert "openrail" in ids

    def test_llama2_in_list(self) -> None:
        licenses = list_known_licenses()
        ids = [e["spdx_id"] for e in licenses]
        assert "llama2" in ids


# ---------------------------------------------------------------------------
# _notes_for_* generators
# ---------------------------------------------------------------------------


class TestNotesGenerators:
    def test_notes_for_permissive_returns_list(self) -> None:
        notes = _notes_for_permissive("mit")
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_notes_for_permissive_are_info(self) -> None:
        notes = _notes_for_permissive("apache-2.0")
        for note in notes:
            assert note.severity == "info"

    def test_notes_for_copyleft_returns_list(self) -> None:
        notes = _notes_for_copyleft("gpl-3.0")
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_notes_for_copyleft_are_warnings(self) -> None:
        notes = _notes_for_copyleft("gpl-3.0")
        for note in notes:
            assert note.severity == "warning"

    def test_notes_for_non_commercial_returns_list(self) -> None:
        notes = _notes_for_non_commercial("cc-by-nc-4.0")
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_notes_for_non_commercial_are_critical(self) -> None:
        notes = _notes_for_non_commercial("cc-by-nc-4.0")
        for note in notes:
            assert note.severity == "critical"

    def test_notes_for_conditional_returns_list(self) -> None:
        notes = _notes_for_conditional("openrail")
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_notes_for_conditional_base_are_warnings(self) -> None:
        notes = _notes_for_conditional("openrail")
        base_notes = notes[:2]  # First two are always the base warnings.
        for note in base_notes:
            assert note.severity == "warning"

    def test_notes_for_conditional_llama2_extra_note(self) -> None:
        notes = _notes_for_conditional("llama2")
        assert len(notes) > 2  # Additional Llama-specific note

    def test_notes_for_conditional_gemma_extra_note(self) -> None:
        notes = _notes_for_conditional("gemma")
        assert len(notes) > 2  # Additional Gemma-specific note

    def test_notes_for_proprietary_returns_list(self) -> None:
        notes = _notes_for_proprietary("proprietary")
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_notes_for_proprietary_are_critical(self) -> None:
        notes = _notes_for_proprietary("custom")
        for note in notes:
            assert note.severity == "critical"

    def test_notes_for_unknown_returns_list(self) -> None:
        notes = _notes_for_unknown()
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_notes_for_unknown_are_warnings(self) -> None:
        notes = _notes_for_unknown()
        for note in notes:
            assert note.severity == "warning"

    def test_all_notes_have_non_empty_title(self) -> None:
        all_note_lists = [
            _notes_for_permissive("mit"),
            _notes_for_copyleft("gpl-3.0"),
            _notes_for_non_commercial("cc-by-nc-4.0"),
            _notes_for_conditional("openrail"),
            _notes_for_proprietary("proprietary"),
            _notes_for_unknown(),
        ]
        for note_list in all_note_lists:
            for note in note_list:
                assert len(note.title) > 0

    def test_all_notes_have_non_empty_detail(self) -> None:
        all_note_lists = [
            _notes_for_permissive("mit"),
            _notes_for_copyleft("gpl-3.0"),
            _notes_for_non_commercial("cc-by-nc-4.0"),
            _notes_for_conditional("openrail"),
            _notes_for_proprietary("proprietary"),
            _notes_for_unknown(),
        ]
        for note_list in all_note_lists:
            for note in note_list:
                assert len(note.detail) > 0

    def test_all_notes_have_remediation(self) -> None:
        all_note_lists = [
            _notes_for_permissive("mit"),
            _notes_for_copyleft("gpl-3.0"),
            _notes_for_non_commercial("cc-by-nc-4.0"),
            _notes_for_conditional("openrail"),
            _notes_for_proprietary("proprietary"),
            _notes_for_unknown(),
        ]
        for note_list in all_note_lists:
            for note in note_list:
                assert len(note.remediation) > 0

    def test_each_note_set_covers_eu_ai_act(self) -> None:
        for note_list in [
            _notes_for_permissive("mit"),
            _notes_for_copyleft("gpl-3.0"),
            _notes_for_non_commercial("cc-by-nc-4.0"),
            _notes_for_conditional("openrail"),
            _notes_for_proprietary("proprietary"),
            _notes_for_unknown(),
        ]:
            frameworks = {n.framework for n in note_list}
            assert ComplianceFramework.EU_AI_ACT in frameworks


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_contains_spdx_id(self) -> None:
        summary = _build_summary("apache-2.0", LicenseRestrictionLevel.PERMISSIVE, True)
        assert "apache-2.0" in summary

    def test_contains_level(self) -> None:
        summary = _build_summary("gpl-3.0", LicenseRestrictionLevel.COPYLEFT, True)
        assert "copyleft" in summary.lower() or "Copyleft" in summary

    def test_commercial_allowed_text(self) -> None:
        summary = _build_summary("mit", LicenseRestrictionLevel.PERMISSIVE, True)
        assert "allowed" in summary.lower()
        assert "NOT" not in summary

    def test_commercial_not_allowed_text(self) -> None:
        summary = _build_summary("cc-by-nc-4.0", LicenseRestrictionLevel.NON_COMMERCIAL, False)
        assert "NOT" in summary

    def test_returns_string(self) -> None:
        summary = _build_summary("mit", LicenseRestrictionLevel.PERMISSIVE, True)
        assert isinstance(summary, str)

    def test_non_empty(self) -> None:
        summary = _build_summary("proprietary", LicenseRestrictionLevel.PROPRIETARY, False)
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# _build_remediation
# ---------------------------------------------------------------------------


class TestBuildRemediation:
    def test_permissive_has_one_note(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.PERMISSIVE, "mit")
        assert len(notes) >= 1

    def test_permissive_note_mentions_no_action(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.PERMISSIVE, "apache-2.0")
        assert any("no" in n.lower() or "not required" in n.lower() or "permissive" in n.lower() for n in notes)

    def test_copyleft_has_multiple_notes(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.COPYLEFT, "gpl-3.0")
        assert len(notes) >= 2

    def test_copyleft_mentions_copyleft(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.COPYLEFT, "gpl-3.0")
        assert any("copyleft" in n.lower() for n in notes)

    def test_non_commercial_has_multiple_notes(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.NON_COMMERCIAL, "cc-by-nc-4.0")
        assert len(notes) >= 2

    def test_non_commercial_mentions_commercial(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.NON_COMMERCIAL, "cc-by-nc-4.0")
        assert any("commercial" in n.lower() for n in notes)

    def test_conditional_has_multiple_notes(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.CONDITIONAL, "openrail")
        assert len(notes) >= 2

    def test_conditional_mentions_use_case(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.CONDITIONAL, "openrail")
        assert any("use" in n.lower() for n in notes)

    def test_proprietary_has_multiple_notes(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.PROPRIETARY, "proprietary")
        assert len(notes) >= 2

    def test_proprietary_mentions_legal(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.PROPRIETARY, "proprietary")
        assert any("legal" in n.lower() for n in notes)

    def test_unknown_returns_list(self) -> None:
        notes = _build_remediation(LicenseRestrictionLevel.UNKNOWN, "unknown")
        assert isinstance(notes, list)
        assert len(notes) > 0

    def test_all_notes_are_strings(self) -> None:
        for level in LicenseRestrictionLevel:
            notes = _build_remediation(level, "test-id")
            for note in notes:
                assert isinstance(note, str)


# ---------------------------------------------------------------------------
# _build_unknown_report
# ---------------------------------------------------------------------------


class TestBuildUnknownReport:
    def test_restriction_level_unknown(self) -> None:
        report = _build_unknown_report(None)
        assert report.restriction_level == LicenseRestrictionLevel.UNKNOWN

    def test_spdx_id_unknown_when_no_raw(self) -> None:
        report = _build_unknown_report(None)
        assert report.spdx_id == "unknown"

    def test_spdx_id_set_when_provided(self) -> None:
        report = _build_unknown_report("mystery-v2", spdx_id="mystery-v2")
        assert report.spdx_id == "mystery-v2"

    def test_raw_license_preserved(self) -> None:
        report = _build_unknown_report("some-unknown-license")
        assert report.raw_license == "some-unknown-license"

    def test_raw_license_none_preserved(self) -> None:
        report = _build_unknown_report(None)
        assert report.raw_license is None

    def test_is_restricted_true(self) -> None:
        report = _build_unknown_report(None)
        assert report.is_restricted is True

    def test_is_osi_approved_false(self) -> None:
        report = _build_unknown_report(None)
        assert report.is_osi_approved is False

    def test_allows_commercial_false(self) -> None:
        report = _build_unknown_report(None)
        assert report.allows_commercial_use is False

    def test_compliance_notes_populated(self) -> None:
        report = _build_unknown_report(None)
        assert len(report.compliance_notes) > 0

    def test_remediation_notes_populated(self) -> None:
        report = _build_unknown_report(None)
        assert len(report.remediation_notes) > 0

    def test_summary_non_empty(self) -> None:
        report = _build_unknown_report(None)
        assert len(report.summary) > 0

    def test_returns_license_report_instance(self) -> None:
        report = _build_unknown_report(None)
        assert isinstance(report, LicenseReport)


# ---------------------------------------------------------------------------
# Integration: full check_license pipeline
# ---------------------------------------------------------------------------


class TestCheckLicenseIntegration:
    def test_all_known_licenses_return_report(self) -> None:
        """All licenses in the knowledge base should return a valid report."""
        for entry in list_known_licenses():
            spdx_id = entry["spdx_id"]
            report = check_license(spdx_id)
            assert isinstance(report, LicenseReport)
            assert report.spdx_id == spdx_id
            assert isinstance(report.restriction_level, LicenseRestrictionLevel)
            assert len(report.compliance_notes) > 0

    def test_report_consistency_for_permissive(self) -> None:
        """Permissive licenses should never produce critical compliance notes."""
        permissive_ids = [
            e["spdx_id"]
            for e in list_known_licenses()
            if e["restriction_level"] == LicenseRestrictionLevel.PERMISSIVE.value
        ]
        for spdx_id in permissive_ids:
            report = check_license(spdx_id)
            critical = [n for n in report.compliance_notes if n.severity == "critical"]
            assert len(critical) == 0, f"{spdx_id} should not have critical notes"

    def test_report_consistency_for_non_commercial(self) -> None:
        """Non-commercial licenses should always produce critical notes."""
        nc_ids = [
            e["spdx_id"]
            for e in list_known_licenses()
            if e["restriction_level"] == LicenseRestrictionLevel.NON_COMMERCIAL.value
        ]
        for spdx_id in nc_ids:
            report = check_license(spdx_id)
            critical = [n for n in report.compliance_notes if n.severity == "critical"]
            assert len(critical) > 0, f"{spdx_id} should have critical notes"

    def test_report_consistency_for_proprietary(self) -> None:
        """Proprietary licenses should always produce critical notes."""
        prop_ids = [
            e["spdx_id"]
            for e in list_known_licenses()
            if e["restriction_level"] == LicenseRestrictionLevel.PROPRIETARY.value
        ]
        for spdx_id in prop_ids:
            report = check_license(spdx_id)
            critical = [n for n in report.compliance_notes if n.severity == "critical"]
            assert len(critical) > 0, f"{spdx_id} should have critical notes"

    def test_to_dict_fully_serialisable(self) -> None:
        """All reports should be fully serialisable to dict."""
        import json
        for entry in list_known_licenses()[:10]:  # Spot-check first 10.
            report = check_license(entry["spdx_id"])
            d = report.to_dict()
            # Should not raise.
            json_str = json.dumps(d)
            assert len(json_str) > 0

    def test_check_license_from_card_round_trip(self) -> None:
        """check_license_from_card should produce the same result as check_license."""
        for spdx_id in ["mit", "apache-2.0", "gpl-3.0", "cc-by-nc-4.0", "openrail"]:
            card = _card(license_str=spdx_id)
            report_from_card = check_license_from_card(card)
            report_direct = check_license(spdx_id)
            assert report_from_card.spdx_id == report_direct.spdx_id
            assert report_from_card.restriction_level == report_direct.restriction_level
            assert report_from_card.allows_commercial_use == report_direct.allows_commercial_use
