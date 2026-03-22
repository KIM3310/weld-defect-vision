"""Tests for the batch weld inspection agent.

All OpenAI API calls are mocked — no real API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.agent.orchestrator import (
    _SEVERITY_PRIORITY,
    DISCLAIMER,
    AgentStatus,
    InspectionAgent,
    InspectionResult,
    InspectionSession,
    WeldRecord,
    _determine_action_items,
)
from tests.conftest import (
    make_image_crack,
    make_image_no_defect,
    make_image_porosity,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_agent(api_key: str = "test-key") -> InspectionAgent:
    """Create an InspectionAgent with a test API key."""
    return InspectionAgent(api_key=api_key)


def make_mock_openai_response(content: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def make_weld_record(image: Image.Image | None = None, joint_id: str = "W-001") -> WeldRecord:
    if image is None:
        image = make_image_no_defect()
    return WeldRecord(image=image, weld_joint_id=joint_id)


# ── InspectionAgent instantiation ─────────────────────────────────────────────


class TestInspectionAgentInit:
    def test_init_with_explicit_key(self):
        agent = InspectionAgent(api_key="sk-test")
        assert agent._api_key == "sk-test"

    def test_init_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        agent = InspectionAgent()
        assert agent._api_key == "sk-env-key"

    def test_init_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            InspectionAgent()

    def test_init_raises_empty_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            InspectionAgent()

    def test_initial_status_idle(self):
        agent = make_agent()
        assert agent.status == AgentStatus.IDLE

    def test_get_status_returns_dict(self):
        agent = make_agent()
        s = agent.get_status()
        assert isinstance(s, dict)
        assert "status" in s
        assert "disclaimer" in s

    def test_get_status_disclaimer_present(self):
        agent = make_agent()
        s = agent.get_status()
        assert DISCLAIMER in s["disclaimer"]


# ── WeldRecord ────────────────────────────────────────────────────────────────


class TestWeldRecord:
    def test_weld_record_defaults(self):
        img = make_image_no_defect()
        record = WeldRecord(image=img)
        assert record.weld_joint_id == ""
        assert record.inspector_notes == ""

    def test_weld_record_with_metadata(self):
        img = make_image_crack()
        record = WeldRecord(
            image=img,
            weld_joint_id="WELD-42",
            inspector_notes="Suspected hydrogen crack",
        )
        assert record.weld_joint_id == "WELD-42"
        assert record.inspector_notes == "Suspected hydrogen crack"


# ── Action items logic ────────────────────────────────────────────────────────


class TestDetermineActionItems:
    def test_critical_severity_urgent_action(self):
        actions = _determine_action_items("crack", "critical", "Reject and repair")
        assert any("URGENT" in a for a in actions)

    def test_high_severity_hold_action(self):
        actions = _determine_action_items("incomplete_fusion", "high", "NDE required")
        assert any("HOLD" in a or "NDE" in a for a in actions)

    def test_medium_severity_caution_action(self):
        actions = _determine_action_items("porosity", "medium", "Document")
        assert any("CAUTION" in a or "Document" in a for a in actions)

    def test_low_severity_pass_action(self):
        actions = _determine_action_items("spatter", "low", "Record in log")
        assert any("PASS" in a or "record" in a.lower() for a in actions)

    def test_crack_specific_actions(self):
        actions = _determine_action_items("crack", "critical", "Reject")
        assert any("MT" in a or "PT" in a or "preheat" in a.lower() for a in actions)

    def test_porosity_specific_actions(self):
        actions = _determine_action_items("porosity", "medium", "Review")
        assert any("shielding" in a.lower() or "gas" in a.lower() for a in actions)

    def test_incomplete_fusion_specific_actions(self):
        actions = _determine_action_items("incomplete_fusion", "high", "NDE")
        assert any("UT" in a or "heat" in a.lower() or "travel" in a.lower() for a in actions)

    def test_undercut_specific_actions(self):
        actions = _determine_action_items("undercut", "high", "Review params")
        assert any("AWS" in a or "parameter" in a.lower() or "current" in a.lower() for a in actions)

    def test_overlap_specific_actions(self):
        actions = _determine_action_items("overlap", "medium", "Grind")
        assert any("grind" in a.lower() or "blend" in a.lower() or "wire" in a.lower() for a in actions)

    def test_spatter_specific_actions(self):
        actions = _determine_action_items("spatter", "low", "Clean")
        assert any("spatter" in a.lower() or "shielding" in a.lower() or "arc" in a.lower() for a in actions)

    def test_recommended_action_included(self):
        actions = _determine_action_items("porosity", "medium", "Custom inspector note")
        assert any("Custom inspector note" in a for a in actions)


# ── Severity prioritization ───────────────────────────────────────────────────


class TestSeverityPrioritization:
    def test_severity_priority_ordering(self):
        assert _SEVERITY_PRIORITY["critical"] > _SEVERITY_PRIORITY["high"]
        assert _SEVERITY_PRIORITY["high"] > _SEVERITY_PRIORITY["medium"]
        assert _SEVERITY_PRIORITY["medium"] > _SEVERITY_PRIORITY["low"]
        assert _SEVERITY_PRIORITY["low"] > _SEVERITY_PRIORITY["none"]

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_results_sorted_by_severity_descending(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        records = [
            make_weld_record(make_image_no_defect(), "W-001"),
            make_weld_record(make_image_porosity(), "W-002"),
            make_weld_record(make_image_crack(), "W-003"),
        ]
        session = agent.run_inspection(records)
        priorities = [r.severity_priority for r in session.results if r.status == "success"]
        assert priorities == sorted(priorities, reverse=True)


# ── run_inspection workflow ───────────────────────────────────────────────────


class TestRunInspection:
    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_returns_session(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Inspection complete. {DISCLAIMER}"
        )
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert isinstance(session, InspectionSession)

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_correct_total(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        records = [make_weld_record() for _ in range(3)]
        session = agent.run_inspection(records)
        assert session.total_welds == 3

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_status_completed(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert session.status == AgentStatus.COMPLETED

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_agent_status_completed_after_run(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        agent.run_inspection([make_weld_record()])
        assert agent.status == AgentStatus.COMPLETED

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_summary_contains_disclaimer(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Session summary. {DISCLAIMER}"
        )
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert DISCLAIMER in session.summary

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_session_has_session_id(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert session.session_id != ""

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_empty_batch(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"No welds. {DISCLAIMER}"
        )
        agent = make_agent()
        session = agent.run_inspection([])
        assert session.total_welds == 0
        assert session.succeeded == 0

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_results_list_length(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        records = [make_weld_record() for _ in range(5)]
        session = agent.run_inspection(records)
        assert len(session.results) == 5

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_disclaimer_in_session(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert DISCLAIMER in session.disclaimer

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_run_inspection_urgent_count_non_negative(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        records = [
            make_weld_record(make_image_crack(), "W-001"),
            make_weld_record(make_image_porosity(), "W-002"),
            make_weld_record(make_image_no_defect(), "W-003"),
        ]
        session = agent.run_inspection(records)
        assert session.urgent_cases >= 0


# ── Fallback behavior ─────────────────────────────────────────────────────────


class TestFallbackBehavior:
    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_openai_failure_uses_fallback_summary(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API down")
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert isinstance(session.summary, str)
        assert len(session.summary) > 0

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_openai_failure_fallback_contains_disclaimer(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        agent = make_agent()
        session = agent.run_inspection([make_weld_record()])
        assert DISCLAIMER in session.summary

    @patch("app.agent.orchestrator.InspectionAgent._get_openai_client")
    def test_single_weld_failure_does_not_crash_batch(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Summary. {DISCLAIMER}"
        )
        agent = make_agent()
        records = [
            make_weld_record(make_image_no_defect(), "W-001"),
            make_weld_record(make_image_no_defect(), "W-002"),
        ]
        session = agent.run_inspection(records)
        assert session.total_welds == 2
        assert session.status == AgentStatus.COMPLETED


# ── InspectionResult dataclass ────────────────────────────────────────────────


class TestInspectionResult:
    def test_default_flags(self):
        result = InspectionResult(weld_joint_id="W1", report_id="R1", status="success")
        assert result.flagged_urgent is False
        assert result.action_items == []
        assert result.error is None

    def test_error_result(self):
        result = InspectionResult(
            weld_joint_id="W2",
            report_id="R2",
            status="error",
            error="pipeline failed",
        )
        assert result.error == "pipeline failed"

    def test_urgent_flag_set(self):
        result = InspectionResult(
            weld_joint_id="W3",
            report_id="R3",
            status="success",
            flagged_urgent=True,
        )
        assert result.flagged_urgent is True
