"""Tests for the welding inspection AI chatbot assistant.

All OpenAI API calls are mocked — no real API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.chatbot.assistant import (
    DISCLAIMER,
    ChatMessage,
    ChatSession,
    WeldingAssistant,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_mock_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def make_assistant(api_key: str = "test-key-123") -> WeldingAssistant:
    return WeldingAssistant(api_key=api_key)


# ── WeldingAssistant instantiation ────────────────────────────────────────────


class TestWeldingAssistantInit:
    def test_init_with_explicit_key(self):
        assistant = WeldingAssistant(api_key="sk-test")
        assert assistant._api_key == "sk-test"

    def test_init_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        assistant = WeldingAssistant()
        assert assistant._api_key == "sk-env-key"

    def test_init_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            WeldingAssistant()

    def test_init_raises_empty_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            WeldingAssistant()

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assistant = WeldingAssistant(api_key="sk-explicit")
        assert assistant._api_key == "sk-explicit"


# ── ChatSession ───────────────────────────────────────────────────────────────


class TestChatSession:
    def test_create_session_stores_context(self):
        assistant = make_assistant()
        session = assistant.create_session(
            defect_type="crack",
            severity="critical",
            recommended_action="Immediate repair required",
        )
        assert session.inspection_context["defect_type"] == "crack"
        assert session.inspection_context["severity"] == "critical"
        assert session.inspection_context["recommended_action"] == "Immediate repair required"

    def test_create_session_empty_history(self):
        assistant = make_assistant()
        session = assistant.create_session("porosity", "medium", "Document and monitor")
        assert session.messages == []

    def test_create_session_with_session_id(self):
        assistant = make_assistant()
        session = assistant.create_session(
            "undercut", "high", "NDE required", session_id="weld-abc-123"
        )
        assert session.session_id == "weld-abc-123"

    def test_add_message_appends(self):
        session = ChatSession(inspection_context={})
        session.add_message("user", "What does this defect mean?")
        session.add_message("assistant", "This indicates a structural risk.")
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_to_openai_messages_includes_system(self):
        session = ChatSession(inspection_context={})
        session.add_message("user", "What should I do?")
        msgs = session.to_openai_messages("You are a welding expert.")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a welding expert."
        assert msgs[1]["role"] == "user"

    def test_to_openai_messages_ordering(self):
        session = ChatSession(inspection_context={})
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")
        msgs = session.to_openai_messages("sys")
        assert len(msgs) == 4
        assert msgs[1]["content"] == "Q1"
        assert msgs[2]["content"] == "A1"
        assert msgs[3]["content"] == "Q2"

    def test_chat_message_dataclass(self):
        msg = ChatMessage(role="user", content="test message")
        assert msg.role == "user"
        assert msg.content == "test message"


# ── Chat method ───────────────────────────────────────────────────────────────


class TestChatMethod:
    def _make_session(self) -> ChatSession:
        assistant = make_assistant()
        return assistant.create_session(
            defect_type="crack",
            severity="critical",
            recommended_action="Reject weld, immediate repair",
        )

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_returns_string(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "Cracks are critical per AWS D1.1. " + DISCLAIMER
        )
        assistant = make_assistant()
        session = self._make_session()
        reply = assistant.chat(session, "Why is crack so serious?")
        assert isinstance(reply, str)
        assert len(reply) > 0

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_always_contains_disclaimer(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "This crack requires immediate attention."
        )
        assistant = make_assistant()
        session = self._make_session()
        reply = assistant.chat(session, "Explain the findings")
        assert DISCLAIMER in reply

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_does_not_duplicate_disclaimer(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            f"Some response. {DISCLAIMER}"
        )
        assistant = make_assistant()
        session = self._make_session()
        reply = assistant.chat(session, "test")
        assert reply.count(DISCLAIMER) == 1

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_adds_user_message_to_history(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "ok " + DISCLAIMER
        )
        assistant = make_assistant()
        session = self._make_session()
        assistant.chat(session, "What repair procedure is needed?")
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "What repair procedure is needed?"

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_adds_assistant_message_to_history(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "Repair by gouging and re-welding. " + DISCLAIMER
        )
        assistant = make_assistant()
        session = self._make_session()
        reply = assistant.chat(session, "question")
        assert session.messages[1].role == "assistant"
        assert reply in session.messages[1].content or session.messages[1].content in reply

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_multi_turn_history_grows(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "answer " + DISCLAIMER
        )
        assistant = make_assistant()
        session = self._make_session()
        assistant.chat(session, "Q1")
        assistant.chat(session, "Q2")
        assert len(session.messages) == 4  # user, assistant, user, assistant

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_passes_context_in_system_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "ok " + DISCLAIMER
        )
        assistant = make_assistant()
        session = self._make_session()
        assistant.chat(session, "test")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or (
            call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        )
        system_content = messages[0]["content"]
        assert "crack" in system_content
        assert "critical" in system_content

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_api_failure_raises_runtime_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("connection error")
        assistant = make_assistant()
        session = self._make_session()
        with pytest.raises(RuntimeError, match="AI assistant unavailable"):
            assistant.chat(session, "test")

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_chat_uses_gpt4o_mini_model(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "ok " + DISCLAIMER
        )
        assistant = make_assistant()
        session = self._make_session()
        assistant.chat(session, "test")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("model") == "gpt-4o-mini"


# ── System prompt ─────────────────────────────────────────────────────────────


class TestSystemPrompt:
    def test_system_prompt_includes_defect_type(self):
        assistant = make_assistant()
        ctx = {
            "defect_type": "porosity",
            "severity": "medium",
            "recommended_action": "Document and monitor",
        }
        prompt = assistant._build_system_prompt(ctx)
        assert "porosity" in prompt

    def test_system_prompt_includes_severity(self):
        assistant = make_assistant()
        ctx = {
            "defect_type": "incomplete_fusion",
            "severity": "high",
            "recommended_action": "NDE required",
        }
        prompt = assistant._build_system_prompt(ctx)
        assert "high" in prompt

    def test_system_prompt_includes_disclaimer(self):
        assistant = make_assistant()
        ctx = {
            "defect_type": "no_defect",
            "severity": "none",
            "recommended_action": "Pass",
        }
        prompt = assistant._build_system_prompt(ctx)
        assert DISCLAIMER in prompt

    def test_get_disclaimer_returns_standard_text(self):
        assistant = make_assistant()
        d = assistant.get_disclaimer()
        assert "certified welding inspector" in d
        assert "AI-assisted" in d


# ── Welding domain questions ──────────────────────────────────────────────────


class TestWeldingDomainQuestions:
    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_aws_d11_standards_question(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "AWS D1.1 requires immediate rejection of cracks. " + DISCLAIMER
        )
        assistant = make_assistant()
        session = assistant.create_session("crack", "critical", "Reject weld")
        reply = assistant.chat(session, "What does AWS D1.1 say about cracks?")
        assert DISCLAIMER in reply

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_repair_procedure_question(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "Porosity repair involves gouging out the defect area. " + DISCLAIMER
        )
        assistant = make_assistant()
        session = assistant.create_session("porosity", "medium", "Document and monitor")
        reply = assistant.chat(session, "How do I repair porosity?")
        assert DISCLAIMER in reply

    @patch("app.chatbot.assistant.WeldingAssistant._get_client")
    def test_severity_implications_question(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_openai_response(
            "High severity undercut creates stress concentrations. " + DISCLAIMER
        )
        assistant = make_assistant()
        session = assistant.create_session("undercut", "high", "NDE required")
        reply = assistant.chat(session, "What are the structural implications?")
        assert DISCLAIMER in reply
