"""Welding inspection AI chatbot for interpreting weld defect results.

Provides a multi-turn conversational interface for welding engineers and
inspectors to ask questions about weld defect classifications, severity
grades, AWS D1.1 standards, and repair procedures.

DISCLAIMER: This is an AI-assisted tool. All findings must be confirmed by
a certified welding inspector.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "AI-assisted tool. All findings must be confirmed by certified welding inspector."
)

SYSTEM_PROMPT_TEMPLATE = """You are a welding inspection AI assistant specializing in weld defect interpretation and AWS D1.1 structural welding standards.
You help welding engineers and certified inspectors understand weld defect classifications, severity assessments, and repair procedures.

Current inspection context:
- Defect Type: {defect_type}
- Severity: {severity}
- Recommended Action: {recommended_action}

Your role:
- Answer questions about weld defect findings in clear, technical language
- Explain defect types and their implications for structural integrity
- Reference AWS D1.1 Structural Welding Code and ISO 5817 acceptance criteria
- Advise on appropriate repair procedures and NDE (Non-Destructive Examination) methods
- Explain severity levels and what they mean for weld acceptance or rejection
- Discuss contributing factors such as heat input, technique, and base material issues

Important constraints:
- You are an AI assistant, NOT a substitute for a certified welding inspector (CWI)
- Always remind users that findings require confirmation by a qualified inspector
- Do not make final acceptance or rejection decisions
- When uncertain, recommend consultation with a certified welding engineer (CWE)
- Keep responses technically accurate and concise

Disclaimer: {disclaimer}
"""


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatSession:
    """Tracks conversation history for a multi-turn dialogue."""

    inspection_context: dict[str, Any]
    messages: list[ChatMessage] = field(default_factory=list)
    session_id: str = ""

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))

    def to_openai_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Convert session history to OpenAI message format."""
        result = [{"role": "system", "content": system_prompt}]
        for msg in self.messages:
            result.append({"role": msg.role, "content": msg.content})
        return result


class WeldingAssistant:
    """Welding inspection AI assistant for weld defect result interpretation.

    Uses OpenAI GPT-4o-mini to answer engineer/inspector questions about
    weld defect results. Maintains conversation history for multi-turn dialogue.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """

    MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise OSError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before using the WeldingAssistant."
            )
        self._client: Any = None  # lazy-initialized

    def _get_client(self) -> Any:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415

                self._client = OpenAI(api_key=self._api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from exc
        return self._client

    def _build_system_prompt(self, inspection_context: dict[str, Any]) -> str:
        """Build the system prompt with inspection context injected."""
        return SYSTEM_PROMPT_TEMPLATE.format(
            defect_type=inspection_context.get("defect_type", "Not available"),
            severity=inspection_context.get("severity", "Not available"),
            recommended_action=inspection_context.get(
                "recommended_action", "Not available"
            ),
            disclaimer=DISCLAIMER,
        )

    def create_session(
        self,
        defect_type: str,
        severity: str,
        recommended_action: str,
        session_id: str = "",
    ) -> ChatSession:
        """Create a new chat session with inspection context.

        Args:
            defect_type: Detected defect label (e.g., "crack", "porosity").
            severity: Severity level string (e.g., "critical", "medium").
            recommended_action: Recommended corrective action from the report.
            session_id: Optional opaque session identifier.

        Returns:
            A new ChatSession ready for multi-turn conversation.
        """
        inspection_context = {
            "defect_type": defect_type,
            "severity": severity,
            "recommended_action": recommended_action,
        }
        return ChatSession(
            inspection_context=inspection_context, session_id=session_id
        )

    def chat(self, session: ChatSession, message: str) -> str:
        """Send a message and get a welding AI response.

        Args:
            session: The active ChatSession with inspection context.
            message: Engineer/inspector's natural language question.

        Returns:
            AI assistant response string, always includes disclaimer reminder.

        Raises:
            EnvironmentError: If API key is missing.
            RuntimeError: If the OpenAI API call fails.
        """
        session.add_message("user", message)

        system_prompt = self._build_system_prompt(session.inspection_context)
        messages = session.to_openai_messages(system_prompt)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.3,  # low temperature for technical accuracy
            )
            reply = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise RuntimeError(f"AI assistant unavailable: {exc}") from exc

        # Ensure disclaimer is always appended if not already present
        if DISCLAIMER not in reply:
            reply = f"{reply}\n\n---\n{DISCLAIMER}"

        session.add_message("assistant", reply)
        return reply

    def get_disclaimer(self) -> str:
        """Return the standard welding inspection disclaimer."""
        return DISCLAIMER
