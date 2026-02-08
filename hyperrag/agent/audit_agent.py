from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Generator

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from hyperrag.models.schemas import AuditFinding, AuditReport, PDFRect

log = logging.getLogger("hyperrag.agent")


class AuditAgent:
    """LangGraph ReAct agent specialised in audit analysis."""

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list,
        prompts_dir: str = "prompts",
        max_iterations: int = 10,
    ):
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._system_prompt = self._load_prompt(prompts_dir)
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_prompt,
        )

    @staticmethod
    def _load_prompt(prompts_dir: str) -> str:
        path = Path(prompts_dir) / "audit_system.txt"
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Run (blocking)
    # ------------------------------------------------------------------
    def run(self, query: str) -> AuditReport:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": self._max_iterations * 2},
        )

        final_message = result["messages"][-1].content
        return self._parse_report(query, final_message)

    # ------------------------------------------------------------------
    # Stream (for Streamlit)
    # ------------------------------------------------------------------
    def stream(self, query: str) -> Generator[dict, None, None]:
        """Yields intermediate steps for real-time display."""
        log.info("=" * 60)
        log.info(f"[Agent] Audit query: {query}")
        log.info("=" * 60)

        step_num = 0
        for event in self._agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": self._max_iterations * 2},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            step_num += 1
                            tool_info = f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})"
                            log.info(f"[Agent] Step {step_num}: TOOL CALL -> {tool_info}")
                            yield {
                                "type": "tool_call",
                                "content": f"Calling {tool_info}",
                            }
                    elif hasattr(msg, "content") and msg.content:
                        if node_name == "tools":
                            preview = msg.content[:200].replace("\n", " ")
                            log.info(f"[Agent] Step {step_num}: TOOL RESULT <- {preview}...")
                            yield {
                                "type": "tool_result",
                                "content": msg.content[:500],
                            }
                        else:
                            preview = msg.content[:200].replace("\n", " ")
                            log.info(f"[Agent] THINKING: {preview}...")
                            yield {
                                "type": "thinking",
                                "content": msg.content,
                            }

        log.info("[Agent] Audit complete.")
        log.info("=" * 60)
        yield {"type": "final", "content": "Audit complete."}

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------
    def _parse_report(self, query: str, raw: str) -> AuditReport:
        # Try to extract JSON from the response
        try:
            # Strip markdown fences if present
            text = raw
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            data = json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            # Fallback: treat entire response as summary
            return AuditReport(
                query=query,
                findings=[],
                summary=raw[:2000],
            )

        findings = []
        for f in data.get("findings", []):
            source_locations = []
            for sp in f.get("source_pages", []):
                source_locations.append(
                    PDFRect(
                        x0=0, y0=0, x1=0, y1=0,
                        page_num=sp.get("page_num", 0),
                    )
                )

            findings.append(
                AuditFinding(
                    finding_id=f.get("finding_id", uuid.uuid4().hex[:6]),
                    description=f.get("description", ""),
                    severity=f.get("severity", "low"),
                    evidence=f.get("evidence", []),
                    source_locations=source_locations,
                    related_entities=f.get("related_entities", []),
                )
            )

        return AuditReport(
            query=query,
            findings=findings,
            summary=data.get("summary", ""),
        )
