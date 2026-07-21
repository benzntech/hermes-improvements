#!/usr/bin/env python3
"""
Transparent Reasoning System — Chain of Thought + Confidence Scoring

Provides:
  - ReasoningTracer: Step-by-step reasoning trace
  - SourceAttribution: Track and cite information sources
  - UncertaintyManager: Register and report known uncertainties
  - inject_reasoning_trace: Append reasoning to responses

Makes the agent's decision process transparent for the user.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────

class ConfidenceLevel(Enum):
    """Confidence level enumeration."""
    CERTAIN = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    GUESS = 0.1
    UNKNOWN = 0.0


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_num: int
    action: str  # "analyze", "search", "decide", "implement", etc
    input: str
    reasoning: str  # Why this step
    output: str  # What was concluded
    confidence: float  # 0.0-1.0
    sources: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)


@dataclass
class ConfidenceClaim:
    """A factual claim with confidence score."""
    claim: str
    confidence: float
    reasoning: str
    sources: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


# ─── ReasoningTracer ─────────────────────────────────────────────────

class ReasoningTracer:
    """
    Tracks the reasoning process step-by-step.
    Builds a visible trace for inclusion in response.
    """

    def __init__(self, include_in_output: bool = True):
        self.steps: List[ReasoningStep] = []
        self.claims: List[ConfidenceClaim] = []
        self.include_in_output = include_in_output
        self._step_counter = 0

    def add_step(
        self,
        action: str,
        input_data: str,
        reasoning: str,
        output: str,
        confidence: float = 0.5,
        sources: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
    ) -> ReasoningStep:
        """Record a reasoning step."""
        self._step_counter += 1

        step = ReasoningStep(
            step_num=self._step_counter,
            action=action,
            input=input_data,
            reasoning=reasoning,
            output=output,
            confidence=min(1.0, max(0.0, confidence)),
            sources=sources or [],
            uncertainties=uncertainties or [],
        )

        self.steps.append(step)
        logger.debug(
            "Reasoning step %d: %s (conf: %.0%%)",
            self._step_counter, action, step.confidence
        )
        return step

    def claim(
        self,
        claim: str,
        confidence: float,
        reasoning: str,
        sources: Optional[List[str]] = None,
        caveats: Optional[List[str]] = None,
    ) -> ConfidenceClaim:
        """Record a factual claim with confidence score."""
        fact = ConfidenceClaim(
            claim=claim,
            confidence=min(1.0, max(0.0, confidence)),
            reasoning=reasoning,
            sources=sources or [],
            caveats=caveats or [],
        )
        self.claims.append(fact)
        return fact

    def get_trace_markdown(self) -> str:
        """Generate markdown representation of reasoning trace."""
        if not self.steps:
            return ""

        lines = ["## Reasoning Trace\n"]

        for step in self.steps:
            conf_emoji = self._confidence_emoji(step.confidence)
            lines.append(f"### Step {step.step_num}: {step.action} {conf_emoji}")
            lines.append(f"**Input:** {step.input}")
            lines.append(f"**Reasoning:** {step.reasoning}")
            lines.append(f"**Output:** {step.output}")
            lines.append(f"**Confidence:** {step.confidence:.0%}")

            if step.sources:
                lines.append(f"**Sources:** {', '.join(step.sources)}")

            if step.uncertainties:
                lines.append(f"**Uncertainties:** {'; '.join(step.uncertainties)}")

            lines.append("")

        return "\n".join(lines)

    def get_claims_markdown(self) -> str:
        """Generate markdown for claims with confidence."""
        if not self.claims:
            return ""

        lines = ["## Claims & Confidence\n"]

        for claim in self.claims:
            conf_emoji = self._confidence_emoji(claim.confidence)
            conf_text = self._confidence_text(claim.confidence)

            lines.append(f"**{conf_emoji} {conf_text}:** {claim.claim}")

            if claim.reasoning:
                lines.append(f"→ {claim.reasoning}")

            if claim.sources:
                lines.append(f"📚 {', '.join(claim.sources)}")

            for caveat in claim.caveats:
                lines.append(f"⚠️  {caveat}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _confidence_emoji(confidence: float) -> str:
        if confidence >= 0.9:
            return "✅"
        elif confidence >= 0.7:
            return "🟢"
        elif confidence >= 0.5:
            return "🟡"
        elif confidence >= 0.3:
            return "🟠"
        return "❌"

    @staticmethod
    def _confidence_text(confidence: float) -> str:
        if confidence >= 0.9:
            return "Certain"
        elif confidence >= 0.7:
            return "High confidence"
        elif confidence >= 0.5:
            return "Medium confidence"
        elif confidence >= 0.3:
            return "Low confidence"
        return "Uncertain"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.steps and not self.claims:
            return {}

        step_conf = [s.confidence for s in self.steps]
        claim_conf = [c.confidence for c in self.claims]
        all_conf = step_conf + claim_conf

        avg = sum(all_conf) / len(all_conf) if all_conf else 0.0
        weak_steps = [s for s in self.steps if s.confidence < 0.5]
        weak_claims = [c for c in self.claims if c.confidence < 0.5]

        return {
            "total_steps": len(self.steps),
            "total_claims": len(self.claims),
            "average_confidence": avg,
            "weak_points": len(weak_steps) + len(weak_claims),
            "high_confidence_items": sum(1 for x in all_conf if x >= 0.8),
            "needs_verification": (
                [s.output for s in weak_steps] +
                [c.claim for c in weak_claims]
            ),
        }

    def clear(self):
        """Reset tracer for a new query."""
        self.steps = []
        self.claims = []
        self._step_counter = 0


# ─── SourceAttribution ────────────────────────────────────────────────

class SourceAttribution:
    """Track and cite information sources."""

    def __init__(self):
        self.sources: Dict[str, Dict[str, Any]] = {}
        self._source_counter = 0

    def add_source(
        self,
        name: str,
        source_type: str,  # "web", "memory", "tool", "reasoning"
        url: Optional[str] = None,
        date_accessed: Optional[str] = None,
        reliability: float = 0.7,
    ) -> str:
        """Register a source, return its ID."""
        self._source_counter += 1
        source_id = f"src{self._source_counter}"

        self.sources[source_id] = {
            "name": name,
            "type": source_type,
            "url": url,
            "date_accessed": date_accessed,
            "reliability": reliability,
        }
        return source_id

    def cite(self, *source_ids: str) -> str:
        """Get citations for sources by ID."""
        citations = []
        for sid in source_ids:
            if sid in self.sources:
                source = self.sources[sid]
                if source.get("url"):
                    citations.append(f"[{source['name']}]({source['url']})")
                else:
                    citations.append(source["name"])
        return ", ".join(citations)

    def get_bibliography(self) -> str:
        """Generate full bibliography."""
        if not self.sources:
            return ""

        lines = ["## Sources\n"]
        for sid, source in sorted(self.sources.items()):
            rel = source.get("reliability", 0.7)
            emoji = "🟢" if rel >= 0.8 else "🟡" if rel >= 0.6 else "🔴"
            lines.append(f"{sid}. {emoji} **{source['name']}** ({source['type']})")
            if source.get("url"):
                lines.append(f"   {source['url']}")
            if source.get("date_accessed"):
                lines.append(f"   Accessed: {source['date_accessed']}")
            lines.append("")
        return "\n".join(lines)


# ─── UncertaintyManager ───────────────────────────────────────────────

class UncertaintyManager:
    """Track and communicate known uncertainties."""

    def __init__(self):
        self.uncertainties: List[Dict[str, Any]] = []

    def add_uncertainty(
        self,
        aspect: str,
        reason: str,
        impact: str,  # "high", "medium", "low"
        mitigation: Optional[str] = None,
    ):
        """Register an uncertainty."""
        self.uncertainties.append({
            "aspect": aspect,
            "reason": reason,
            "impact": impact,
            "mitigation": mitigation,
        })

    def get_uncertainty_report(self) -> str:
        """Generate report of known uncertainties."""
        if not self.uncertainties:
            return ""

        high = [u for u in self.uncertainties if u["impact"] == "high"]
        medium = [u for u in self.uncertainties if u["impact"] == "medium"]

        lines = ["## Known Uncertainties\n"]

        if high:
            lines.append("### High Impact")
            for u in high:
                lines.append(f"- **{u['aspect']}:** {u['reason']}")
                if u.get("mitigation"):
                    lines.append(f"  → {u['mitigation']}")
            lines.append("")

        if medium:
            lines.append("### Medium Impact")
            for u in medium:
                lines.append(f"- {u['aspect']}: {u['reason']}")
            lines.append("")

        return "\n".join(lines)


# ─── Injection Helper ─────────────────────────────────────────────────

def inject_reasoning_trace(
    response: str,
    tracer: ReasoningTracer,
    include_claims: bool = True,
    include_summary: bool = True,
) -> str:
    """
    Append reasoning trace to response.
    Makes decision process transparent.
    """
    additions = []

    trace_md = tracer.get_trace_markdown()
    if trace_md:
        additions.append(trace_md)

    if include_claims:
        claims_md = tracer.get_claims_markdown()
        if claims_md:
            additions.append(claims_md)

    if include_summary:
        summary = tracer.get_summary()
        if summary:
            additions.append("\n## Confidence Summary\n")
            additions.append(
                f"- **Average Confidence:** {summary['average_confidence']:.0%}"
            )
            additions.append(
                f"- **Weak Points:** {summary['weak_points']}"
            )
            additions.append(
                f"- **High Confidence Items:** {summary['high_confidence_items']}"
            )

    if additions:
        return f"{response}\n\n---\n{''.join(additions)}"

    return response
