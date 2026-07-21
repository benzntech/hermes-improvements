#!/usr/bin/env python3
"""
Adaptive Workflow — Task Complexity Classification

Provides: AdaptiveWorkflow class that classifies user tasks by complexity
and selects appropriate workflows.

Complexity levels:
  - TRIVIAL:  Single-step, no tools needed
  - SIMPLE:   1-3 tools, straightforward
  - MODERATE: Multi-step, 3-8 tools, some planning needed
  - COMPLEX:  Multi-file changes, research required, 8+ tools

Workflows:
  - TRIVIAL: Direct response, no structure
  - SIMPLE:  Brief plan → execute → report
  - MODERATE: Analyze → plan → execute → verify → report
  - COMPLEX:  Analyze → research → plan → execute → verify → iterate → report
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class WorkflowMetrics:
    """Track workflow execution metrics for adaptive improvement."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def record(
        self,
        complexity: str,
        tool_count: int,
        duration_ms: float,
        success: bool,
    ):
        self._history.append({
            "complexity": complexity,
            "tool_count": tool_count,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": time.time(),
        })

        # Keep only last 100 entries
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated workflow statistics."""
        if not self._history:
            return {"entries": 0}

        by_complexity = {}
        for entry in self._history:
            c = entry["complexity"]
            if c not in by_complexity:
                by_complexity[c] = []
            by_complexity[c].append(entry)

        stats = {"entries": len(self._history), "by_complexity": {}}

        for complexity, entries in by_complexity.items():
            avg_tools = sum(e["tool_count"] for e in entries) / len(entries)
            avg_duration = sum(e["duration_ms"] for e in entries) / len(entries)
            success_rate = sum(1 for e in entries if e["success"]) / len(entries)

            stats["by_complexity"][complexity] = {
                "count": len(entries),
                "avg_tools": round(avg_tools, 1),
                "avg_duration_ms": round(avg_duration, 0),
                "success_rate": round(success_rate, 2),
            }

        return stats


class AdaptiveWorkflow:
    """
    Classify task complexity and provide appropriate workflow steps.

    Uses heuristics based on:
      - Message length and keyword density
      - Presence of multi-step indicators ("first...then", "steps", etc.)
      - Domain-specific complexity markers (code review, deployment, etc.)
    """

    # Complexity indicators (EN + RO)
    _COMPLEX_INDICATORS = [
        r"\b(review|audit|refactor|migrate|deploy|orchestr)\w*\b",
        r"\b(multiple files?|several|all|every)\b.*\b(file|module|component)\b",
        r"\b(production|security|critical|breaking)\b",
        r"\b(analyze|investigate|diagnose|debug)\b.*\b(complex|deep|thorough)\b",
        # Romanian — with \w* for inflected forms (securitatea, producția, etc.)
        # deploy is already in EN pattern (line 104), not duplicated here
        r"\b(review|audit|migrare|refactorizare|repozitor)\w*\b",
        r"\b(mai multe|toate|fiecare)\b.*\b(fișier[e]?|modul[e]?|funcții?|component[a-ză]?)\b",
        r"\b(producție|securitate|critic|avarie|protecție)\w*\b",
        r"\b(analizează|investighează|diagnostic|depanare|debughează)\b",
    ]

    _MODERATE_INDICATORS = [
        r"\b(create|build|implement|generate|write)\b.*\b(script|function|class)\b",
        r"\b(search|find|look up|research)\b",
        r"\b(explain|describe|summarize|document)\b",
        r"\b(configure|set up|install|setup)\b",
        r"\b(compare|versus|vs\.?)\b",
        # Romanian — with context (verb + object)
        r"\b(creează|construiește|implementează|generează|scrie)\b.*\b(script|funcții?|clasă|pagină|site|teste?|aplicație)\b",
        r"\b(test[e]?|testare|verific[ăa])\b",
        r"\b(caută|găsește|cercetează|caut)\b",
        r"\b(explică|descrie|sumarizează|documentează|prezintă)\b",
        r"\b(configurare|instalare|configurează|instalează)\b",
        r"\b(compară|versus|diferență)\b",
        r"\b(fă|făcut|execută|rulează|creează)\b",
        r"\b(cum|de ce|care|ce fel)\b",
        # Romanian — standalone verbs with simple object (write a X, do a Y)
        r"\b(scrie|fă|creează|adaugă|șterge|modifică|trimite|rezolvă)\b\s+\w+",
    ]

    _TRIVIAL_INDICATORS = [
        r"^(hi|hey|hello|yo|sup)\b",
        r"\b(thanks|thank you|ok|okay|got it|noted)\b",
        r"^(what|who|when|where)\b.*\?$",
        r"^\w+$",  # Single word
        # Romanian
        r"^(salut|bună|noroc|servus|hei)\b",
        r"\b(mersi|mulțumesc|ok|bine|notat|înțeles)\b",
        r"^(ce|cine|când|unde)\b.*\?$",
    ]

    def __init__(self):
        self.metrics = WorkflowMetrics()

    def classify_task(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskComplexity:
        """
        Classify a user message by task complexity.

        Args:
            message: The user's message text
            context: Optional context dictionary (previous messages, etc.)

        Returns:
            TaskComplexity enum value
        """
        if not message or not message.strip():
            return TaskComplexity.TRIVIAL

        message_lower = message.lower().strip()

        # Check for explicit complexity markers ("This is complex", etc.)
        if self._has_explicit_complexity_flag(message_lower):
            return TaskComplexity.COMPLEX

        # Score by indicators
        score = self._score_complexity(message_lower)

        # Adjust by message length
        word_count = len(message_lower.split())
        if word_count > 200:
            score += 1.5
        elif word_count > 100:
            score += 0.75
        elif word_count < 10:
            score -= 0.1

        # Adjust by multi-step markers
        if self._has_multi_step_markers(message_lower):
            score += 1.0

        # Boost for multiple Romanian action verbs (each extra verb = more steps)
        verb_count = self._count_ro_action_verbs(message_lower)
        if verb_count >= 3:
            score += 1.5
        elif verb_count == 2:
            score += 0.75
        elif verb_count == 1:
            score += 0.35  # single action verb → at least SIMPLE

        # Check for code-related patterns
        code_blocks = message.count("```")
        if code_blocks >= 2:
            score += 0.75

        # Classify
        if score >= 2.3:
            return TaskComplexity.COMPLEX
        elif score >= 1.1:
            return TaskComplexity.MODERATE
        elif score >= 0.3:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.TRIVIAL

    def _has_explicit_complexity_flag(self, text: str) -> bool:
        """Detect explicit complexity markers in text."""
        markers = [
            "this is complex",
            "this is complicated",
            "complex task",
            "difficult problem",
            "this is hard",
            "big task",
            "major change",
        ]
        return any(m in text for m in markers)

    def _score_complexity(self, text: str) -> float:
        """Score text for complexity based on regex patterns.
        Counts ALL matches per pattern using findall, not just the first."""
        score = 0.0

        for pattern in self._COMPLEX_INDICATORS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Count actual matched groups — each match adds 0.75
                count = len(matches) if isinstance(matches, list) else 1
                score += min(count * 0.75, 2.0)  # Cap at 2.0 per pattern line

        for pattern in self._MODERATE_INDICATORS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                count = len(matches) if isinstance(matches, list) else 1
                score += min(count * 0.4, 1.5)  # Cap at 1.5 per pattern line

        # Trivial indicators reduce score
        for pattern in self._TRIVIAL_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3

        return max(0.0, score)

    # Romanian imperative verbs that signal distinct action steps
    _RO_ACTION_VERBS = (
        r"analizează|repară|optimizează|creează|rulează|verifică|testează|"
        r"configurează|implementează|construiește|scrie|fă|instalează|"
        r"șterge|adaugă|modifică|actualizează|migrează|refactorizează|"
        r"deployează|trimite|raportează|documentează|rezolvă|depanează"
    )

    def _has_multi_step_markers(self, text: str) -> bool:
        """Check for explicit multi-step workflow markers (EN + RO)."""
        markers = [
            r"\bfirst\b.*\bthen\b",
            r"\bstep[s]?\b.*\d+",
            r"\b\d+\b.*\bstep[s]?\b",
            r"\b(?:after|before|next|finally|lastly)\b.*\b(when|once)\b",
            r"\blist\b.*\bstep[s]?\b",
            # Romanian — two action verbs connected by "și" (and) / comma
            rf"\b({self._RO_ACTION_VERBS})\b.{{0,60}}(?:și|,)\s*(?:{self._RO_ACTION_VERBS})\b",
            r"\b(paș[i]?|etapă|etape)\b.*\d+",
            r"\b\d+\b.*\b(paș[i]?|etapă|etape)\b",
            r"\b(întâi|mai întâi|apoi|după aceea|în final|în continuare|mai departe)\b",
        ]
        return any(re.search(m, text, re.IGNORECASE) for m in markers)

    def _count_ro_action_verbs(self, text: str) -> int:
        """Count distinct Romanian action verbs — more verbs = more complex task."""
        return len(re.findall(
            rf"\b(?:{self._RO_ACTION_VERBS})\b", text, re.IGNORECASE
        ))

    def get_workflow_steps(self, complexity: TaskComplexity) -> List[str]:
        """
        Return the workflow steps appropriate for a given complexity level.

        Args:
            complexity: Task complexity classification

        Returns:
            Ordered list of workflow step names
        """
        workflows = {
            TaskComplexity.TRIVIAL: [
                "Respond directly — no planning needed",
            ],
            TaskComplexity.SIMPLE: [
                "1. Understand the request",
                "2. Execute the task directly",
                "3. Report the result concisely",
            ],
            TaskComplexity.MODERATE: [
                "1. Analyze the task and identify key components",
                "2. Plan the approach (choose tools/framework)",
                "3. Execute step by step",
                "4. Verify the output",
                "5. Report with summary",
            ],
            TaskComplexity.COMPLEX: [
                "1. Deep analysis — understand all requirements",
                "2. Research — gather context and dependencies",
                "3. Design — plan architecture and approach",
                "4. Execute in phases — implement core first",
                "5. Verify each phase — test incrementally",
                "6. Iterate — refine based on verification",
                "7. Document — provide comprehensive report",
            ],
        }

        return workflows.get(complexity, workflows[TaskComplexity.SIMPLE])

    def execute_workflow(
        self,
        message: str,
        complexity: Optional[TaskComplexity] = None,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Classify and prepare a workflow plan for a user message.

        Args:
            message: User's message
            complexity: Pre-classified complexity (auto-detected if None)
            context: Additional context

        Returns:
            Workflow plan dictionary with steps and metadata
        """
        if complexity is None:
            complexity = self.classify_task(message, context)

        steps = self.get_workflow_steps(complexity)

        return {
            "complexity": complexity.value,
            "steps": steps,
            "estimated_tools": self._estimate_tool_count(complexity),
            "requires_planning": complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.MODERATE,
            ),
            "requires_verification": complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.MODERATE,
            ),
        }

    def _estimate_tool_count(self, complexity: TaskComplexity) -> Tuple[int, int]:
        """Estimate min/max tool calls for a complexity level."""
        estimates = {
            TaskComplexity.TRIVIAL: (0, 1),
            TaskComplexity.SIMPLE: (1, 3),
            TaskComplexity.MODERATE: (3, 8),
            TaskComplexity.COMPLEX: (8, 50),
        }
        return estimates.get(complexity, (1, 3))
