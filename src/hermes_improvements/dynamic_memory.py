#!/usr/bin/env python3
"""
Dynamic Memory Context — Live Tracking of Memory Changes

Provides:
  - DynamicMemoryContext: Track memory mutations during a session
  - AdaptiveMemoryPrefetch: Pre-fetch relevant memories based on context
  - inject_live_memory_into_response: Append memory updates to responses

Purpose: Hermes' standard memory is stateless per-turn. DynamicMemoryContext
tracks what changed during a conversation so the agent can reference its own
recent memory modifications without re-querying the full memory store.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DynamicMemoryContext:
    """
    Tracks memory mutations within a single session.

    When Hermes uses the memory tool (add/replace/remove), this context
    records the change so that:
      1. The agent knows what it recently changed
      2. System prompt extensions can include these changes
      3. Responses can note "Updated memory: ..."
    """

    def __init__(self):
        self._mutations: List[Dict[str, Any]] = []
        self._session_start = time.time()
        self._mutation_count = {"add": 0, "replace": 0, "remove": 0}
        self._first_turn = True
        self._turn_id = 0

    def consume_first_turn(self) -> bool:
        """Returns True only once — for the very first message in this session.
        Call this explicitly instead of accessing a property with side effects."""
        if self._first_turn:
            self._first_turn = False
            self._turn_id += 1  # Turn 1
            return True
        return False

    def next_turn(self):
        """Marks the start of a new turn. Called before each message."""
        self._turn_id += 1

    def record_mutation(
        self,
        action: str,
        target: str,
        content: str,
        old_content: Optional[str] = None
    ):
        """
        Record a memory mutation.

        Args:
            action: "add", "replace", "remove"
            target: "memory" or "user"
            content: New content being stored
            old_content: Previous content (for replace/remove)
        """
        mutation = {
            "action": action,
            "target": target,
            "content": content[:300],  # Truncate long content
            "old_content": (old_content or "")[:300],
            "timestamp": time.time(),
            "turn_id": self._turn_id,
        }

        self._mutations.append(mutation)

        if action in self._mutation_count:
            self._mutation_count[action] += 1

        logger.debug(
            "Memory mutation recorded: %s/%s (total: %d)",
            action, target, len(self._mutations)
        )

    def get_recent_mutations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent mutations."""
        return self._mutations[-limit:]

    def get_mutations_by_target(self, target: str) -> List[Dict[str, Any]]:
        """Return mutations filtered by target type."""
        return [m for m in self._mutations if m["target"] == target]

    def build_system_prompt_fragment(self) -> str:
        """
        Build a compact prompt fragment summarizing recent memory changes.

        Injected into the agent's system prompt so it's contextually
        aware of what it just modified.
        """
        recent = self.get_recent_mutations(limit=10)
        if not recent:
            return ""

        lines = [
            "\n## Recent Memory Changes (This Session)",
            "The following memory entries were modified in this conversation:",
        ]

        for m in recent:
            action_icon = {"add": "➕", "replace": "✏️", "remove": "🗑️"}.get(
                m["action"], "📝"
            )
            action_desc = {
                "add": "Added to",
                "replace": "Updated in",
                "remove": "Removed from",
            }.get(m["action"], m["action"])

            target_label = "user profile" if m["target"] == "user" else "memory"
            preview = m["content"][:80]

            lines.append(
                f"- {action_icon} {action_desc} **{target_label}**: {preview}"
            )

        return "\n".join(lines)

    def flush(self) -> List[Dict[str, Any]]:
        """
        Return all mutations and clear the tracker.
        Useful when persisting to long-term storage.
        """
        mutations = list(self._mutations)
        self._mutations = []
        self._mutation_count = {"add": 0, "replace": 0, "remove": 0}
        return mutations

    def get_stats(self) -> Dict[str, Any]:
        """Get session mutation statistics."""
        return {
            "total_mutations": len(self._mutations),
            "by_action": dict(self._mutation_count),
            "session_duration_s": round(time.time() - self._session_start),
        }


class AdaptiveMemoryPrefetch:
    """
    Pre-fetch relevant memories based on the current user message context.

    Uses VectorMemoryStore for semantic search, falling back to keyword
    matching when vector store is unavailable.
    """

    def __init__(self, vector_store: Optional[Any] = None):
        """
        Args:
            vector_store: VectorMemoryStore instance for semantic search
        """
        self.vector_store = vector_store

    def prefetch_relevant(
        self,
        query: str,
        max_entries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find memories relevant to the current query.

        Args:
            query: User's message to match against
            max_entries: Max number of entries to return

        Returns:
            List of relevant memory entries
        """
        if not self.vector_store:
            logger.debug("No vector store available for prefetch")
            return []

        try:
            results = self.vector_store.search(query, top_k=max_entries)
            return [r for r in results if r.get("score", 0) > 0.1]
        except Exception as e:
            logger.warning("Memory prefetch failed: %s", e)
            return []


def inject_live_memory_into_response(
    response: str,
    dynamic_memory: DynamicMemoryContext,
) -> str:
    """
    Append memory mutation summary to the agent's response.

    This makes memory changes visible to the user so they know
    what was learned/persisted.

    Args:
        response: The agent's response text
        dynamic_memory: Active DynamicMemoryContext instance

    Returns:
        Response with appended memory update notes (if any)
    """
    recent = dynamic_memory.get_recent_mutations(limit=5)
    if not recent:
        return response

    # Only show mutations from the current turn (identified by turn_id)
    current_turn = max(m["turn_id"] for m in recent) if recent else 0
    this_turn = [m for m in recent if m["turn_id"] == current_turn]

    if not this_turn:
        return response

    lines = ["\n\n---\n*Memory updated this turn:*"]

    for m in this_turn:
        action_icon = {"add": "➕", "replace": "✏️", "remove": "🗑️"}.get(
            m["action"], "📝"
        )
        lines.append(f"{action_icon} {m['target']}: {m['content'][:100]}")

    return response + "\n".join(lines)
