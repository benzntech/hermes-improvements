#!/usr/bin/env python3
"""
Integration Layer for Hermes Improvements

Hooks the adaptive components into the existing Hermes agent.
Placed in ~/.hermes/improvements/ to survive `pip install --upgrade`.

The apply-improvements.sh script injects a one-liner into
agent/agent_init.py that imports and calls this module's
patch_agent_for_improvements() at the end of init_agent().
"""

import atexit
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Memory File Indexing & Watcher ──────────────────────────────────

def _index_memory_files(vs, hermes_home: Path):
    """Index MEMORY.md + USER.md into VectorMemoryStore (split on §)."""
    memories_dir = hermes_home / "memories"
    for fname in ["MEMORY.md", "USER.md"]:
        fpath = memories_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("§") if p.strip()]
        for i, para in enumerate(paragraphs):
            vs.add(f"{fname}:{i}", para, {"source": fname, "index": i}, auto_save=False)
        if hasattr(vs, "_save"):
            vs._save()
        logger.info("✅ VectorMemory: %d paragraphs indexed from %s", len(paragraphs), fname)


def _start_memory_watcher(vs, hermes_home: Path):
    """
    Background thread: re-index MEMORY.md/USER.md when they change on disk.
    Uses mtime polling every 30s — no extra dependencies.
    """
    import threading

    memories_dir = hermes_home / "memories"
    watched = {
        fname: (memories_dir / fname).stat().st_mtime
        if (memories_dir / fname).exists() else 0
        for fname in ["MEMORY.md", "USER.md"]
    }

    def _watch():
        while True:
            time.sleep(30)
            try:
                changed_files = []
                for fname, last_mtime in list(watched.items()):
                    fpath = memories_dir / fname
                    if not fpath.exists():
                        continue
                    mtime = fpath.stat().st_mtime
                    if mtime != last_mtime:
                        changed_files.append(fname)
                if changed_files:
                    # refresh all watched mtimes first, then reindex once
                    for fname in watched:
                        fpath = memories_dir / fname
                        watched[fname] = fpath.stat().st_mtime if fpath.exists() else 0
                    _index_memory_files(vs, hermes_home)
                    logger.info("🔄 VectorMemory re-indexed: %s changed", ", ".join(changed_files))
            except Exception as e:
                logger.warning("Memory watcher error: %s", e)

    t = threading.Thread(target=_watch, daemon=True, name="memory-watcher")
    t.start()
    logger.info("✅ Memory file watcher started (poll every 30s)")


# ─── Core: Initialize All Components ─────────────────────────────────

def initialize_hermes_improvements(
    agent_instance,
    hermes_home: Optional[Path] = None,
) -> dict:
    """
    Initialize all improvement components and attach to agent.

    Args:
        agent_instance: The AIAgent instance
        hermes_home: Override Hermes home directory

    Returns:
        Components dict: {
            "vector_store": VectorMemoryStore | None,
            "dynamic_memory": DynamicMemoryContext | None,
            "adaptive_soul": AdaptiveSoul | None,
            "style_learner": StyleLearner | None,
            "adaptive_workflow": AdaptiveWorkflow | None,
            "reasoning_tracer": ReasoningTracer | None,
        }
    """
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home as _get_home
            hermes_home = _get_home()
        except ImportError:
            hermes_home = Path.home() / ".hermes"

    components = {}

    # 1. Vector Memory Store (cu prag de RAM)
    try:
        # Prag RAM minim: 2GB liberi pentru SentenceTransformer
        _free_ram = 0
        try:
            with open('/proc/meminfo') as _f:
                for _line in _f:
                    if _line.startswith('MemAvailable:'):
                        _free_ram = int(_line.split()[1]) // 1024  # KiB → MB
                        break
        except Exception:
            pass

        if _free_ram < 2048:  # < 2GB liberi
            logger.warning("⚠️ VectorMemory: doar %d MB RAM liber (prag minim 2048 MB) — dezactivat", _free_ram)
            components["vector_store"] = None
        else:
            from improvements.vector_memory import VectorMemoryStore
            memory_dir = hermes_home / "memories"
            vs = VectorMemoryStore(memory_dir)
            components["vector_store"] = vs
            logger.info("✅ Vector memory store initialized (%d MB RAM liber)", _free_ram)
    except Exception as e:
        logger.warning("Failed to init vector memory: %s", e)
        components["vector_store"] = None

    # 2. Dynamic Memory Context
    try:
        from improvements.dynamic_memory import DynamicMemoryContext
        dm = DynamicMemoryContext()
        components["dynamic_memory"] = dm
        logger.info("✅ Dynamic memory context initialized")
    except Exception as e:
        logger.warning("Failed to init dynamic memory: %s", e)
        components["dynamic_memory"] = None

    # 3. Adaptive Soul + Style Learner
    try:
        from improvements.adaptive_soul import AdaptiveSoul, StyleLearner
        soul_dir = hermes_home / "memories"
        soul = AdaptiveSoul(soul_dir)
        style = StyleLearner()
        components["adaptive_soul"] = soul
        components["style_learner"] = style
        logger.info("✅ Adaptive soul initialized")
    except Exception as e:
        logger.warning("Failed to init adaptive soul: %s", e)
        components["adaptive_soul"] = None
        components["style_learner"] = None

    # 4. Adaptive Workflow
    try:
        from improvements.adaptive_workflow import AdaptiveWorkflow
        wf = AdaptiveWorkflow()
        components["adaptive_workflow"] = wf
        logger.info("✅ Adaptive workflow initialized")
    except Exception as e:
        logger.warning("Failed to init adaptive workflow: %s", e)
        components["adaptive_workflow"] = None

    # 5. Reasoning Tracer
    try:
        from improvements.reasoning_trace import ReasoningTracer
        rt = ReasoningTracer()
        components["reasoning_tracer"] = rt
        logger.info("✅ Reasoning tracer initialized")
    except Exception as e:
        logger.warning("Failed to init reasoning tracer: %s", e)
        components["reasoning_tracer"] = None

    # Attach to agent instance
    agent_instance._hermes_improvements = components
    agent_instance._hermes_home = hermes_home

    # Populate vector memory from MEMORY.md + USER.md
    vs = components.get("vector_store")
    if vs:
        try:
            _index_memory_files(vs, hermes_home)
            # Start background watcher for live re-indexing on file change
            _start_memory_watcher(vs, hermes_home)
        except Exception as e:
            logger.warning("Failed to populate vector memory from files: %s", e)

    total_ok = sum(1 for v in components.values() if v is not None)
    logger.info(
        "✅ Hermes improvements loaded: %d/%d components",
        total_ok, len(components)
    )
    return components


# ─── Accessors ───────────────────────────────────────────────────────

def get_improvements(agent_instance) -> dict:
    """Get initialized improvements from agent instance."""
    return getattr(agent_instance, "_hermes_improvements", {})


# ─── Pre-Turn Analysis ──────────────────────────────────────────────

def analyze_user_turn(
    agent_instance,
    user_message: str,
    context: dict = None,
) -> dict:
    """
    Pre-turn analysis: classify complexity, detect style, prefetch memories.

    Call this BEFORE processing the user message.
    """
    improvements = get_improvements(agent_instance)
    context = context or {}
    results = {}

    # Classify task complexity
    wf = improvements.get("adaptive_workflow")
    rt = improvements.get("reasoning_tracer")

    # Reset tracer at start of each turn
    if rt:
        rt.clear()

    if wf:
        try:
            complexity = wf.classify_task(user_message, context)
            plan = wf.execute_workflow(user_message, complexity, context)
            results["workflow"] = plan

            # Log reasoning via tracer
            rt = improvements.get("reasoning_tracer")
            if rt:
                rt.add_step(
                    action="analyze",
                    input_data=user_message[:200],
                    reasoning=f"Task classified as {complexity.value} with {len(plan.get('steps', []))} steps",
                    output=f"Workflow: {plan.get('complexity', '?')} — {plan.get('estimated_tools', '?')} tools",
                    confidence=0.8,
                )
        except Exception as e:
            logger.warning("Workflow analysis failed: %s", e)

    # Detect user style
    style = improvements.get("style_learner")
    if style:
        try:
            style.analyze_user_message(user_message)
            results["style_guidance"] = style.get_style_guidance()
        except Exception as e:
            logger.warning("Style analysis failed: %s", e)

    # Prefetch relevant memories
    vs = improvements.get("vector_store")
    if vs:
        try:
            from improvements.dynamic_memory import AdaptiveMemoryPrefetch
            prefetcher = AdaptiveMemoryPrefetch(vs)
            memories = prefetcher.prefetch_relevant(user_message, max_entries=3)
            if memories:
                results["relevant_memories"] = memories

                # Log via tracer
                rt = improvements.get("reasoning_tracer")
                if rt:
                    sources = [m.get("key", "?") for m in memories[:3]]
                    rt.add_step(
                        action="search",
                        input_data=user_message[:100],
                        reasoning=f"VectorMemory found {len(memories)} relevant entries",
                        output=f"Sources: {', '.join(sources)}",
                        confidence=0.7,
                    )
        except Exception as e:
            logger.warning("Memory prefetch failed: %s", e)

    return results


# ─── Feedback Recording ──────────────────────────────────────────────

def record_behavior_feedback(
    agent_instance,
    category: str,
    sentiment: int,
    text: str,
    context: str = "",
):
    """
    Record user feedback about agent behavior.

    Args:
        category: "tone", "accuracy", "formatting", "speed", etc.
        sentiment: -1 (bad), 0 (neutral), +1 (good)
        text: What user said
        context: What agent did
    """
    improvements = get_improvements(agent_instance)
    soul = improvements.get("adaptive_soul")
    if soul:
        try:
            soul.record_feedback(
                category=category,
                sentiment=sentiment,
                text=text,
                context=context,
            )
            logger.debug("Feedback recorded: %s (%+d)", category, sentiment)
        except Exception as e:
            logger.warning("Failed to record feedback: %s", e)


def record_memory_mutation(
    agent_instance,
    action: str,
    target: str,
    content: str,
    old_content: str = None,
):
    """
    Record memory changes for live tracking.

    Args:
        action: "add", "replace", "remove"
        target: "memory", "user"
        content: New content
        old_content: Previous content (for replace/remove)
    """
    improvements = get_improvements(agent_instance)
    dm = improvements.get("dynamic_memory")
    if dm:
        try:
            dm.record_mutation(action, target, content, old_content)
        except Exception as e:
            logger.warning("Failed to record memory mutation: %s", e)


# ─── System Prompt Builder ───────────────────────────────────────────

def build_enhanced_system_prompt(agent_instance) -> str:
    """
    Build system prompt extensions from adaptive components.
    These get appended to the main system prompt.
    """
    improvements = get_improvements(agent_instance)
    extensions = []

    # Dynamic memory instructions
    dm = improvements.get("dynamic_memory")
    if dm:
        ext = dm.build_system_prompt_fragment()
        if ext:
            extensions.append(ext)

    # Adaptive soul rules
    soul = improvements.get("adaptive_soul")
    if soul:
        ext = soul.build_soul_extension()
        if ext:
            extensions.append(ext)

    # Style guidance
    style = improvements.get("style_learner")
    if style:
        guidance = style.get_style_guidance()
        if guidance:
            extensions.append(f"\n## User Style Preferences\n{guidance}")

    # Workflow guidance
    wf = improvements.get("adaptive_workflow")
    if wf:
        extensions.append(
            "\n## Adaptive Workflow\n"
            "Select workflow complexity (TRIVIAL/SIMPLE/MODERATE/COMPLEX) "
            "based on task, use appropriate steps."
        )

    # Reasoning trace — inject only if there are actual steps
    rt = improvements.get("reasoning_tracer")
    if rt and rt.steps:
        trace = rt.get_trace_markdown()
        if trace:
            extensions.append(f"\n## Reasoning Trace (this turn)\n{trace}")
    if rt and rt.claims:
        claims = rt.get_claims_markdown()
        if claims:
            extensions.append(f"\n## Confidence Claims\n{claims}")

    return "\n".join(extensions)


# ─── Response Enhancement ────────────────────────────────────────────

def inject_response_enhancements(
    agent_instance,
    response: str,
    include_reasoning: bool = False,
    include_memory_updates: bool = True,
) -> str:
    """
    Enhance response with live data from adaptive components.

    Args:
        response: Original assistant response
        include_reasoning: Append reasoning trace
        include_memory_updates: Append memory mutation summary

    Returns:
        Enhanced response string
    """
    improvements = get_improvements(agent_instance)

    if include_memory_updates:
        dm = improvements.get("dynamic_memory")
        if dm:
            try:
                from improvements.dynamic_memory import inject_live_memory_into_response
                response = inject_live_memory_into_response(response, dm)
            except Exception as e:
                logger.warning("Memory injection failed: %s", e)

    if include_reasoning:
        rt = improvements.get("reasoning_tracer")
        if rt:
            try:
                from improvements.reasoning_trace import inject_reasoning_trace
                response = inject_reasoning_trace(response, rt)
            except Exception as e:
                logger.warning("Reasoning injection failed: %s", e)

    return response


# ─── Stats ───────────────────────────────────────────────────────────

def get_agent_stats(agent_instance) -> dict:
    """Get comprehensive stats from all adaptive components."""
    improvements = get_improvements(agent_instance)
    import time as _time

    stats = {
        "timestamp": _time.time(),
        "components": {},
    }

    soul = improvements.get("adaptive_soul")
    if soul:
        stats["components"]["adaptive_soul"] = soul.get_behavioral_stats()

    vs = improvements.get("vector_store")
    if vs:
        stats["components"]["memory_vectors"] = vs.get_memory_stats()

    dm = improvements.get("dynamic_memory")
    if dm:
        stats["components"]["dynamic_memory"] = dm.get_stats()

    wf = improvements.get("adaptive_workflow")
    if wf:
        stats["components"]["workflow_metrics"] = wf.metrics.get_stats()

    return stats


# ─── Session Persistence ──────────────────────────────────────────────

def persist_session_learnings(agent_instance):
    """
    Persist all learnings from this turn across sessions.
    Called after every response. Idempotent — safe to call always.

    Does 3 things:
    1. Syncs DynamicMemory mutations into VectorMemory index
    2. Saves AdaptiveSoul rules as a readable summary file
    3. Logs what was learned
    """
    import json
    improvements = get_improvements(agent_instance)
    learned = []

    # 1. DynamicMemory → VectorMemory sync
    dm = improvements.get("dynamic_memory")
    vs = improvements.get("vector_store")
    if dm and vs:
        mutations = dm.get_recent_mutations()
        if mutations:
            vs.sync_from_memory(mutations)
            learned.append(f"sync {len(mutations)} memory mutations to vector index")

    # 2. AdaptiveSoul rules → readable summary file
    soul = improvements.get("adaptive_soul")
    if soul:
        rules = soul.get_active_rules(limit=10)
        if rules:
            summary_path = Path(agent_instance._hermes_home if hasattr(agent_instance, '_hermes_home') else Path.home() / ".hermes") / "memories" / "_learned_rules.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            rules_data = []
            for r in rules:
                rules_data.append({
                    "name": r.name,
                    "category": r.category,
                    "condition": r.condition,
                    "action": r.action,
                    "confidence": r.confidence,
                    "hits": r.hits,
                })
            with open(summary_path, 'w') as f:
                json.dump(rules_data, f, indent=2)

            learned.append(f"saved {len(rules)} adaptive soul rules")

        # Also: persist any recent feedback as a memory entry
        if hasattr(soul, 'feedback_history') and soul.feedback_history:
            recent = soul.feedback_history[-3:]
            negative = [f for f in recent if f.sentiment < 0]
            if negative:
                for fb in negative:
                    learned.append(
                        f"corecție: '{fb.text[:60]}' (categorie: {fb.category})"
                    )

    if learned:
        logger.info("📝 Session learnings persisted: %s", "; ".join(learned))

    # 3. Save DynamicMemory session conclusions (for one-shot mode -z)
    dm = improvements.get("dynamic_memory")
    if dm:
        try:
            conclusions_path = Path(
                agent_instance._hermes_home
                if hasattr(agent_instance, '_hermes_home')
                else Path.home() / ".hermes"
            ) / "memories" / "_session_conclusions.jsonl"

            stats = dm.get_stats()
            mutations = dm.get_recent_mutations(limit=50)
            session_summary = {
                "timestamp": time.time(),
                "session_duration_s": stats.get("session_duration_s", 0),
                "total_mutations": stats.get("total_mutations", 0),
                "by_action": stats.get("by_action", {}),
                "key_learnings": [
                    m["content"][:200]
                    for m in mutations[-10:]
                    if m["action"] in ("add", "replace")
                ],
            }

            # Append one JSON line per session (JSONL — easy to append)
            import json as _json
            import fcntl
            with open(conclusions_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(_json.dumps(session_summary) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)

            learned.append(
                f"saved {len(session_summary['key_learnings'])} session conclusions"
            )
        except Exception as e:
            logger.warning("Session conclusions save failed: %s", e)


# ─── Agent Patching ──────────────────────────────────────────────────

def patch_agent_for_improvements(agent_instance):
    """
    Patch AIAgent with improvement hooks.

    Call this right after agent initialization.
    Safe to call multiple times — skips if already patched.
    """
    if getattr(agent_instance, "_hermes_patched", False):
        logger.debug("Agent already patched with improvements, skipping")
        return

    # Initialize all components
    initialize_hermes_improvements(agent_instance)

    # Store the original handle_message if it exists
    original_handle = getattr(agent_instance, "handle_message", None)

    def enhanced_handle_message(user_message, *args, **kwargs):
        """Wrapped message handler with pre/post-turn enhancements."""
        # Reset ReasoningTracer for fresh trace each turn
        try:
            rt = get_improvements(agent_instance).get("reasoning_tracer")
            if rt:
                rt.clear()
        except Exception:
            pass

        # Pre-turn: analyze
        try:
            turn_ctx = analyze_user_turn(agent_instance, str(user_message))
            # Auto-routing based on task complexity (Flash 2.5 Lite by default, upgrades to Flash 3.5 / Pro if complex)
            if "workflow" in turn_ctx:
                c_val = turn_ctx["workflow"].get("complexity", "trivial")
                
                # TRIVIAL/SIMPLE: gemini-2.5-flash-lite (default)
                # MODERATE: gemini-3.5-flash-low
                # COMPLEX: gemini-3.1-pro-low
                target_model = "gemini-2.5-flash-lite"
                
                if c_val == "moderate":
                    target_model = "gemini-3.5-flash-low"
                elif c_val == "complex":
                    target_model = "gemini-3.1-pro-low"
                
                # Auto-routing dezactivat temporar (config.yaml este imutabil chattr +i)
                # agent_instance.model = target_model
                logger.info("🔄 [Bypassed] Auto-routing: task complexity [%s] -> Target would be [%s]", c_val.upper(), target_model)
        except Exception as e:
            logger.warning("Pre-turn analysis or auto-routing failed: %s", e)
            turn_ctx = {}

        # Detect first turn — inject system intro
        dm = get_improvements(agent_instance).get("dynamic_memory")
        is_first = dm and dm.consume_first_turn() if dm else False

        # Call original
        if original_handle:
            response = original_handle(user_message, *args, **kwargs)
        else:
            response = ""

        # Post-turn: enhance response
        try:
            response = inject_response_enhancements(
                agent_instance,
                str(response),
                include_reasoning=False,  # Expensive — only when requested
                include_memory_updates=True,
            )
        except Exception as e:
            logger.warning("Post-turn enhancement failed: %s", e)

        # Persist everything — learnings survive across sessions
        try:
            persist_session_learnings(agent_instance)
        except Exception as e:
            logger.warning("Session persistence failed: %s", e)

        return response

    if original_handle:
        agent_instance.handle_message = enhanced_handle_message

    agent_instance._hermes_patched = True

    # Register atexit handler so conclusions are saved even in one-shot mode (-z)
    # and on any exit path (normal, error, Ctrl+C)
    import weakref
    ref = weakref.ref(agent_instance)

    def _save_on_exit():
        agent = ref()
        if agent is None:
            return
        try:
            persist_session_learnings(agent)
        except Exception as e:
            logger.warning("atexit session persistence failed: %s", e)

    atexit.register(_save_on_exit)

    logger.info("✅ Agent patched with Hermes improvements (atexit handler registered)")


# ─── Public API ─────────────────────────────────────────────────────

__all__ = [
    "initialize_hermes_improvements",
    "get_improvements",
    "analyze_user_turn",
    "record_behavior_feedback",
    "record_memory_mutation",
    "build_enhanced_system_prompt",
    "inject_response_enhancements",
    "get_agent_stats",
    "patch_agent_for_improvements",
    "persist_session_learnings",
]
