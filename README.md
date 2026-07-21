# Hermes Improvements Package (v2.0.0)

An architectural enhancement suite for [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research. 

It introduces dynamic vector-backed memory retrieval, adaptive persona alignment, real-time reasoning tracing, and intelligent multi-provider task classification without breaking per-conversation prompt caching.

---

## 🌟 Key Features

1. **VectorMemoryStore (`vector_memory.py`)**
   - High-performance local semantic search over user memories using embeddings.
   - Enables fast context lookup without bloating the primary prompt window.

2. **DynamicMemoryContext & Prefetch (`dynamic_memory.py`)**
   - Automatically prefetches and injects relevant memories dynamically based on current user prompts.

3. **AdaptiveSoul & StyleLearner (`adaptive_soul.py`)**
   - Learns user style preferences, rules, and corrections over time.
   - Logs persistent behavioral rules and audits adaptations transparently.

4. **AdaptiveWorkflow (`adaptive_workflow.py`)**
   - Classifies task complexity in real time to route sub-tasks to optimal LLM models (e.g. lightweight vs. pro models).

5. **ReasoningTracer & SourceAttribution (`reasoning_trace.py`)**
   - Traces step-by-step reasoning, manages uncertainty scores, and tracks source attribution across execution steps.

6. **Seamless Integration Layer (`integration.py` & `inject_hook.py`)**
   - Non-destructive hook injection into `agent_init.py`.
   - Survives `hermes-agent` upgrades and `pip install --upgrade` routines.

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Hermes Agent installed and active:
```bash
hermes --version
```

### 2. Copy Package Files
Copy the `hermes_improvements` directory into your Hermes configuration home:
```bash
mkdir -p ~/.hermes/improvements
cp -r src/hermes_improvements/* ~/.hermes/improvements/
```

### 3. Inject Hook
Run the non-destructive injector script to pair the improvements with Hermes Agent's execution core:
```bash
python3 src/inject_hook.py
```

### 4. Verify Installation
Check if the improvements package is initialized properly:
```bash
python3 -c "import sys; sys.path.insert(0, '/home/$USER/.hermes'); import improvements; print(improvements.__version__)"
```

---

## 🛡️ License
MIT License. Free to use, modify, and distribute for the Hermes Agent community.
