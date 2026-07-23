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

## 📉 Token & Cost Reduction (How it Works)

Normally, Hermes Agent loads **your entire memory file** (all custom rules, preferences, and details) into the System Prompt of every single message. As your memory grows, your system prompt bloats, wasting thousands of tokens per turn and skyrocketing your API costs.

This package solves this through two main optimizations:

1. **Semantic Memory Selection (Vector Search)**:
   - Instead of injecting the entire memory dump, `VectorMemoryStore` indexes your memories using local embeddings (via `sentence-transformers` or a lightweight TF-IDF fallback).
   - On each message, it performs a quick search and **only injects the 1-3 memories relevant to your current prompt** (e.g., retrieving your TV setup details only when you talk about Stremio).

2. **Preserving Prompt Caching**:
   - Because the bulk of inactive memories is kept out of the prompt, the system prompt stays stable.
   - This allows LLM providers (like Google Gemini or Anthropic Claude) to hit **100% Prompt Cache** on repetitive turns. You only pay for processing new messages, saving up to 50-80% on API bills.

---

## 📋 Requirements

- **Python:** `>= 3.10`
- **Memory:** `≥ 2 GB RAM` (if using `sentence-transformers` embeddings)
- **Optional Dependencies:**
  - `sentence-transformers` & `numpy` (for vector semantic search; falls back to TF-IDF if omitted)

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
python3 -c "import sys, os; sys.path.insert(0, os.path.expanduser('~/.hermes')); import improvements; print(improvements.__version__)"
```

---

## 💡 Troubleshooting & Notes

- **Upgrades:** Running `python3 src/inject_hook.py` is safe to run after every `pip install --upgrade hermes-agent` or `git pull`. It checks if the hook is already present before modifying `agent_init.py`.
- **Platform Support:** Full support for Linux and macOS. Windows environments gracefully fall back when file-locking is unavailable.

---

## 🛡️ License
MIT License. Free to use, modify, and distribute for the Hermes Agent community.
