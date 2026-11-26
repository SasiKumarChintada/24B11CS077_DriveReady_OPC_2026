# Agentic AI Workshop

Repository for daily tasks, progressive agent architectures, and hands-on implementations in Agentic AI.

---

## 🗂️ Daily Task Tracker
 
### [DAY 1] Fundamentals & First API Call
- **Directory:** [`DAY1/`](file:///c:/Users/ramuv/AgenticAI/DAY1)
- **Notebook:** [`agenticAI_Day1.ipynb`](file:///c:/Users/ramuv/AgenticAI/DAY1/agenticAI_Day1.ipynb)
- **Domain:** Sweet Crumbs Home Bakery
- **Concepts & Implementation:**
  - Understanding LLM core mechanics: Next-token prediction, token estimation, temperature, context window limits, and statelessness.
  - Making the first API call to Groq using LangChain `ChatGroq` (`openai/gpt-oss-120b`).
  - Integrating safety checks using Meta Llama Prompt Guard (`meta-llama/llama-prompt-guard-2-22m`).
  - Interactive notebook testing interface with preset prompts.

---

### [DAY 2] Prompts, Memory, and Building an Agent
- **Directory:** [`DAY2/`](file:///c:/Users/ramuv/AgenticAI/DAY2)
- **Notebook:** [`agenticAI_Day2.ipynb`](file:///c:/Users/ramuv/AgenticAI/DAY2/agenticAI_Day2.ipynb)
- **Domain:** BrightPage Stationery & Office Supplies
- **Concepts & Implementation:**
  - **4-Pillar System Prompt:** Calibrating **Role**, **Facts** (catalogue prices, free delivery over ₹500, bulk discounts), **Rules** (unlisted item fallback: *"I'll check with the store manager..."*), and **Format** (concise, under 60 words).
  - **Few-Shot In-Context Prompting:** High-accuracy customer intent classifier returning exact labels (`order`, `inquiry`, `complaint`).
  - **Structured JSON Output:** Extracting unstructured conversational customer messages into database-ready order JSON schemas.
  - **Python Foundations for Agents:** Inventory dictionaries, type hints, docstrings, and `@log_call` execution logging decorators.
  - **Memory by Hand:** Stateful multi-turn conversation loop with sliding-window history trimming (`MAX_TURNS`) to preserve context window limits.
  - **Error Resilience:** Wrapping API calls in `try / except` to catch `RateLimitError` (429) or connection drops gracefully.
  - **Interactive Colab Dashboard:** `ipywidgets` interface featuring live chat bubbles, quick policy test presets, intent classifier tester, and JSON order extraction.
  - **Student's 10% Verification Suite:** Experiments testing temperature variations ($T=0.2$ vs $T=1.2$), hallucination boundaries, and memory window limits.

---

## 🔑 Setup & API Keys

- All tasks use the same **Groq API Key** (`GROQ_API_KEY`).
- **Google Colab:** Add `GROQ_API_KEY` to Colab Secrets (the 🔑 Key icon on the left sidebar) and toggle **"Notebook access"** to **ON**.
- **Local Runs:** Managed via the single root `.env` (automatically discovered across all days via `python-dotenv`). Copy `.env.example` to `.env` and add your key.
