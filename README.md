# Agentic AI Workshop (`24B11CS077_DriveReady_OPC_2026`)

Repository for daily tasks, progressive agent architectures, and hands-on Python implementations in Agentic AI.

## 📁 Repository Structure

```text
24B11CS077_DriveReady_OPC_2026/
├── Day-1/
│   ├── task.py
│   └── README.md
├── Day-2/
│   ├── task.py
│   └── README.md
├── Day-3/
│   ├── task.py
│   ├── README.md
│   ├── tests.csv
│   └── brightpage_tests.csv
├── .env.example
├── .gitignore
└── README.md
```

## 🗂️ Daily Task Tracker

### `[DAY 1]` Fundamentals & First API Call
- **Directory:** `Day-1/`
- **Script:** `Day-1/task.py`
- **Domain:** Sweet Crumbs Home Bakery
- **Concepts & Implementation:**
  - Understanding LLM core mechanics: Next-token prediction, token estimation, temperature, context window limits, and statelessness.
  - Making the first API call to Groq using LangChain `ChatGroq` (`openai/gpt-oss-120b`).
  - Integrating prompt-injection and safety checks using Meta Llama Prompt Guard (`meta-llama/llama-prompt-guard-2-22m`).
  - Testing structured system prompts (Role, Facts, Rules, Format) across customer inquiry presets.

### `[DAY 2]` Prompts, Memory, and Building an Agent
- **Directory:** `Day-2/`
- **Script:** `Day-2/task.py`
- **Domain:** BrightPage Stationery & Office Supplies
- **Concepts & Implementation:**
  - **4-Pillar System Prompt:** Calibrating Role, Facts (catalogue prices, free delivery over ₹500, bulk discounts), Rules (unlisted item fallback: *"I'll check with the store manager..."*), and Format (concise, under 60 words).
  - **Few-Shot In-Context Prompting:** High-accuracy customer intent classifier returning exact labels (`order`, `inquiry`, `complaint`).
  - **Structured JSON Output:** Extracting unstructured conversational customer messages into database-ready order JSON schemas.
  - **Python Foundations for Agents:** Inventory dictionaries, type hints, docstrings, and `@log_call` execution logging decorators.
  - **Memory by Hand:** Stateful multi-turn conversation loop with sliding-window history trimming (`max_turns`) to preserve context window limits.
  - **Error Resilience:** Wrapping API calls in `try / except` to catch `RateLimitError` (429), authentication, and connection issues gracefully.
  - **Verification Suite:** Experiments testing temperature variations (`T=0.2` vs `T=1.2`), unstocked item hallucination boundaries, and memory window limits.

### `[DAY 3]` Prompt Templates, Memory, Evaluation & Agent Preview
- **Directory:** `Day-3/`
- **Script:** `Day-3/task.py`
- **Test Suites:** `Day-3/tests.csv` & `Day-3/brightpage_tests.csv`
- **Domains:** Sweet Crumbs Home Bakery & BrightPage Stationery & Office Supplies
- **Concepts & Implementation:**
  - **Composable LCEL Chains (`prompt | llm | StrOutputParser()`):** Refactoring manual message lists into reusable `ChatPromptTemplate` pipelines with `MessagesPlaceholder("history")` and `StrOutputParser()`.
  - **Multi-Turn Chain Memory:** Managing conversational context across turns via `ask_bakery()` while enforcing pricing and 24-hour advance notice rules.
  - **Automated Regression Evaluation (`run_tests`):** Running deterministic `must_include` assertions against `tests.csv` and `brightpage_tests.csv` to measure prompt pass rates.
  - **Prompt Iteration & Temperature Stability:** Comparing Baseline (`V1`) vs. Improved (`V2`) BrightPage prompts and measuring output drift across two runs at `temperature=1.0`.
  - **Agent Preview & Workflow Routing:** Demonstrating `langchain.agents.create_agent` (`tools=[]`) and categorizing tasks into Plain Code, One LLM Call, and Agent.

## 🔑 Setup & API Keys

- All tasks use the same **Groq API Key (`GROQ_API_KEY`)**.
- **Local Runs:** Copy `.env.example` to `.env` in the repository root and add your key (`.env` is listed in `.gitignore` so it is never committed to GitHub):

```env
GROQ_API_KEY=your_groq_api_key_here
