# Day-3: Prompt Templates, Memory, Automated Evaluation & Agent Preview

## Task
Refactor single-call LLM scripts into composable LangChain LCEL chains (`ChatPromptTemplate`, `MessagesPlaceholder`, `StrOutputParser`), evaluate prompt reliability using CSV assertion test suites (`tests.csv` and `brightpage_tests.csv`), compare output stability across temperature settings (`0.3` vs `1.0`), and preview LangChain's `create_agent` loop.

## Implementation
- **Composable LCEL Chain (`prompt | llm | StrOutputParser()`)**: Structures system instructions, conversation history (`MessagesPlaceholder("history")`), and user queries cleanly using `ChatGroq` (`openai/gpt-oss-120b`).
- **Multi-Turn Memory (`ask_bakery`)**: Appends `("human", ...)` and `("ai", ...)` turns to `chat_history` so the Sweet Crumbs Home Bakery assistant recalls customer names and enforces 24-hour advance notice and pricing rules.
- **Automated Regression Evaluation (`run_tests`)**: Generates and executes `tests.csv` (Sweet Crumbs Home Bakery) and `brightpage_tests.csv` (BrightPage Stationery & Office Supplies) using `must_include` assertions to calculate exact pass rates.
- **Prompt Iteration & Temperature Analysis**: Compares Baseline (`V1`) vs. Improved (`V2`) system prompts for BrightPage Stationery and measures wording drift across two runs at `temperature=1.0`.
- **Agent Sneak Peek & Task Routing**: Demonstrates `langchain.agents.create_agent` with `tools=[]` and categorizes three BrightPage operational tasks into **Plain Code**, **One LLM Call**, and **Agent**.

## How to Run
1. Install dependencies:
   ```bash
   pip install langchain langchain-groq python-dotenv
