# Day-1: Sweet Crumbs Home Bakery AI Assistant & Prompt Guard

## Task
Build a domain-specific AI customer assistant for **Sweet Crumbs Home Bakery** using structured prompt engineering (Role, Facts, Rules, Format), measure API token usage, and validate user inputs against prompt injection using a safety guard model.

## Implementation
- **Environment & API Setup**: Loads `GROQ_API_KEY` from environment variables or a `.env` file (with an interactive `getpass` fallback) and initializes both `ChatGroq` (`langchain_groq`) and the official `Groq` client.
- **Structured System Prompt (`ask`)**: Enforces four constraints on `openai/gpt-oss-120b` (`temperature=0.2`):
  - **Role**: Assistant for Sweet Crumbs Home Bakery.
  - **Facts**: Made-to-order policy, 24-hour advance notice requirement, eggless availability, and menu pricing (Plum cake Rs 500, Chocolate cake Rs 650, Cupcake Rs 80).
  - **Rules**: Never promise delivery times and escalate unknown queries to the owner.
  - **Format**: Friendly replies under 50 words matching the customer's language (English or Telugu).
- **Token Usage Tracking**: Uses direct Groq chat completions to inspect `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- **Safety & Prompt Injection Detection (`check_prompt_safety`)**: Evaluates incoming prompts with `meta-llama/llama-prompt-guard-2-22m` and flags prompts with a risk score `>= 0.5`.

## How to Run
1. Install dependencies:
   ```bash
   pip install langchain-groq groq python-dotenv
