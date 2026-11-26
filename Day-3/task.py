import os
import csv
import time
import getpass
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent

# 1. Setup & Model Initialization (Groq API)
load_dotenv(find_dotenv())
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    for env_path in ["../.env", ".env", os.path.join(os.getcwd(), "..", ".env")]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                break

if not api_key:
    api_key = getpass.getpass("GROQ_API_KEY not found. Paste your Groq API key: ").strip()
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        raise ValueError("A valid GROQ_API_KEY is required to run Day-3/task.py.")

MODEL_NAME = "openai/gpt-oss-120b"
llm = ChatGroq(model=MODEL_NAME, temperature=0.3, groq_api_key=api_key)
print(f"Groq LLM ready: {MODEL_NAME} (temperature=0.3)\n")

# 2. LangChain LCEL Chain + Multi-Turn Memory (Sweet Crumbs Home Bakery)
BAKERY_SYSTEM = """ROLE:   You are the helpful customer assistant for Sweet Crumbs Home Bakery.
FACTS:  Menu: Plum cake Rs 500, Chocolate cake Rs 650, Cupcake Rs 80 each.
        All cakes are available in eggless versions.
        Every order requires at least 24 hours' advance notice.
RULES:  Never promise a specific delivery time or out-of-town delivery.
        If a question is not covered in FACTS, reply: "I'll check with the owner."
        Calculate item totals accurately.
FORMAT: Keep replies under 50 words, warm, in English or Telugu matching the customer.
        Write all prices as plain numbers without commas (e.g. Rs 1460)."""

bakery_prompt = ChatPromptTemplate.from_messages([
    ("system", BAKERY_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

bakery_chain = bakery_prompt | llm | StrOutputParser()
chat_history = []


def ask_bakery(question: str) -> str:
    """Invoke the LCEL bakery chain while maintaining multi-turn chat history."""
    reply = bakery_chain.invoke({"history": chat_history, "question": question})
    chat_history.extend([("human", question), ("ai", reply)])
    return reply


print("--- Multi-Turn Memory & Customer Verification (Sweet Crumbs) ---")
for user_msg in [
    "Hi, I am Karthik. Do you make eggless chocolate cakes?",
    "Can you remember my name and tell me the cost for 1 chocolate cake and 4 cupcakes?",
    "Can I get this delivered to Visakhapatnam today evening?",
]:
    print(f"Customer:  {user_msg}")
    print(f"Assistant: {ask_bakery(user_msg)}\n")

# 3. Automated Prompt Evaluation (tests.csv)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAKERY_TESTS_PATH = os.path.join(BASE_DIR, "tests.csv")


def run_tests(csv_path: str, active_chain, delay: float = 0.3) -> tuple[float, list[str]]:
    """Run all test rows in csv_path and return (pass_rate, answers)."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    passed = 0
    answers = []
    for row in rows:
        answer = active_chain.invoke({"history": [], "question": row["question"]})
        ok = row["must_include"].lower() in answer.lower()
        passed += int(ok)
        answers.append(answer)
        status = "PASS" if ok else "FAIL"
        print(f"{status} | {row['question']} -> {answer[:75]}")
        if delay:
            time.sleep(delay)

    rate = passed / len(rows)
    print(f"\nResult: {passed}/{len(rows)} passed ({rate:.0%})\n")
    return rate, answers


print("--- Automated Evaluation: Sweet Crumbs (tests.csv) ---")
bakery_rate, _ = run_tests(BAKERY_TESTS_PATH, bakery_chain)

# 4. Agent Sneak Peek (LangChain create_agent with tools=[])
print("--- Agent Sneak Peek (create_agent) ---")
preview_agent = create_agent(model=llm, tools=[], system_prompt=BAKERY_SYSTEM)
agent_res = preview_agent.invoke({
    "messages": [{"role": "user", "content": "What is the price of 2 chocolate cakes?"}]
})
last_msg = agent_res["messages"][-1]
print("Agent Reply:", getattr(last_msg, "content", str(last_msg)), "\n")

# 5. Student's 30%: BrightPage Stationery & Office Supplies Evaluation
STATIONERY_ITEMS = {
    "spiral notebook (200 pgs)": 85,
    "a4 copier paper ream (500 sheets)": 320,
    "gel pen pack (10 pens, blue/black)": 150,
    "pastel highlighter set (6 colors)": 180,
    "heavy-duty stapler + pins": 240,
    "sticky notes pack (400 sheets)": 95,
    "geometry box set": 160,
    "whiteboard marker set (4 colors)": 120,
    "permanent marker (black)": 35,
    "clear file folder (pack of 5)": 110,
}

CATALOGUE_LINES = "\n".join(
    f"  - {name}: Rs {price}" for name, price in STATIONERY_ITEMS.items()
)

BRIGHTPAGE_V1 = f"""ROLE:   You are the customer assistant for BrightPage Stationery & Office Supplies.
FACTS:  Catalogue (INR):
{CATALOGUE_LINES}
        Store Hours: Mon-Sat 9:00 AM - 8:30 PM, Sunday 10:00 AM - 4:00 PM.
        Delivery: Free local delivery above Rs 500; Rs 40 delivery fee for Rs 500 or below.
        Bulk Discount: 10% off on 10+ identical items or orders above Rs 1000.
        Returns: Damaged items replaced within 3 days with receipt.
RULES:  Only deliver within local city limits. If an item is not listed in FACTS,
        say "I'll check with the store manager about availability and get back to you shortly."
FORMAT: Friendly, under 60 words, English or Telugu."""

bp_prompt_v1 = ChatPromptTemplate.from_messages([
    ("system", BRIGHTPAGE_V1),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
bp_chain_v1 = bp_prompt_v1 | llm | StrOutputParser()

BP_TESTS_PATH = os.path.join(BASE_DIR, "brightpage_tests.csv")

print("=== BrightPage Evaluation: Baseline Prompt (V1) ===")
v1_rate, _ = run_tests(BP_TESTS_PATH, bp_chain_v1)

BRIGHTPAGE_V2 = BRIGHTPAGE_V1 + """
        Always write totals as plain numbers without commas (e.g. Rs 1200) and
        explicitly state 'local city limits' whenever asked about out-of-city delivery."""

bp_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", BRIGHTPAGE_V2),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
bp_chain_v2 = bp_prompt_v2 | llm | StrOutputParser()

print("=== BrightPage Evaluation: Improved Prompt (V2) ===")
v2_rate, _ = run_tests(BP_TESTS_PATH, bp_chain_v2)
print(f"Pass rate comparison (V1 -> V2): {v1_rate:.0%} -> {v2_rate:.0%}\n")

# 6. Temperature Stability Check (temperature=1.0 across two runs)
llm_creative = ChatGroq(model=MODEL_NAME, temperature=1.0, groq_api_key=api_key)
bp_chain_temp1 = bp_prompt_v2 | llm_creative | StrOutputParser()

print("=== Temperature = 1.0 Stability Check (Run 1 vs Run 2) ===")
t1_rate_a, t1_answers_a = run_tests(BP_TESTS_PATH, bp_chain_temp1)
t1_rate_b, t1_answers_b = run_tests(BP_TESTS_PATH, bp_chain_temp1)

drifted = sum(a != b for a, b in zip(t1_answers_a, t1_answers_b))
print(f"Responses that changed wording between runs at T=1.0: {drifted}/{len(t1_answers_a)}")
print(f"Pass rates at T=1.0: Run 1 = {t1_rate_a:.0%}, Run 2 = {t1_rate_b:.0%}\n")

# 7. Decision Framework: Plain Code vs One LLM Call vs Agent
WORKFLOW_DECISIONS = [
    (
        "Compute cart subtotal, apply 10% bulk discount for 10+ items, and add Rs 40 delivery fee",
        "Plain Code",
        "Fixed arithmetic rules require deterministic calculation with zero LLM cost or error.",
    ),
    (
        "Label an incoming customer message as order, inquiry, or complaint",
        "One LLM Call",
        "Single-step intent understanding on natural text without needing external tool loops.",
    ),
    (
        "Verify stock across items, calculate discounts, and record a confirmed order in the log",
        "Agent",
        "Multi-step goal where the model dynamically chooses which tools to call and inspect.",
    ),
]

print("=== BrightPage Workflow Classification ===")
for task_desc, category, justification in WORKFLOW_DECISIONS:
    print(f"[{category}] {task_desc}\n  -> {justification}\n")
