import os
import re
import json
import time
import getpass
import functools
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, AuthenticationError
from langchain_groq import ChatGroq

# 1. Load GROQ_API_KEY from .env or environment
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
    api_key = getpass.getpass("GROQ_API_KEY not found. Please paste your Groq API key: ").strip()
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        raise ValueError("A valid GROQ_API_KEY is required to proceed.")

# Model selection and client initialization
MODEL_NAME = "openai/gpt-oss-120b"

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
    groq_api_key=api_key,
)

print(f"Groq client connected successfully! Target Model: {MODEL_NAME}\n")

# Step 3: Inventory Catalogue & 4-Pillar System Prompt (Role, Facts, Rules, Format)
STATIONERY_CATALOGUE = {
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

SYSTEM_PROMPT = f"""ROLE:
You are the friendly, helpful AI virtual assistant for 'BrightPage Stationery & Office Supplies'.

FACTS:
- Store Catalogue & Prices (INR): 
{json.dumps(STATIONERY_CATALOGUE)}
- Operating Hours: Mon-Sat 9:00 AM - 8:30 PM, Sunday 10:00 AM - 4:00 PM.
- Delivery Policy: Free local home/office delivery on orders above ₹500. Standard ₹40 delivery fee for orders below ₹500.
- Delivery Timing: Orders placed before 3:00 PM delivered same evening; orders after 3:00 PM delivered next morning.
- Bulk Discount: 10% discount automatically applied on orders of 10+ identical items or order total exceeding ₹1,000.
- Return/Exchange Policy: Defective or damaged items replaced within 3 days with receipt.

RULES:
- Never promise delivery outside our local city limits.
- If an item is NOT in our FACTS catalogue (e.g. laptops, heavy furniture, textbooks), say:
  "I'll check with the store manager about availability and get back to you shortly."
- Never invent unlisted products or guess unverified delivery times.
- Always confirm item quantities, total pricing, and delivery preference before finalizing an order.
- Maintain a warm, courteous tone at all times.

FORMAT:
- Keep responses friendly, clear, and under 60 words.
- Match the customer's language (English, Telugu, or Hinglish if requested)."""


# Step 4: Few-Shot Prompting (Customer Intent Classifier)
def classify_intent(customer_message: str) -> str:
    """Classifies customer message into 'order', 'inquiry', or 'complaint' using few-shot prompting."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer intent classifier for BrightPage Stationery. "
                "Respond with EXACTLY ONE word from these three options: order, inquiry, or complaint. "
                "Do not write any punctuation or explanation."
            ),
        },
        {"role": "user", "content": "Can I get 5 reams of A4 paper delivered to Tech Park tomorrow morning?"},
        {"role": "assistant", "content": "order"},
        {"role": "user", "content": "Do you have Classmate spiral notebooks with grid lines?"},
        {"role": "assistant", "content": "inquiry"},
        {"role": "user", "content": "The gel pen pack I bought yesterday is leaking ink everywhere!"},
        {"role": "assistant", "content": "complaint"},
        {"role": "user", "content": customer_message},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=messages,
    )
    return response.choices[0].message.content.strip().lower()


# Step 5: Structured Output (Extracting Clean JSON Orders)
ORDER_EXTRACTION_SYSTEM = """You are an order extraction engine for BrightPage Stationery.
Extract customer order details from the message into a strictly valid JSON object.
Return ONLY raw JSON. No markdown backticks, no code fences, no conversational filler.

JSON Schema:
{
  "customer_name": string or null,
  "items": [
    {
      "item_name": string,
      "quantity": integer
    }
  ],
  "delivery_type": "home_delivery" or "store_pickup",
  "urgency": "normal" or "urgent",
  "notes": string or null
}"""


def extract_order_json(customer_message: str) -> dict:
    """Extracts structured order dictionary from conversational customer text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ORDER_EXTRACTION_SYSTEM},
            {"role": "user", "content": customer_message},
        ],
    )
    raw_content = response.choices[0].message.content.strip()
    cleaned = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        print(f"JSON parsing error: {err}. Raw output was:\n{raw_content}")
        return {"error": "Failed to parse JSON", "raw": raw_content}


# Step 6: Python Foundations for Agents (Decorator + Billing Tool)
def log_call(func):
    """Decorator that logs function execution, arguments, and return values."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        print(f"   [LOG CALL] Entering {func.__name__}({arg_str})")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        print(f"   [LOG RETURN] {func.__name__} returned {result!r} ({duration_ms:.2f} ms)")
        return result

    return wrapper


@log_call
def calculate_item_total(item_name: str, qty: int = 1) -> dict:
    """
    Calculate the total cost for a specific stationery item and quantity.
    Applies a 10% bulk discount if quantity is 10 or more.
    """
    cleaned_name = item_name.lower().strip()
    unit_price = STATIONERY_CATALOGUE.get(cleaned_name)

    if unit_price is None:
        for cat_item, price in STATIONERY_CATALOGUE.items():
            if cleaned_name in cat_item or cat_item in cleaned_name:
                unit_price = price
                cleaned_name = cat_item
                break

    if unit_price is None:
        return {"error": f"Item '{item_name}' is not currently in the catalogue."}

    subtotal = unit_price * qty
    discount = (subtotal * 0.10) if qty >= 10 else 0.0
    final_total = subtotal - discount

    return {
        "item": cleaned_name,
        "unit_price": unit_price,
        "quantity": qty,
        "subtotal": subtotal,
        "bulk_discount": discount,
        "final_total": final_total,
    }


# Step 7: Multi-Turn Memory & Resilient Chat Loop
class StationeryAgent:
    """Conversational Agent for BrightPage Stationery with stateful memory and error safety."""

    def __init__(self, system_prompt: str, max_turns: int = 8, model: str = MODEL_NAME):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.model = model
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens_used = 0

    def reset(self):
        """Reset memory to initial system prompt state."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        print("Agent memory cleared. Started new session.")

    def chat(self, user_message: str) -> str:
        """Send message to agent, retain history, and return assistant reply."""
        if not user_message.strip():
            return "Please type a message!"

        self.history.append({"role": "user", "content": user_message})
        trimmed_messages = [self.history[0]] + self.history[1:][-(self.max_turns * 2) :]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=trimmed_messages,
                temperature=0.3,
            )
            reply = response.choices[0].message.content.strip()
            if hasattr(response, "usage") and response.usage:
                self.total_tokens_used += response.usage.total_tokens

        except RateLimitError:
            reply = "We are currently experiencing high customer volume. Please wait a moment and try again!"
        except APIConnectionError:
            reply = "I cannot reach the store servers right now. Please check your internet connection."
        except AuthenticationError:
            reply = "Authentication error: Store API key is misconfigured. Please notify the manager."
        except Exception as e:
            reply = f"An unexpected error occurred: {str(e)}. I'll notify the store owner."

        self.history.append({"role": "assistant", "content": reply})
        return reply


if __name__ == "__main__":
    # 1. Sanity Check
    test_response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Respond with 'BrightPage Stationery Agent is Ready!'"},
        ],
    )
    print("API Sanity Check:", test_response.choices[0].message.content.strip(), "\n")

    # 2. Few-Shot Intent Classification
    print("--- Few-Shot Intent Classification ---")
    test_queries = [
        "I want to order 3 packs of pastel highlighters and 2 sticky note pads.",
        "What are your store opening hours on Sunday?",
        "The stapler pins you gave me don't fit the stapler model!",
        "Do you offer free delivery for orders around 600 rupees?",
    ]
    for q in test_queries:
        print(f'  "{q}" -> [{classify_intent(q).upper()}]')

    # 3. Structured JSON Order Extraction
    print("\n--- Structured JSON Order Extraction ---")
    sample_order_msg = (
        "Hello! This is Ramesh Kumar. Please send 2 reams of A4 copier paper "
        "and 3 pastel highlighter sets to my clinic urgently."
    )
    extracted = extract_order_json(sample_order_msg)
    print(json.dumps(extracted, indent=2))

    # 4. Tool Calculation & @log_call Decorator
    print("\n--- Item Billing Calculation with @log_call ---")
    calc_result = calculate_item_total("spiral notebook (200 pgs)", 12)
    print("Result:", calc_result)

    # 5. Multi-Turn Memory Verification
    print("\n--- Multi-Turn Memory Verification ---")
    stationery_agent = StationeryAgent(system_prompt=SYSTEM_PROMPT, max_turns=6)
    turns = [
        "Hi, I'm Ananya. Do you have heavy duty staplers in stock?",
        "How much is it, and is delivery free if I also take an A4 paper ream?",
        "Great! By the way, what was my name again?",
    ]
    for msg in turns:
        print(f"Customer: {msg}")
        print(f"Agent:    {stationery_agent.chat(msg)}\n")
    print(f"Total tokens used in multi-turn session: {stationery_agent.total_tokens_used}")

    # 6. Verification Suite: Temperature Dial (T=0.2 vs T=1.2)
    print("\n--- Experiment 1: Temperature Dial Test ---")
    slogan_prompt = "Write a catchy 6-word tagline for BrightPage Stationery & Office Supplies."
    for temp in (0.2, 1.2):
        print(f"Temperature = {temp}:")
        for i in range(3):
            res = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=temp,
                messages=[{"role": "user", "content": slogan_prompt}],
            )
            print(f"  Run {i + 1}: {res.choices[0].message.content.strip()}")

    # 7. Verification Suite: Hallucination Boundary Test
    print("\n--- Experiment 2: Unstocked Item Boundary Test ---")
    eval_agent = StationeryAgent(system_prompt=SYSTEM_PROMPT, max_turns=2)
    for q in [
        "Do you sell resin 3D printer filaments?",
        "Can you deliver a 27-inch computer monitor to my office today?",
        "Do you have watercolor sketchbooks?",
    ]:
        print(f"Customer: {q}")
        print(f"Agent:    {eval_agent.chat(q)}\n")

    # 8. Verification Suite: Sliding-Window Memory Trimming
    print("--- Experiment 3: Sliding-Window Memory Trimming (max_turns=2) ---")
    small_window_agent = StationeryAgent(system_prompt=SYSTEM_PROMPT, max_turns=2)
    for step_msg in [
        "My name is Suresh and my account number is BP-904.",
        "What is the price of your spiral notebooks?",
        "How many colors come in the pastel highlighter set?",
        "What was my name and account number?",
    ]:
        print(f"Customer: {step_msg}")
        print(f"Bot:      {small_window_agent.chat(step_msg)}\n")
    print("Active history message count in memory:", len(small_window_agent.history))
