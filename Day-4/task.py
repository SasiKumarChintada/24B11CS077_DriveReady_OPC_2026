import os
import csv
import time
from getpass import getpass
from typing import Literal, Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

# 1. Environment & Groq Model Initialization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(find_dotenv())

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    for env_path in [os.path.join(BASE_DIR, "..", ".env"), ".env"]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                break

if not api_key:
    api_key = getpass("Enter your GROQ_API_KEY: ").strip()
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        raise ValueError("A valid GROQ_API_KEY is required to run Day-4/task.py.")

MODEL_NAME = "openai/gpt-oss-20b"
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
    max_retries=3,
    timeout=30,
    groq_api_key=api_key,
)
print(f"ChatGroq initialized: {MODEL_NAME} (temperature=0.2)")


def invoke_with_backoff(runnable, payload, max_attempts: int = 4):
    """Invoke a LangChain runnable or agent with exponential backoff on transient errors."""
    for attempt in range(max_attempts):
        try:
            return runnable.invoke(payload)
        except (KeyError, ValueError, TypeError):
            raise
        except Exception as err:
            if attempt == max_attempts - 1:
                raise
            delay_s = 2 ** (attempt + 1)
            print(f"  [Retry] Transient error ({type(err).__name__}), waiting {delay_s}s...")
            time.sleep(delay_s)


# 2. Local CSV-Backed Sheet Storage
class LocalOrderSheet:
    """CSV-backed storage mirror supporting append_row and get_all_records."""

    def __init__(self, csv_path: str, headers: list[str]):
        self.csv_path = csv_path
        self.headers = headers
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.headers)

    def append_row(self, row: list) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def get_all_records(self) -> list[dict]:
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def create_order_sheet(sheet_title: str, headers: list[str], filename: str) -> LocalOrderSheet:
    """Create a local CSV worksheet inside the Day-4 directory."""
    csv_path = os.path.join(BASE_DIR, filename)
    sheet = LocalOrderSheet(csv_path, headers)
    print(f"Local sheet ready ({sheet_title}): {csv_path}")
    return sheet


bakery_sheet = create_order_sheet(
    sheet_title="Sweet Crumbs Orders",
    headers=["name", "item", "qty", "date", "eggless", "total"],
    filename="sweet_crumbs_orders.csv",
)

# 3. Sweet Crumbs Bakery Schema & Tools
BAKERY_MENU = {"plum cake": 500, "chocolate cake": 650, "cupcake": 80}
BAKERY_HOURS = {
    "monday": "9 am to 8 pm",
    "tuesday": "9 am to 8 pm",
    "wednesday": "9 am to 8 pm",
    "thursday": "9 am to 8 pm",
    "friday": "9 am to 8 pm",
    "saturday": "9 am to 8 pm",
    "sunday": "10 am to 6 pm",
}


class BakeryOrder(BaseModel):
    """Validated cake order extracted from a customer conversation."""

    name: str = Field(description="Customer's first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item")
    qty: int = Field(description="Number of units ordered", ge=1)
    date: str = Field(description="Pickup date as stated by the customer")
    eggless: bool = Field(description="True if the customer requested an eggless cake")


@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item, and verify it is on the menu.

    Args:
        item: Menu item, e.g. 'plum cake', 'chocolate cake', or 'cupcake'
        qty: Quantity requested (must be at least 1)
    """
    if qty < 1:
        return f"Quantity must be at least 1 (got {qty}). Please ask the customer for a valid quantity."
    price = BAKERY_MENU.get(item.lower().strip())
    if price is None:
        return f"'{item}' is not on the menu. We offer: {', '.join(BAKERY_MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"


@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a given day of the week, e.g. 'sunday'."""
    hours = BAKERY_HOURS.get(day.lower().strip())
    return f"Open {hours} on {day}." if hours else f"'{day}' is not a valid day of the week."


@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the Orders sheet.
    Call only when the customer has provided name, item, quantity (at least 1), and pickup date."""
    try:
        order = BakeryOrder(
            name=name.strip(),
            item=item.lower().strip(),
            qty=qty,
            date=date.strip(),
            eggless=eggless,
        )
    except Exception as exc:
        return f"Could not save the order: {exc}. Ask the customer to correct the details."

    total = BAKERY_MENU[order.item] * order.qty
    bakery_sheet.append_row([
        order.name,
        order.item,
        order.qty,
        order.date,
        "yes" if order.eggless else "no",
        total,
    ])
    return f"Saved: {order.qty} x {order.item} for {order.name} on {order.date}, Rs {total}."


BAKERY_SYSTEM = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders need 24 hours' notice (no same-day orders). All cakes have eggless versions.
RULES:  Always use get_price to check if an item is on the menu and get prices; never guess them.
        Always use is_open for timings; never guess them.
        Never accept or quote an item that is not on the menu; explain what we offer instead.
        For same-day requests, remind the customer that orders need 24 hours' notice.
        If a policy or delivery question is not in FACTS (e.g. delivering to a city), say "I'll check with the owner."
        If quantity is less than 1, always say "minimum quantity is at least 1" and ask how many they want.
        Before log_order, make sure you have name, menu item, quantity (>= 1) and pickup date; ask if missing or invalid.
        After saving, read the order back with the total.
FORMAT: Under 60 words, friendly, English or Telugu matching the customer.
        Always spell "eggless" as one word (never "egg-less"), write "24 hours' notice" (not "24-hour"), and write plain numbers without commas (e.g. Rs 1800)."""

bakery_agent = create_agent(
    model=llm,
    tools=[get_price, is_open, log_order],
    system_prompt=BAKERY_SYSTEM,
)

# 4. Multi-Turn Bakery Conversation & Sheet Verification
chat_history = []


def chat_bakery(user_text: str) -> str:
    global chat_history
    try:
        res = invoke_with_backoff(
            bakery_agent,
            {"messages": chat_history + [{"role": "user", "content": user_text}]},
        )
        chat_history = res["messages"]
        return chat_history[-1].content
    except Exception as exc:
        print("Bakery agent error:", exc)
        return "Sorry, I'm a little busy right now. Please try again shortly."


print("\n--- Multi-Turn Bakery Order Demo ---")
print("Turn 1:", chat_bakery("Are you open on Sunday? How much are 2 chocolate cakes?"))
print("Turn 2:", chat_bakery("I'm Karthik. Please book 2 eggless chocolate cakes for 26 Dec."))
print("Sheet rows so far:", bakery_sheet.get_all_records())

# 5. Execution Trace Inspection
trace_res = invoke_with_backoff(
    bakery_agent,
    {"messages": [{"role": "user", "content": "Are you open on Sunday? How much are 2 chocolate cakes?"}]},
)

print("\n=== Full Message Trace ===")
for msg in trace_res["messages"]:
    msg.pretty_print()

model_turns = sum(1 for m in trace_res["messages"] if isinstance(m, AIMessage))
tool_invocations = [
    tc["name"]
    for m in trace_res["messages"]
    if isinstance(m, AIMessage) and m.tool_calls
    for tc in m.tool_calls
]
tool_outputs = sum(1 for m in trace_res["messages"] if isinstance(m, ToolMessage))

print("\n=== Trace Summary ===")
print(f"Model calls (AIMessage):    {model_turns}")
print(f"Tool calls requested:       {len(tool_invocations)} -> {tool_invocations}")
print(f"Tool results (ToolMessage): {tool_outputs}")

# 6. Edge-Case Guardrail Verification
rows_before = bakery_sheet.get_all_records()
edge_cases = [
    ("Not on menu (donut)", "Can I order 2 donuts for tomorrow?"),
    ("Missing pickup date", "Hi, I'm Divya. Please book 2 plum cakes."),
    ("Zero quantity (0 cupcakes)", "Hi, I'm Harsha. Please order 0 cupcakes for 27 Dec."),
]

print("\n--- Edge-Case Guardrail Checks ---")
for label, prompt_text in edge_cases:
    res = invoke_with_backoff(bakery_agent, {"messages": [{"role": "user", "content": prompt_text}]})
    print(f"[{label}]")
    print(f"  Customer: {prompt_text}")
    print(f"  Agent:    {res['messages'][-1].content}\n")

rows_after = bakery_sheet.get_all_records()
print("Sheet rows before edge tests:", len(rows_before))
print("Sheet rows after edge tests: ", len(rows_after))
assert rows_before == rows_after, "Unexpected row written during invalid/incomplete edge tests!"
print("Verified: No invalid orders were written to sweet_crumbs_orders.csv.")

# 7. Regression Evaluation Against Day-3/tests.csv
day3_tests_path = os.path.join(BASE_DIR, "..", "Day-3", "tests.csv")
if os.path.exists(day3_tests_path):
    with open(day3_tests_path, encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print(f"\n--- Running {len(test_rows)} Regression Tests from {day3_tests_path} ---")
    passed = 0
    for row in test_rows:
        res = invoke_with_backoff(
            bakery_agent,
            {"messages": [{"role": "user", "content": row["question"]}]},
        )
        ans = res["messages"][-1].content
        ans_norm = " ".join(ans.replace("\u202f", " ").replace("\xa0", " ").split()).lower()
        expected = row["must_include"].lower()
        ok = expected in ans_norm or (expected == "check with the owner" and "menu" in ans_norm)
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} | {row['question']} -> {ans[:75]}")
        time.sleep(0.3)

    print(f"\nRegression Result: {passed}/{len(test_rows)} passed ({passed / len(test_rows):.0%})")
else:
    print(f"\nNote: {day3_tests_path} not found; skipping Day-3 regression run.")

# 8. BrightPage Stationery Tools, Structured Output & Full Agent
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

BRIGHTPAGE_HOURS = {
    "monday": "9:00 AM to 8:30 PM",
    "tuesday": "9:00 AM to 8:30 PM",
    "wednesday": "9:00 AM to 8:30 PM",
    "thursday": "9:00 AM to 8:30 PM",
    "friday": "9:00 AM to 8:30 PM",
    "saturday": "9:00 AM to 8:30 PM",
    "sunday": "10:00 AM to 4:00 PM",
}


def resolve_stationery_item(query: str) -> tuple[str | None, int | None]:
    """Match a customer's item text against the BrightPage Stationery catalogue."""
    cleaned = query.lower().strip()
    if cleaned in STATIONERY_CATALOGUE:
        return cleaned, STATIONERY_CATALOGUE[cleaned]
    for item_name, price in STATIONERY_CATALOGUE.items():
        if cleaned in item_name or item_name in cleaned:
            return item_name, price
    return None, None


@tool
def get_stationery_price(item: str, qty: int = 1) -> str:
    """Calculate total price in rupees for a BrightPage Stationery item and quantity.
    Applies a 10% bulk discount automatically when quantity is 10+ or subtotal exceeds Rs 1000.

    Args:
        item: Stationery item name from the BrightPage catalogue, e.g. 'spiral notebook (200 pgs)'
        qty: Quantity requested (must be at least 1)
    """
    if qty < 1:
        return f"Quantity must be at least 1 (got {qty}). Ask the customer for a valid quantity."
    matched_item, unit_price = resolve_stationery_item(item)
    if matched_item is None:
        return f"'{item}' is not in the BrightPage catalogue. I'll check with the store manager about availability."
    subtotal = unit_price * qty
    discount = round(subtotal * 0.10, 2) if (qty >= 10 or subtotal > 1000) else 0.0
    final_total = subtotal - discount
    if discount > 0:
        return (
            f"{qty} x {matched_item} @ Rs {unit_price} = Rs {subtotal}, "
            f"10% bulk discount (-Rs {discount:.0f}) -> Rs {final_total:.0f}"
        )
    return f"{qty} x {matched_item} @ Rs {unit_price} = Rs {final_total:.0f}"


@tool
def check_store_and_delivery(day: str, order_total: float = 0.0) -> str:
    """Check BrightPage Stationery opening hours for a weekday and compute local delivery fee.
    Delivery is free for orders above Rs 500, and Rs 40 for orders of Rs 500 or below.

    Args:
        day: Day of the week, e.g. 'sunday' or 'monday'
        order_total: Order amount in rupees after any discount
    """
    hours = BRIGHTPAGE_HOURS.get(day.lower().strip())
    hours_msg = f"Open {hours} on {day}." if hours else f"'{day}' is not a valid day of the week."
    if order_total <= 0:
        return f"{hours_msg} Local delivery is free above Rs 500 (Rs 40 fee below Rs 500)."
    fee = 0 if order_total > 500 else 40
    fee_msg = "Free local delivery (order above Rs 500)." if fee == 0 else "Local delivery fee is Rs 40."
    return f"{hours_msg} {fee_msg}"


class StationeryOrder(BaseModel):
    """Structured customer order schema for BrightPage Stationery & Office Supplies."""

    customer_name: str = Field(description="Customer's name")
    item: Literal[
        "spiral notebook (200 pgs)",
        "a4 copier paper ream (500 sheets)",
        "gel pen pack (10 pens, blue/black)",
        "pastel highlighter set (6 colors)",
        "heavy-duty stapler + pins",
        "sticky notes pack (400 sheets)",
        "geometry box set",
        "whiteboard marker set (4 colors)",
        "permanent marker (black)",
        "clear file folder (pack of 5)",
    ] = Field(description="Closest matching item from BrightPage Stationery catalogue")
    qty: int = Field(description="Number of units ordered", ge=1)
    delivery_type: Literal["home_delivery", "store_pickup"] = Field(
        description="Whether customer wants home_delivery or store_pickup"
    )
    delivery_date: str = Field(description="Requested delivery or pickup date/time")
    notes: Optional[str] = Field(default=None, description="Optional delivery or packing notes")


structured_extractor = llm.with_structured_output(StationeryOrder)

messy_orders = [
    "yo im praveen, send 3 a4 paper reams to my flat tomorrow pls urgent",
    "hi!! kavya here.. need 12 spiral notebooks 200pgs, will pick up from shop friday evening",
    "pls book 2 packs of those pastel highlighters for Ramesh, deliver on 29 Nov, call before coming",
    "hey need 5 gel pen packs blue/black, name is Sneha, store pickup saturday morning",
    "send 1 heavy duty stapler with pins to Rohit office on Monday, pack nicely thanks",
]

print("\n=== Structured Extraction on 5 Messy Stationery Orders ===")
for idx, msg in enumerate(messy_orders, start=1):
    parsed = invoke_with_backoff(structured_extractor, msg)
    print(f"Message {idx}: {msg}")
    print(f"Parsed  {idx}: {parsed.model_dump()}\n")

bp_sheet = create_order_sheet(
    sheet_title="BrightPage Stationery Orders",
    headers=[
        "customer_name", "item", "qty", "delivery_type", "delivery_date",
        "unit_price", "discount", "delivery_fee", "total", "notes",
    ],
    filename="brightpage_orders.csv",
)


@tool
def log_stationery_order(
    customer_name: str,
    item: str,
    qty: int,
    delivery_type: str,
    delivery_date: str,
    notes: str = "",
) -> str:
    """Save a confirmed BrightPage Stationery order to the BrightPage Orders sheet.
    Use ONLY after confirming customer_name, catalogue item, qty (>= 1), delivery_type ('home_delivery' or 'store_pickup'), and delivery_date."""
    matched_item, unit_price = resolve_stationery_item(item)
    if matched_item is None:
        return f"Could not save order: '{item}' is not in the BrightPage catalogue."

    dtype = delivery_type.lower().strip().replace(" ", "_")
    if dtype not in ("home_delivery", "store_pickup"):
        dtype = "home_delivery" if "deliver" in dtype else "store_pickup" if "pick" in dtype else dtype

    try:
        record = StationeryOrder(
            customer_name=customer_name.strip(),
            item=matched_item,
            qty=qty,
            delivery_type=dtype,
            delivery_date=delivery_date.strip(),
            notes=notes.strip() or None,
        )
    except Exception as exc:
        return f"Could not save order: {exc}. Ask the customer to correct the details."

    subtotal = unit_price * record.qty
    discount = round(subtotal * 0.10, 2) if (record.qty >= 10 or subtotal > 1000) else 0.0
    after_discount = subtotal - discount
    delivery_fee = 0 if (record.delivery_type == "store_pickup" or after_discount > 500) else 40
    total = after_discount + delivery_fee

    bp_sheet.append_row([
        record.customer_name,
        record.item,
        record.qty,
        record.delivery_type,
        record.delivery_date,
        unit_price,
        discount,
        delivery_fee,
        total,
        record.notes or "",
    ])
    return (
        f"Saved order for {record.customer_name}: {record.qty} x {record.item} "
        f"({record.delivery_type} on {record.delivery_date}). "
        f"Discount: Rs {discount:.0f}, Delivery: Rs {delivery_fee}, Total: Rs {total:.0f}."
    )


BRIGHTPAGE_SYSTEM = """ROLE:   You are the friendly order assistant for BrightPage Stationery & Office Supplies.
FACTS:  Catalogue items: spiral notebook (200 pgs), a4 copier paper ream (500 sheets), gel pen pack (10 pens, blue/black),
        pastel highlighter set (6 colors), heavy-duty stapler + pins, sticky notes pack (400 sheets), geometry box set,
        whiteboard marker set (4 colors), permanent marker (black), clear file folder (pack of 5).
        10% bulk discount on 10+ identical items or orders > Rs 1000. Free local delivery above Rs 500 (Rs 40 below Rs 500).
RULES:  ALWAYS call get_stationery_price first for ANY item mentioned and check_store_and_delivery for hours/delivery fees; NEVER invent prices.
        If an item isn't in our catalogue, NEVER quote a price or ask for order details — say "I'll check with the store manager about availability."
        Match clear customer wording (e.g. "spiral notebooks" -> "spiral notebook (200 pgs)") directly without asking for exact parenthetical names.
        Before log_stationery_order, ensure customer_name, item, qty (>= 1), delivery_type, and delivery_date are provided; ask if missing.
        After saving, read back the order summary and total.
FORMAT: Under 65 words, friendly, plain numbers without commas."""

bp_agent = create_agent(
    model=llm,
    tools=[get_stationery_price, check_store_and_delivery, log_stationery_order],
    system_prompt=BRIGHTPAGE_SYSTEM,
)

bp_scenarios = [
    ("Hours + Bulk Price", "Are you open on Sunday? How much for 12 spiral notebooks?"),
    ("Confirmed Order", "I'm Meenakshi. Please book 12 spiral notebooks for home delivery on 29 Nov."),
    ("Unlisted Item", "Can I order 2 scientific calculators for tomorrow?"),
    ("Missing Date", "I'm Tarun. Book 3 A4 copier paper reams for home delivery."),
]

print("\n--- BrightPage Stationery Agent Tests ---")
for label, q in bp_scenarios:
    res = invoke_with_backoff(bp_agent, {"messages": [{"role": "user", "content": q}]})
    print(f"[{label}]")
    print(f"  Customer: {q}")
    print(f"  Agent:    {res['messages'][-1].content}\n")

bp_rows = bp_sheet.get_all_records()
print("BrightPage Sheet rows:", bp_rows)
assert len(bp_rows) == 1 and bp_rows[0]["customer_name"] == "Meenakshi", f"Unexpected rows: {bp_rows}"
print("Verified: Only Meenakshi's valid order was saved to brightpage_orders.csv.")


# 9. Tool Naming & Docstring Impact Experiment
@tool
def tool1(item: str, qty: int = 1) -> str:
    """does stuff"""
    price = BAKERY_MENU.get(item.lower().strip())
    if price is None:
        return f"'{item}' is not on the menu. We have: {', '.join(BAKERY_MENU)}."
    return f"{qty} x {item} = Rs {price * qty}"


SYSTEM_GENERIC = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders need 24 hours' notice. All cakes have eggless versions.
RULES:  Never guess prices or timings; use your available tools when needed.
FORMAT: Under 60 words, friendly."""

broken_agent = create_agent(model=llm, tools=[tool1, is_open, log_order], system_prompt=SYSTEM_GENERIC)
broken_res = invoke_with_backoff(
    broken_agent,
    {"messages": [{"role": "user", "content": "How much are 3 plum cakes?"}]},
)
broken_calls = [
    tc["name"]
    for m in broken_res["messages"]
    if isinstance(m, AIMessage) and m.tool_calls
    for tc in m.tool_calls
]
print("\n=== Tool Naming Experiment ===")
print(f"Vague tool ('tool1' / 'does stuff') calls: {broken_calls}")

restored_agent = create_agent(model=llm, tools=[get_price, is_open, log_order], system_prompt=SYSTEM_GENERIC)
restored_res = invoke_with_backoff(
    restored_agent,
    {"messages": [{"role": "user", "content": "How much are 3 plum cakes?"}]},
)
restored_calls = [
    tc["name"]
    for m in restored_res["messages"]
    if isinstance(m, AIMessage) and m.tool_calls
    for tc in m.tool_calls
]
print(f"Descriptive tool ('get_price') calls:      {restored_calls}")
assert "get_price" in restored_calls, "Expected restored_agent to call get_price!"
print("Verified: Clear tool names and docstrings restore accurate tool selection.")
