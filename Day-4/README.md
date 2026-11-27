# Day-4: Tool-Calling Agents, Structured Order Extraction & Sheet Logging

## 📁 Folder Contents

```text
Day-4/
├── task.py
├── README.md
├── sweet_crumbs_orders.csv
└── brightpage_orders.csv
```
🛠️ Key Implementations in Day-4/task.py
1. Resilience & Local Storage Mirror
invoke_with_backoff: Wraps agent and structured extractor calls with exponential backoff (2s -> 4s -> 8s) for transient API errors while immediately surfacing validation exceptions.

LocalOrderSheet: Provides a clean CSV-backed storage mirror (sweet_crumbs_orders.csv and brightpage_orders.csv) supporting append_row() and get_all_records().

2. Sweet Crumbs Home Bakery Agent
Tools (@tool):

get_price(item, qty): Verifies menu availability and computes exact rupee totals.

is_open(day): Returns accurate bakery operating hours for any weekday.

log_order(name, item, qty, date, eggless): Validates inputs through the BakeryOrder Pydantic model and appends confirmed orders to sweet_crumbs_orders.csv.

Verification & Guardrails:

Multi-turn order booking (chat_bakery) with full AIMessage and ToolMessage execution trace inspection.

Edge-case guardrails confirming unlisted items (donut), missing dates, and zero-quantity requests (0 cupcakes) never write invalid rows to the sheet.

Regression evaluation run against Day-3/tests.csv.

3. BrightPage Stationery & Office Supplies Agent
Tools & Structured Extraction:

get_stationery_price(item, qty): Resolves catalogue items and applies an automatic 10% bulk discount for 10+ identical items or subtotals above Rs 1000.

check_store_and_delivery(day, order_total): Checks store hours and calculates local delivery fees (Rs 0 above Rs 500, Rs 40 at or below Rs 500).

StationeryOrder (with_structured_output): Parses 5 informal, messy customer messages into validated JSON records (customer_name, item, qty, delivery_type, delivery_date, notes).

log_stationery_order(...): Validates confirmed orders, calculates discounts and delivery fees, and logs rows to brightpage_orders.csv.

4. Tool Naming & Docstring Impact Experiment
Compares an agent given a vague tool (tool1 with docstring "does stuff") against a properly named and documented tool (get_price), demonstrating how descriptive names and docstrings govern reliable LLM tool selection.

▶️ How to Run
Ensure dependencies are installed:

Bash
pip install langchain langchain-groq pydantic python-dotenv
Set GROQ_API_KEY in the repository root .env file:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
Run the Day-4 script:

Bash
python Day-4/task.py
