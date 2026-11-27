# 🛠️ Day 4: Tools, Structured Records, Sheets & Agent Traces

Comprehensive guide, concepts, and verified implementation details for **Day 4** of the Agentic AI Workshop (`24B11CS077_DriveReady_OPC_2026`).

---

## 🎯 Task Overview
Upgrade from tool-less chains to **tool-calling agents** using LangChain `create_agent`, `@tool` decorators, Pydantic schemas (`BakeryOrder`, `StationeryOrder`), `with_structured_output`, exponential backoff retries (`invoke_with_backoff`), and persistent CSV-backed order logging (`LocalOrderSheet`).

- **Primary Script:** [`Day-4/task.py`](./task.py)
- **Domains:**
  - **Sweet Crumbs Home Bakery** (Core order agent, trace inspection, customer edge-case guardrails, and `tool1` naming experiment)
  - **BrightPage Stationery & Office Supplies** (2 `@tool` functions, `StationeryOrder` structured extraction, and `log_stationery_order`)
- **Regression Test Suite:** [`../Day-3/tests.csv`](../Day-3/tests.csv)
- **Local Sheet Outputs:** [`sweet_crumbs_orders.csv`](./sweet_crumbs_orders.csv), [`brightpage_orders.csv`](./brightpage_orders.csv)
- **Model Used:** `openai/gpt-oss-20b` via Groq (`ChatGroq`, `temperature=0.2`)

---

## 🧠 Core Engineering Concepts

### 1. `@tool` Functions & Why Docstrings Are the Model's API Docs
- Tools expose Python functions to the LLM with typed arguments (`item: str`, `qty: int = 1`) and clear Google-style docstrings.
- **Break-It Experiment (`tool1` / `"does stuff"`):** Renaming `get_price` to `tool1` with docstring `"does stuff"` caused the model to make **0 tool calls** (`[]`) on `"How much are 3 plum cakes?"` instead of quoting a price. Restoring the descriptive name `get_price` and clear docstring immediately restored tool selection (`['get_price']` -> `3 x plum cake = Rs 1500`).

### 2. Pydantic Schemas & `with_structured_output`
- **Sweet Crumbs (`BakeryOrder`):** Validates `name`, `item` (`Literal["plum cake", "chocolate cake", "cupcake"]`), `qty >= 1`, `date`, and `eggless`.
- **BrightPage (`StationeryOrder`):** Validates `customer_name`, `item` (10-item `Literal`), `qty >= 1`, `delivery_type` (`"home_delivery"` | `"store_pickup"`), `delivery_date`, and optional `notes`.
- Tested `llm.with_structured_output(StationeryOrder)` on **5 messy customer messages** (slang, typos, abbreviations, pickup vs. delivery, and notes) — all 5 parsed cleanly into valid `StationeryOrder` records.

### 3. Reading Agent Traces with `pretty_print()`
For `"Are you open on Sunday? How much are 2 chocolate cakes?"`:
- **Model calls (`AIMessage`):** `3`
- **Tool calls requested:** `2` (`['is_open', 'get_price']`)
- **Tool results (`ToolMessage`):** `2`

Execution sequence:
1. `HumanMessage` -> `AIMessage` calls `is_open(day="sunday")` -> `ToolMessage` returns `"Open 10 am to 6 pm on sunday."`
2. `AIMessage` calls `get_price(item="chocolate cake", qty=2)` -> `ToolMessage` returns `"2 x chocolate cake = Rs 1300"`
3. Final `AIMessage` synthesizes both tool results into a concise customer reply.

### 4. Customer Edge-Case Testing & Sheet Integrity
Tested three incomplete or invalid customer queries on Sweet Crumbs:
1. **Unlisted item (`"donut"`):** `"Can I order 2 donuts for tomorrow?"` -> Explains donuts are not on the menu and lists available items.
2. **Missing pickup date:** `"Hi, I'm Divya. Please book 2 plum cakes."` -> Asks for the pickup date without calling `log_order`.
3. **Invalid quantity (`0 cupcakes`):** `"Hi, I'm Harsha. Please order 0 cupcakes for 27 Dec."` -> Explains minimum quantity is at least 1 and asks how many cupcakes are needed.

- **Sheet integrity verified:** Before and after all 3 edge tests, [`sweet_crumbs_orders.csv`](./sweet_crumbs_orders.csv) retained only the 1 valid order (`Karthik`, `chocolate cake`, `2`, `26 Dec`, `yes`, `1300`).
- **Day-3 `tests.csv` regression run:** Evaluated the Day-4 Sweet Crumbs agent against [`../Day-3/tests.csv`](../Day-3/tests.csv).

---

## 🏗️ Script Implementation Breakdown

1. **Section 1 & 2 · Setup & Local Sheet Connection:** Loads `GROQ_API_KEY` from the root `.env` file (with `getpass` fallback), defines `invoke_with_backoff` for transient errors, and initializes `LocalOrderSheet` CSV storage (`sweet_crumbs_orders.csv`).
2. **Section 3 & 4 · Sweet Crumbs Bakery Agent:** Defines `BakeryOrder`, `@tool`s (`get_price`, `is_open`, `log_order`), and `create_agent`, verifying multi-turn conversation and Karthik's booked order in `sweet_crumbs_orders.csv`.
3. **Section 5 · Execution Trace Inspection:** Prints the full `pretty_print()` trace and counts `AIMessage` model calls (`3`) vs. `ToolMessage` / tool calls (`2`).
4. **Section 6 & 7 · Guardrail Checks & Regression Tests:** Runs the `"donut"`, missing-date (`Divya`), and `0 cupcakes` (`Harsha`) checks, asserts no invalid rows entered the CSV sheet, and evaluates [`../Day-3/tests.csv`](../Day-3/tests.csv).
5. **Section 8 · BrightPage Stationery & Office Supplies:**
   - Converts 2 stationery functions into `@tool`s: `get_stationery_price` (with 10% bulk discount on `qty >= 10` or subtotal `> Rs 1000`) and `check_store_and_delivery` (hours + free delivery above `Rs 500`, `Rs 40` otherwise).
   - Tests `StationeryOrder` with `with_structured_output` on 5 messy messages (`Praveen`, `Kavya`, `Ramesh`, `Sneha`, `Rohit`).
   - Initializes [`brightpage_orders.csv`](./brightpage_orders.csv), implements `log_stationery_order`, tests confirmed/unlisted/missing-date queries, and verifies only Meenakshi's valid order (`12` spiral notebooks, `Rs 102` discount, `Rs 0` delivery, `Rs 918` total) is logged.
6. **Section 9 · Tool Naming & Docstring Impact Experiment:** Demonstrates `tool1` (`"does stuff"`) failing to trigger tool calls (`[]`), then restores `get_price` and verifies accurate tool invocation (`['get_price']`).

---

## 🚀 How to Run

1. Ensure `GROQ_API_KEY` is set in the repository root `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
2. Install dependencies:
   ```bash
   pip install langchain langchain-groq pydantic python-dotenv
   ```
3. Run the Day-4 script from the repository root:
   ```bash
   python Day-4/task.py
   ```
