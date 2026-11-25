# Day-2: BrightPage Stationery & Office Supplies Conversational Agent

## Task
Build a stateful, policy-aware conversational AI assistant for **BrightPage Stationery & Office Supplies** that classifies customer intents, extracts structured JSON orders, calculates item totals with bulk discounts, manages multi-turn conversation memory with sliding-window trimming, and handles API errors gracefully.

## Implementation
- **4-Pillar System Prompt (`SYSTEM_PROMPT`)**: Grounds `openai/gpt-oss-120b` with store catalogue pricing (INR), operating hours, free delivery rules (orders above ₹500), 10% bulk discount rules (10+ identical items or orders above ₹1,000), and strict escalation rules for unstocked items.
- **Few-Shot Intent Classifier (`classify_intent`)**: Uses worked in-context examples at `temperature=0.0` to classify messages into `order`, `inquiry`, or `complaint`.
- **Structured Order Extraction (`extract_order_json`)**: Converts unstructured customer requests into validated JSON (`customer_name`, `items`, `delivery_type`, `urgency`, `notes`).
- **Tool Function & Observability (`@log_call`, `calculate_item_total`)**: Demonstrates typed Python functions, docstrings, dictionary lookups, bulk discount calculation, and execution-time logging.
- **Stateful Agent with Sliding-Window Memory (`StationeryAgent`)**: Retains conversation turns across messages, trims older history using `max_turns` to control token usage, and catches `RateLimitError`, `APIConnectionError`, and `AuthenticationError` exceptions.
- **Verification Suite**: Tests temperature variation (`0.2` vs. `1.2`), unstocked item hallucination boundaries, and sliding-window memory eviction.

## How to Run
1. Install dependencies:
   ```bash
   pip install openai langchain-groq python-dotenv
