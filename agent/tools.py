"""
tools.py — financial tools available to the LangGraph agent.

Each function is decorated with @tool so LangGraph can bind it to the agent.
The docstring of each tool is what the LLM reads to decide when to call it,
so keep them clear and specific.

Data source
-----------
Uses a static mock dataset instead of live API calls. Yahoo Finance blocks
requests from container/cloud environments. Mock data is preferable for eval
pipelines anyway — results are deterministic and reproducible.

To swap in a real API later, replace _MOCK_DATA lookups with live calls
(e.g. Finnhub, Polygon.io) without changing any tool signatures.
"""

from langchain_core.tools import tool

# Realistic reference data (approximate values)
_MOCK_DATA: dict[str, dict] = {
    "AAPL": {
        "name": "Apple Inc.",
        "price": 211.45,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3_180_000_000_000,
        "pe": 32.4,
        "forward_pe": 28.7,
        "pb": 48.2,
        "ev_ebitda": 22.8,
        "eps": 6.53,
        "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "price": 415.20,
        "sector": "Technology",
        "industry": "Software\u2014Infrastructure",
        "market_cap": 3_080_000_000_000,
        "pe": 35.1,
        "forward_pe": 30.2,
        "pb": 13.4,
        "ev_ebitda": 24.6,
        "eps": 11.83,
        "description": "Microsoft Corporation develops and supports software, services, devices, and solutions worldwide, including Azure cloud and Office 365.",
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "price": 178.90,
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "market_cap": 2_200_000_000_000,
        "pe": 22.3,
        "forward_pe": 19.1,
        "pb": 6.8,
        "ev_ebitda": 15.2,
        "eps": 8.02,
        "description": "Alphabet Inc. provides various technology-related products and services including Google Search, YouTube, and Google Cloud.",
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "price": 248.50,
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "market_cap": 795_000_000_000,
        "pe": 58.7,
        "forward_pe": 42.3,
        "pb": 12.1,
        "ev_ebitda": 38.9,
        "eps": 4.23,
        "description": "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, energy generation and storage systems.",
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "price": 875.30,
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 2_150_000_000_000,
        "pe": 68.2,
        "forward_pe": 38.5,
        "pb": 38.7,
        "ev_ebitda": 52.1,
        "eps": 12.84,
        "description": "NVIDIA Corporation provides graphics, computing, and networking solutions used in gaming, data centers, and AI workloads.",
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "price": 192.30,
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "market_cap": 2_020_000_000_000,
        "pe": 41.5,
        "forward_pe": 32.0,
        "pb": 9.3,
        "ev_ebitda": 19.4,
        "eps": 4.63,
        "description": "Amazon.com, Inc. engages in the retail sale of consumer products and subscriptions through online and physical stores, and AWS cloud services.",
    },
}


def _lookup(ticker: str) -> dict | None:
    return _MOCK_DATA.get(ticker.upper().strip())


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.
    Use this when the user asks for the price or value of a stock.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'.

    Returns:
        A string with the current price in USD.
    """
    data = _lookup(ticker)
    if not data:
        return f"No data available for ticker '{ticker.upper()}'."
    return f"The current price of {ticker.upper()} is ${data['price']:.2f} USD."


@tool
def get_stock_info(ticker: str) -> str:
    """
    Get general information about a company including its name, sector,
    industry, market capitalisation, and a brief description.
    Use this when the user asks about what a company does, its sector,
    or its market cap.

    Args:
        ticker: The stock ticker symbol, e.g. 'TSLA', 'NVDA', 'AMZN'.

    Returns:
        A formatted string with company details.
    """
    data = _lookup(ticker)
    if not data:
        return f"No data available for ticker '{ticker.upper()}'."
    return (
        f"Company: {data['name']} ({ticker.upper()})\n"
        f"Sector: {data['sector']}\n"
        f"Industry: {data['industry']}\n"
        f"Market Cap: ${data['market_cap']:,.0f}\n"
        f"Description: {data['description']}"
    )


@tool
def get_financial_ratios(ticker: str) -> str:
    """
    Get key financial ratios for a stock including P/E ratio, forward P/E,
    price-to-book (P/B), and EV/EBITDA.
    Use this when the user asks about valuation metrics or financial ratios.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT'.

    Returns:
        A formatted string with key financial ratios.
    """
    data = _lookup(ticker)
    if not data:
        return f"No data available for ticker '{ticker.upper()}'."
    return (
        f"Financial ratios for {ticker.upper()}:\n"
        f"  Trailing P/E:  {data['pe']:.2f}\n"
        f"  Forward P/E:   {data['forward_pe']:.2f}\n"
        f"  Price/Book:    {data['pb']:.2f}\n"
        f"  EV/EBITDA:     {data['ev_ebitda']:.2f}\n"
        f"  Trailing EPS:  {data['eps']:.2f}"
    )


@tool
def calculate_roi(buy_price: float, sell_price: float) -> str:
    """
    Calculate the Return on Investment (ROI) given a buy price and sell price.
    Use this when the user wants to know their profit or loss percentage
    from buying and selling a stock.

    ROI = ((sell_price - buy_price) / buy_price) * 100

    Args:
        buy_price:  The price at which the stock was purchased.
        sell_price: The price at which the stock was sold.

    Returns:
        A string with the ROI as a percentage.
    """
    if buy_price <= 0:
        return "Buy price must be greater than zero."
    roi = ((sell_price - buy_price) / buy_price) * 100
    direction = "gain" if roi >= 0 else "loss"
    return f"ROI: {roi:.2f}% ({direction})"


@tool
def calculate_percent_change(old_value: float, new_value: float) -> str:
    """
    Calculate the percentage change between two values.
    Use this when the user asks how much a price has changed as a percentage.

    Percent change = ((new - old) / old) * 100

    Args:
        old_value: The original (starting) value.
        new_value: The new (ending) value.

    Returns:
        A string with the percentage change.
    """
    if old_value <= 0:
        return "Original value must be greater than zero."
    change = ((new_value - old_value) / old_value) * 100
    direction = "increase" if change >= 0 else "decrease"
    return f"Percentage change: {change:.2f}% ({direction})"


ALL_TOOLS = [
    get_stock_price,
    get_stock_info,
    get_financial_ratios,
    calculate_roi,
    calculate_percent_change,
]
