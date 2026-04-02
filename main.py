from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import math
import traceback
import httpx
import json
import os

yf.set_tz_cache_location("/tmp/yf_cache")

app = FastAPI()

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

class DCFRequest(BaseModel):
    ticker: str
    wacc: float
    terminal_growth: float
    revenue_growth: float
    projection_years: int = 10

class TickerLookup(BaseModel):
    ticker: str

class NewsRequest(BaseModel):
    ticker: str

# ---------------------------------------------------------------------------
# Fetch financials
# ---------------------------------------------------------------------------

def fetch_financials(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker, session=None)
        info = stock.info
        if not info or len(info) <= 1:
            raise Exception("Empty response from Yahoo Finance")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for '{ticker}': {str(e)}")

    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found or has no price data.")

    try:
        income = stock.financials
        cashflow = stock.cashflow
        balance = stock.balance_sheet
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch financial statements for '{ticker}': {str(e)}")

    if income is None or income.empty:
        raise HTTPException(status_code=404, detail=f"No income statement found for '{ticker}'.")
    if cashflow is None or cashflow.empty:
        raise HTTPException(status_code=404, detail=f"No cash flow statement found for '{ticker}'.")
    if balance is None or balance.empty:
        raise HTTPException(status_code=404, detail=f"No balance sheet found for '{ticker}'.")

    def safe(df, keys, col=None):
        if col is None:
            col = df.columns[0]
        for k in keys:
            if k in df.index:
                val = df.loc[k, col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return float(val)
        return 0.0

    revenue = safe(income, ["Total Revenue", "Revenue"])
    ebit = safe(income, ["EBIT", "Operating Income"])
    depreciation = safe(cashflow, ["Depreciation And Amortization", "Depreciation & Amortization", "Depreciation"])
    capex = abs(safe(cashflow, ["Capital Expenditure", "Capital Expenditures"]))
    tax_provision = safe(income, ["Tax Provision", "Income Tax Expense"])
    pretax_income = safe(income, ["Pretax Income", "Income Before Tax"])
    interest_expense = abs(safe(income, ["Interest Expense", "Interest Expense Non Operating"]))
    net_income = safe(income, ["Net Income", "Net Income Common Stockholders"])

    if pretax_income != 0:
        tax_rate = max(0, min(tax_provision / pretax_income, 0.40))
    else:
        tax_rate = 0.21

    def wc(col):
        ca = safe(balance, ["Current Assets", "Total Current Assets"], col)
        cl = safe(balance, ["Current Liabilities", "Total Current Liabilities"], col)
        return ca - cl

    nwc_change_abs = 0.0
    if len(balance.columns) >= 2:
        nwc_change_abs = wc(balance.columns[0]) - wc(balance.columns[1])

    rev_prior = 0.0
    if len(income.columns) >= 2:
        rev_prior = safe(income, ["Total Revenue", "Revenue"], income.columns[1])

    rev_delta = revenue - rev_prior if rev_prior > 0 else revenue
    if rev_delta != 0 and abs(rev_delta) > 1e6:
        nwc_pct_of_rev_change = nwc_change_abs / rev_delta
        nwc_pct_of_rev_change = max(-0.20, min(nwc_pct_of_rev_change, 0.20))
    else:
        nwc_pct_of_rev_change = 0.02

    da_pct = (depreciation / revenue) if revenue > 0 else 0.03
    capex_pct = (capex / revenue) if revenue > 0 else 0.04

    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")

    sector_mature_capex = {
        "Technology": 0.06, "Communication Services": 0.07,
        "Consumer Cyclical": 0.06, "Consumer Defensive": 0.04,
        "Healthcare": 0.05, "Financial Services": 0.02,
        "Industrials": 0.05, "Energy": 0.08,
        "Basic Materials": 0.06, "Real Estate": 0.03,
        "Utilities": 0.10,
    }
    mature_capex_pct = sector_mature_capex.get(sector, 0.06)
    if capex_pct <= mature_capex_pct:
        mature_capex_pct = capex_pct

    price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
    shares = info.get("sharesOutstanding", 0) or 0
    cash = info.get("totalCash", 0) or 0
    debt = info.get("totalDebt", 0) or 0
    name = info.get("shortName", ticker.upper())
    market_cap = info.get("marketCap", 0) or 0
    pe = info.get("trailingPE", None)
    ev_ebitda = info.get("enterpriseToEbitda", None)
    beta = info.get("beta", None)
    dividend_yield = info.get("dividendYield", None)
    profit_margin = info.get("profitMargins", None)
    roe = info.get("returnOnEquity", None)
    roa = info.get("returnOnAssets", None)
    revenue_per_share = info.get("revenuePerShare", None)
    fifty_two_high = info.get("fiftyTwoWeekHigh", None)
    fifty_two_low = info.get("fiftyTwoWeekLow", None)

    # Historical data for charts
    hist_revenue = []
    hist_ebit = []
    hist_net_income = []
    hist_fcf = []
    hist_years = []
    for i, col in enumerate(reversed(income.columns)):
        yr_label = str(col.year) if hasattr(col, 'year') else f"Y{i+1}"
        hist_years.append(yr_label)
        hist_revenue.append(safe(income, ["Total Revenue", "Revenue"], col))
        hist_ebit.append(safe(income, ["EBIT", "Operating Income"], col))
        hist_net_income.append(safe(income, ["Net Income", "Net Income Common Stockholders"], col))
        cf_val = safe(cashflow, ["Free Cash Flow"], col)
        if cf_val == 0:
            ocf = safe(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"], col)
            cx = abs(safe(cashflow, ["Capital Expenditure", "Capital Expenditures"], col))
            cf_val = ocf - cx
        hist_fcf.append(cf_val)

    # Historical margins
    hist_margins = []
    for i in range(len(hist_revenue)):
        r = hist_revenue[i]
        if r > 0:
            hist_margins.append({
                "year": hist_years[i],
                "gross": None,
                "ebit": round(hist_ebit[i] / r * 100, 1),
                "net": round(hist_net_income[i] / r * 100, 1),
            })

    # Stock price history (1 year)
    try:
        hist = stock.history(period="1y")
        price_history = []
        if not hist.empty:
            # Sample ~60 data points
            step = max(1, len(hist) // 60)
            for idx in range(0, len(hist), step):
                row = hist.iloc[idx]
                price_history.append({
                    "date": str(hist.index[idx].date()),
                    "close": round(float(row["Close"]), 2),
                })
    except:
        price_history = []

    if shares == 0:
        raise HTTPException(status_code=400, detail=f"No shares outstanding data for '{ticker}'.")

    # Fallback WACC
    risk_free_rate = 0.043
    equity_risk_premium = 0.055
    if beta and beta > 0:
        cost_of_equity = risk_free_rate + beta * equity_risk_premium
    else:
        cost_of_equity = risk_free_rate + 1.0 * equity_risk_premium
    if debt > 0 and interest_expense > 0:
        cost_of_debt = interest_expense / debt
    else:
        cost_of_debt = 0.05
    equity_value_market = market_cap if market_cap > 0 else (price * shares)
    total_capital = equity_value_market + debt
    if total_capital > 0:
        weight_equity = equity_value_market / total_capital
        weight_debt = debt / total_capital
    else:
        weight_equity = 1.0
        weight_debt = 0.0
    calc_wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    calc_wacc = max(0.05, min(calc_wacc, 0.20))

    growth_rates = []
    for i in range(1, len(hist_revenue)):
        if hist_revenue[i - 1] > 0 and hist_revenue[i] > 0:
            gr = (hist_revenue[i] - hist_revenue[i - 1]) / hist_revenue[i - 1]
            growth_rates.append(gr)
    if growth_rates:
        avg_revenue_growth = sum(growth_rates) / len(growth_rates)
        avg_revenue_growth = max(-0.10, min(avg_revenue_growth, 0.30))
    else:
        avg_revenue_growth = 0.05

    return {
        "name": name, "ticker": ticker.upper(), "sector": sector, "industry": industry,
        "price": price, "shares_outstanding": shares, "market_cap": market_cap,
        "cash": cash, "debt": debt, "pe_ratio": pe, "ev_ebitda": ev_ebitda, "beta": beta,
        "dividend_yield": dividend_yield, "profit_margin": profit_margin,
        "roe": roe, "roa": roa, "revenue_per_share": revenue_per_share,
        "fifty_two_high": fifty_two_high, "fifty_two_low": fifty_two_low,
        "latest_revenue": revenue, "latest_ebit": ebit, "net_income": net_income,
        "depreciation": depreciation, "capex": capex,
        "da_pct": da_pct, "capex_pct": capex_pct, "mature_capex_pct": mature_capex_pct,
        "tax_rate": tax_rate, "interest_expense": interest_expense,
        "nwc_pct_of_rev_change": nwc_pct_of_rev_change,
        "hist_revenue": hist_revenue, "hist_ebit": hist_ebit,
        "hist_net_income": hist_net_income, "hist_fcf": hist_fcf,
        "hist_years": hist_years, "hist_margins": hist_margins,
        "price_history": price_history,
        "ebit_margin": (ebit / revenue * 100) if revenue else 0,
        "calc_wacc": round(calc_wacc, 4),
        "calc_revenue_growth": round(avg_revenue_growth, 4),
        "calc_terminal_growth": 0.025,
        "cost_of_equity": round(cost_of_equity, 4),
        "cost_of_debt": round(cost_of_debt, 4),
        "weight_equity": round(weight_equity, 4),
        "weight_debt": round(weight_debt, 4),
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
    }

# ---------------------------------------------------------------------------
# Claude AI
# ---------------------------------------------------------------------------

async def get_ai_assumptions(data: dict) -> dict | None:
    if not CLAUDE_API_KEY:
        return None

    hist_rev_str = ""
    for i, r in enumerate(data["hist_revenue"]):
        hist_rev_str += f"  Year {i+1}: ${r/1e9:.1f}B\n"

    growth_rates_str = ""
    hr = data["hist_revenue"]
    for i in range(1, len(hr)):
        if hr[i-1] > 0:
            gr = (hr[i] - hr[i-1]) / hr[i-1] * 100
            growth_rates_str += f"  Year {i} to {i+1}: {gr:.1f}%\n"

    prompt = f"""You are a senior equity research analyst. First, search the web for the latest news, earnings, analyst estimates, and recent developments for this company. Then, using BOTH the financial data below AND the current news you found, recommend DCF model assumptions for {data['name']} ({data['ticker']}).

COMPANY DATA:
- Sector: {data['sector']}, Industry: {data['industry']}
- Market Cap: ${data['market_cap']/1e9:.1f}B, Price: ${data['price']:.2f}
- Beta: {data['beta'] if data['beta'] else 'N/A'}
- P/E: {f"{data['pe_ratio']:.1f}x" if data['pe_ratio'] else 'N/A'}, EV/EBITDA: {f"{data['ev_ebitda']:.1f}x" if data['ev_ebitda'] else 'N/A'}

FINANCIALS:
- Revenue: ${data['latest_revenue']/1e9:.1f}B, EBIT Margin: {data['ebit_margin']:.1f}%
- CapEx: {data['capex_pct']*100:.1f}% of rev, D&A: {data['da_pct']*100:.1f}% of rev
- Historical Revenue:
{hist_rev_str}- Growth Rates:
{growth_rates_str}- Cash: ${data['cash']/1e9:.1f}B, Debt: ${data['debt']/1e9:.1f}B
- Tax Rate: {data['tax_rate']*100:.1f}%, CoE: {data['cost_of_equity']*100:.1f}%, CoD: {data['cost_of_debt']*100:.1f}%
- Equity Weight: {data['weight_equity']*100:.1f}%, Debt Weight: {data['weight_debt']*100:.1f}%

NOTE: Model uses CapEx fade from {data['capex_pct']*100:.1f}% to {data['mature_capex_pct']*100:.1f}% over projection period.

RESPOND WITH ONLY THIS JSON (no markdown/backticks):
{{"wacc":<num>,"wacc_reasoning":"<1-2 sentences>","revenue_growth":<num>,"revenue_growth_reasoning":"<1-2 sentences>","terminal_growth":<num>,"terminal_growth_reasoning":"<1-2 sentences>","projection_years":<int>,"projection_years_reasoning":"<1 sentence>","overall_analysis":"<2-3 sentences>"}}

IMPORTANT: Factor in recent news, earnings, forward guidance, analyst consensus, and major developments into your assumptions. Reference specific recent events in your reasoning.

Rules: WACC 6-15%, Revenue growth realistic, Terminal growth 1.5-3.5%, Years 5-15. Be specific to THIS company."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01"},
                json={"model":"claude-sonnet-4-20250514","max_tokens":1000,"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":prompt}]},
            )
        if response.status_code != 200:
            print(f"Claude API error: {response.status_code} {response.text}")
            return None
        result = response.json()
        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        text = text.strip().replace("```json","").replace("```","").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        parsed = json.loads(text)
        return {
            "wacc": round(max(5.0, min(float(parsed["wacc"]), 20.0)), 2),
            "wacc_reasoning": parsed.get("wacc_reasoning", ""),
            "revenue_growth": round(max(-10.0, min(float(parsed["revenue_growth"]), 35.0)), 2),
            "revenue_growth_reasoning": parsed.get("revenue_growth_reasoning", ""),
            "terminal_growth": round(max(1.0, min(float(parsed["terminal_growth"]), 4.0)), 2),
            "terminal_growth_reasoning": parsed.get("terminal_growth_reasoning", ""),
            "projection_years": max(5, min(int(parsed.get("projection_years", 10)), 15)),
            "projection_years_reasoning": parsed.get("projection_years_reasoning", ""),
            "overall_analysis": parsed.get("overall_analysis", ""),
        }
    except Exception:
        print(f"Claude AI error: {traceback.format_exc()}")
        return None

# ---------------------------------------------------------------------------
# News via Claude
# ---------------------------------------------------------------------------

async def get_ai_news(ticker: str, company_name: str) -> list | None:
    if not CLAUDE_API_KEY:
        return None

    prompt = f"""You are a financial news analyst. Provide the 6 most important recent news items and developments for {company_name} ({ticker}) that would be relevant to an investor making a buy/hold/sell decision.

For each item, provide a JSON array (no markdown, no backticks):
[
  {{
    "headline": "<concise headline>",
    "summary": "<2-3 sentence summary of the news and its investment implications>",
    "sentiment": "<positive/negative/neutral>",
    "category": "<one of: earnings, product, regulatory, market, management, macro, competitive, partnership>"
  }}
]

Focus on: recent earnings, product launches, regulatory changes, competitive dynamics, management changes, macro factors affecting the company. Be factual and specific. Return ONLY the JSON array."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

        if response.status_code != 200:
            print(f"Claude News API error: {response.status_code} {response.text}")
            return None

        result = response.json()

        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        text = text.strip().replace("```json", "").replace("```", "").strip()
        if not text:
            return None

        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        parsed = json.loads(text)

        if isinstance(parsed, list):
            clean_items = []
            for item in parsed[:8]:
                if not isinstance(item, dict):
                    continue
                clean_items.append({
                    "headline": item.get("headline", "Untitled update"),
                    "summary": item.get("summary", "No summary available."),
                    "sentiment": item.get("sentiment", "neutral"),
                    "category": item.get("category", "market"),
                })
            return clean_items

        return None

    except Exception:
        print(f"Claude News raw parse failure: {text if 'text' in locals() else 'NO_TEXT'}")
        print(f"Claude News error: {traceback.format_exc()}")
        return None

# ---------------------------------------------------------------------------
# DCF calculation with CapEx fade
# ---------------------------------------------------------------------------

def run_dcf(data: dict, wacc: float, terminal_growth: float, revenue_growth: float, years: int) -> dict:
    revenue = data["latest_revenue"]
    ebit = data["latest_ebit"]
    da_pct_start = data["da_pct"]
    capex_pct_start = data["capex_pct"]
    mature_capex_pct = data["mature_capex_pct"]
    tax_rate = data["tax_rate"]
    nwc_pct = data["nwc_pct_of_rev_change"]
    shares = data["shares_outstanding"]
    cash = data["cash"]
    debt = data["debt"]

    if revenue == 0:
        raise HTTPException(status_code=400, detail="Revenue is zero — cannot project.")

    ebit_margin = ebit / revenue if revenue else 0
    mature_da_pct = mature_capex_pct
    if da_pct_start < mature_da_pct:
        mature_da_pct = da_pct_start

    projections = []
    prev_revenue = revenue

    for yr in range(1, years + 1):
        proj_revenue = revenue * ((1 + revenue_growth) ** yr)
        proj_ebit = proj_revenue * ebit_margin
        nopat = proj_ebit * (1 - tax_rate)
        fade_progress = (yr - 1) / max(years - 1, 1)
        capex_pct_yr = capex_pct_start + (mature_capex_pct - capex_pct_start) * fade_progress
        da_pct_yr = da_pct_start + (mature_da_pct - da_pct_start) * fade_progress
        proj_da = proj_revenue * da_pct_yr
        proj_capex = proj_revenue * capex_pct_yr
        rev_increase = proj_revenue - prev_revenue
        proj_nwc_change = rev_increase * nwc_pct
        ufcf = nopat + proj_da - proj_capex - proj_nwc_change
        discount_factor = 1 / ((1 + wacc) ** yr)
        pv = ufcf * discount_factor
        projections.append({
            "year": yr,
            "revenue": round(proj_revenue),
            "ebit": round(proj_ebit),
            "nopat": round(nopat),
            "da": round(proj_da),
            "capex": round(proj_capex),
            "nwc_change": round(proj_nwc_change),
            "ufcf": round(ufcf),
            "discount_factor": round(discount_factor, 4),
            "pv_ufcf": round(pv),
            "capex_pct": round(capex_pct_yr * 100, 1),
        })
        prev_revenue = proj_revenue

    final_ufcf = projections[-1]["ufcf"]
    if wacc <= terminal_growth:
        raise HTTPException(status_code=400, detail="WACC must be greater than terminal growth rate.")

    terminal_value = (final_ufcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    sum_pv_ufcf = sum(p["pv_ufcf"] for p in projections)
    enterprise_value = sum_pv_ufcf + pv_terminal
    equity_value = enterprise_value + cash - debt
    implied_share_price = equity_value / shares if shares else 0
    current_price = data["price"]
    upside = ((implied_share_price - current_price) / current_price * 100) if current_price else 0

    if upside > 15:
        recommendation = "BUY"
    elif upside < -15:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    wacc_range = [round(wacc + d, 4) for d in [-0.02, -0.01, 0, 0.01, 0.02]]
    tg_range = [round(terminal_growth + d, 4) for d in [-0.01, -0.005, 0, 0.005, 0.01]]
    sensitivity = []
    for w in wacc_range:
        row = {"wacc": w, "values": []}
        for tg in tg_range:
            if w <= tg or w <= 0:
                row["values"].append(None)
            else:
                tv = (final_ufcf * (1 + tg)) / (w - tg)
                pv_tv = tv / ((1 + w) ** years)
                spv = sum(p["ufcf"] / ((1 + w) ** p["year"]) for p in projections)
                ev = spv + pv_tv
                eq = ev + cash - debt
                imp = eq / shares if shares else 0
                row["values"].append(round(imp, 2))
        sensitivity.append(row)

    return {
        "projections": projections,
        "terminal_value": round(terminal_value),
        "pv_terminal": round(pv_terminal),
        "sum_pv_ufcf": round(sum_pv_ufcf),
        "enterprise_value": round(enterprise_value),
        "equity_value": round(equity_value),
        "implied_share_price": round(implied_share_price, 2),
        "current_price": current_price,
        "upside_pct": round(upside, 2),
        "recommendation": recommendation,
        "ebit_margin": round(ebit_margin * 100, 2),
        "tax_rate_used": round(tax_rate * 100, 2),
        "capex_pct_start": round(capex_pct_start * 100, 2),
        "capex_pct_mature": round(mature_capex_pct * 100, 2),
        "sensitivity": sensitivity,
        "sensitivity_tg_range": tg_range,
    }

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/lookup")
async def lookup_endpoint(req: TickerLookup):
    try:
        ticker = req.ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required.")
        data = fetch_financials(ticker)
        ai_result = await get_ai_assumptions(data)

        if ai_result:
            assumptions = {
                "wacc": ai_result["wacc"],
                "revenue_growth": ai_result["revenue_growth"],
                "terminal_growth": ai_result["terminal_growth"],
                "projection_years": ai_result["projection_years"],
            }
            ai_reasoning = {
                "wacc": ai_result["wacc_reasoning"],
                "revenue_growth": ai_result["revenue_growth_reasoning"],
                "terminal_growth": ai_result["terminal_growth_reasoning"],
                "projection_years": ai_result["projection_years_reasoning"],
                "overall": ai_result["overall_analysis"],
                "powered_by_ai": True,
            }
        else:
            assumptions = {
                "wacc": round(data["calc_wacc"] * 100, 2),
                "revenue_growth": round(data["calc_revenue_growth"] * 100, 2),
                "terminal_growth": round(data["calc_terminal_growth"] * 100, 2),
                "projection_years": 10,
            }
            ai_reasoning = {
                "wacc": "Calculated using CAPM weighted by capital structure.",
                "revenue_growth": "Based on average historical revenue growth.",
                "terminal_growth": "Standard 2.5% long-term GDP assumption.",
                "projection_years": "Standard 10-year projection.",
                "overall": "AI unavailable. Using formula-based assumptions.",
                "powered_by_ai": False,
            }

        return {
            "company": {
                "name": data["name"],
                "ticker": data["ticker"],
                "sector": data["sector"],
                "industry": data["industry"],
                "price": data["price"],
                "shares_outstanding": data["shares_outstanding"],
                "market_cap": data["market_cap"],
                "cash": data["cash"],
                "debt": data["debt"],
                "pe_ratio": data["pe_ratio"],
                "ev_ebitda": data["ev_ebitda"],
                "beta": data["beta"],
                "dividend_yield": data["dividend_yield"],
                "profit_margin": data["profit_margin"],
                "roe": data["roe"],
                "roa": data["roa"],
                "fifty_two_high": data["fifty_two_high"],
                "fifty_two_low": data["fifty_two_low"],
                "hist_revenue": data["hist_revenue"],
                "hist_ebit": data["hist_ebit"],
                "hist_net_income": data["hist_net_income"],
                "hist_fcf": data["hist_fcf"],
                "hist_years": data["hist_years"],
                "hist_margins": data["hist_margins"],
                "price_history": data["price_history"],
            },
            "auto_assumptions": assumptions,
            "ai_reasoning": ai_reasoning,
            "wacc_breakdown": {
                "risk_free_rate": round(data["risk_free_rate"] * 100, 2),
                "equity_risk_premium": round(data["equity_risk_premium"] * 100, 2),
                "beta": data["beta"],
                "cost_of_equity": round(data["cost_of_equity"] * 100, 2),
                "cost_of_debt": round(data["cost_of_debt"] * 100, 2),
                "weight_equity": round(data["weight_equity"] * 100, 2),
                "weight_debt": round(data["weight_debt"] * 100, 2),
                "tax_rate": round(data["tax_rate"] * 100, 2),
            },
            "model_inputs": {
                "ebit_margin": round(data["ebit_margin"], 2),
                "da_pct": round(data["da_pct"] * 100, 2),
                "capex_pct_current": round(data["capex_pct"] * 100, 2),
                "capex_pct_mature": round(data["mature_capex_pct"] * 100, 2),
                "nwc_pct": round(data["nwc_pct_of_rev_change"] * 100, 2),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR lookup {req.ticker}: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

@app.post("/api/dcf")
def dcf_endpoint(req: DCFRequest):
    try:
        ticker = req.ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required.")
        data = fetch_financials(ticker)
        result = run_dcf(data, req.wacc, req.terminal_growth, req.revenue_growth, req.projection_years)
        return {
            "company": {
                "name": data["name"],
                "ticker": data["ticker"],
                "sector": data["sector"],
                "price": data["price"],
                "shares_outstanding": data["shares_outstanding"],
                "market_cap": data["market_cap"],
                "cash": data["cash"],
                "debt": data["debt"],
                "pe_ratio": data["pe_ratio"],
                "ev_ebitda": data["ev_ebitda"],
                "beta": data["beta"],
                "hist_revenue": data["hist_revenue"],
            },
            "assumptions": {
                "wacc": req.wacc,
                "terminal_growth": req.terminal_growth,
                "revenue_growth": req.revenue_growth,
                "projection_years": req.projection_years,
            },
            "valuation": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR {req.ticker}: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

@app.post("/api/news")
async def news_endpoint(req: NewsRequest):
    try:
        ticker = req.ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required.")

        if not CLAUDE_API_KEY:
            return JSONResponse(
                status_code=503,
                content={"detail": "News feature is not configured. Add CLAUDE_API_KEY in Railway environment variables."},
            )

        stock = yf.Ticker(ticker, session=None)
        name = stock.info.get("shortName", ticker) if stock.info else ticker

        news = await get_ai_news(ticker, name)
        if news is None:
            return JSONResponse(
                status_code=502,
                content={"detail": "News lookup failed. Check Railway logs for the Claude response/parsing error."},
            )

        return {"ticker": ticker, "company": name, "news": news}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR news {req.ticker}: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Server error: {str(e)}"})

@app.get("/api/health")
def health():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""):
    return FileResponse("static/index.html")
