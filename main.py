from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import math
import traceback

# Fix for cloud environments
yf.set_tz_cache_location("/tmp/yf_cache")

app = FastAPI()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DCFRequest(BaseModel):
    ticker: str
    wacc: float
    terminal_growth: float
    revenue_growth: float
    projection_years: int = 5

class TickerLookup(BaseModel):
    ticker: str

# ---------------------------------------------------------------------------
# Helper: pull financials from yfinance
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

    if pretax_income != 0:
        tax_rate = max(0, min(tax_provision / pretax_income, 0.40))
    else:
        tax_rate = 0.21

    # Compute NWC change as % of revenue change (more stable)
    def wc(col):
        ca = safe(balance, ["Current Assets", "Total Current Assets"], col)
        cl = safe(balance, ["Current Liabilities", "Total Current Liabilities"], col)
        return ca - cl

    nwc_change_abs = 0.0
    if len(balance.columns) >= 2:
        nwc_change_abs = wc(balance.columns[0]) - wc(balance.columns[1])

    # Revenue from prior year for delta
    rev_prior = 0.0
    if len(income.columns) >= 2:
        rev_prior = safe(income, ["Total Revenue", "Revenue"], income.columns[1])

    rev_delta = revenue - rev_prior if rev_prior > 0 else revenue
    # NWC as % of revenue change (how much working capital grows per $ of revenue growth)
    if rev_delta != 0 and abs(rev_delta) > 1e6:
        nwc_pct_of_rev_change = nwc_change_abs / rev_delta
        # Cap this to a reasonable range — can be negative (source of cash) or positive (use of cash)
        nwc_pct_of_rev_change = max(-0.20, min(nwc_pct_of_rev_change, 0.20))
    else:
        nwc_pct_of_rev_change = 0.02  # default 2%

    # D&A and CapEx as % of revenue (much more stable for projections)
    da_pct = (depreciation / revenue) if revenue > 0 else 0.03
    capex_pct = (capex / revenue) if revenue > 0 else 0.04

    price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
    shares = info.get("sharesOutstanding", 0) or 0
    cash = info.get("totalCash", 0) or 0
    debt = info.get("totalDebt", 0) or 0
    name = info.get("shortName", ticker.upper())
    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", 0) or 0
    pe = info.get("trailingPE", None)
    ev_ebitda = info.get("enterpriseToEbitda", None)
    beta = info.get("beta", None)

    # Historical revenue for growth calc
    hist_revenue = []
    for col in reversed(income.columns):
        r = safe(income, ["Total Revenue", "Revenue"], col)
        hist_revenue.append(r)

    if shares == 0:
        raise HTTPException(status_code=400, detail=f"No shares outstanding data for '{ticker}'.")

    # --- Auto-calculate assumptions ---

    # 1. Revenue growth (average historical)
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

    # 2. WACC (CAPM-based)
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

    terminal_growth = 0.025

    return {
        "name": name, "ticker": ticker.upper(), "sector": sector,
        "price": price, "shares_outstanding": shares, "market_cap": market_cap,
        "cash": cash, "debt": debt, "pe_ratio": pe, "ev_ebitda": ev_ebitda,
        "beta": beta,
        "latest_revenue": revenue, "latest_ebit": ebit,
        "depreciation": depreciation, "capex": capex,
        "da_pct": da_pct, "capex_pct": capex_pct,
        "tax_rate": tax_rate, "interest_expense": interest_expense,
        "nwc_pct_of_rev_change": nwc_pct_of_rev_change,
        "hist_revenue": hist_revenue,
        # Auto-calculated assumptions
        "calc_wacc": round(calc_wacc, 4),
        "calc_revenue_growth": round(avg_revenue_growth, 4),
        "calc_terminal_growth": terminal_growth,
        # WACC components
        "cost_of_equity": round(cost_of_equity, 4),
        "cost_of_debt": round(cost_of_debt, 4),
        "weight_equity": round(weight_equity, 4),
        "weight_debt": round(weight_debt, 4),
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
    }

# ---------------------------------------------------------------------------
# DCF calculation (FIXED)
# ---------------------------------------------------------------------------

def run_dcf(data: dict, wacc: float, terminal_growth: float, revenue_growth: float, years: int) -> dict:
    revenue = data["latest_revenue"]
    ebit = data["latest_ebit"]
    da_pct = data["da_pct"]
    capex_pct = data["capex_pct"]
    tax_rate = data["tax_rate"]
    nwc_pct = data["nwc_pct_of_rev_change"]
    shares = data["shares_outstanding"]
    cash = data["cash"]
    debt = data["debt"]

    if revenue == 0:
        raise HTTPException(status_code=400, detail="Revenue is zero — cannot project.")

    ebit_margin = ebit / revenue if revenue else 0

    # --- Project free cash flows ---
    projections = []
    prev_revenue = revenue

    for yr in range(1, years + 1):
        proj_revenue = revenue * ((1 + revenue_growth) ** yr)
        proj_ebit = proj_revenue * ebit_margin
        nopat = proj_ebit * (1 - tax_rate)

        # D&A and CapEx as % of projected revenue (stable scaling)
        proj_da = proj_revenue * da_pct
        proj_capex = proj_revenue * capex_pct

        # NWC change = % of the incremental revenue this year
        rev_increase = proj_revenue - prev_revenue
        proj_nwc_change = rev_increase * nwc_pct

        # UFCF = NOPAT + D&A - CapEx - Change in NWC
        ufcf = nopat + proj_da - proj_capex - proj_nwc_change

        discount_factor = 1 / ((1 + wacc) ** yr)
        pv = ufcf * discount_factor

        projections.append({
            "year": yr, "revenue": round(proj_revenue), "ebit": round(proj_ebit),
            "nopat": round(nopat), "da": round(proj_da), "capex": round(proj_capex),
            "nwc_change": round(proj_nwc_change), "ufcf": round(ufcf),
            "discount_factor": round(discount_factor, 4), "pv_ufcf": round(pv),
        })

        prev_revenue = proj_revenue

    # --- Terminal value ---
    final_ufcf = projections[-1]["ufcf"]
    if wacc <= terminal_growth:
        raise HTTPException(status_code=400, detail="WACC must be greater than terminal growth rate.")

    # TV = FCF_(n+1) / (WACC - g)  where FCF_(n+1) = final_ufcf * (1 + g)
    terminal_value = (final_ufcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)

    # --- Enterprise & equity value ---
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

    # --- Sensitivity table ---
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
        "projections": projections, "terminal_value": round(terminal_value),
        "pv_terminal": round(pv_terminal), "sum_pv_ufcf": round(sum_pv_ufcf),
        "enterprise_value": round(enterprise_value), "equity_value": round(equity_value),
        "implied_share_price": round(implied_share_price, 2), "current_price": current_price,
        "upside_pct": round(upside, 2), "recommendation": recommendation,
        "ebit_margin": round(ebit_margin * 100, 2), "tax_rate_used": round(tax_rate * 100, 2),
        "da_pct_used": round(da_pct * 100, 2), "capex_pct_used": round(capex_pct * 100, 2),
        "nwc_pct_used": round(nwc_pct * 100, 2),
        "sensitivity": sensitivity, "sensitivity_tg_range": tg_range,
    }

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/lookup")
def lookup_endpoint(req: TickerLookup):
    try:
        ticker = req.ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required.")

        data = fetch_financials(ticker)

        return {
            "company": {
                "name": data["name"], "ticker": data["ticker"], "sector": data["sector"],
                "price": data["price"], "shares_outstanding": data["shares_outstanding"],
                "market_cap": data["market_cap"], "cash": data["cash"], "debt": data["debt"],
                "pe_ratio": data["pe_ratio"], "ev_ebitda": data["ev_ebitda"],
                "beta": data["beta"], "hist_revenue": data["hist_revenue"],
            },
            "auto_assumptions": {
                "wacc": round(data["calc_wacc"] * 100, 2),
                "revenue_growth": round(data["calc_revenue_growth"] * 100, 2),
                "terminal_growth": round(data["calc_terminal_growth"] * 100, 2),
            },
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
                "ebit_margin": round((data["latest_ebit"] / data["latest_revenue"] * 100) if data["latest_revenue"] else 0, 2),
                "da_pct": round(data["da_pct"] * 100, 2),
                "capex_pct": round(data["capex_pct"] * 100, 2),
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
                "name": data["name"], "ticker": data["ticker"], "sector": data["sector"],
                "price": data["price"], "shares_outstanding": data["shares_outstanding"],
                "market_cap": data["market_cap"], "cash": data["cash"], "debt": data["debt"],
                "pe_ratio": data["pe_ratio"], "ev_ebitda": data["ev_ebitda"],
                "beta": data["beta"], "hist_revenue": data["hist_revenue"],
            },
            "assumptions": {
                "wacc": req.wacc, "terminal_growth": req.terminal_growth,
                "revenue_growth": req.revenue_growth, "projection_years": req.projection_years,
            },
            "valuation": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR for ticker {req.ticker}: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Server error analyzing '{req.ticker}': {str(e)}"})

@app.get("/api/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""):
    return FileResponse("static/index.html")
