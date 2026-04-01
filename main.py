from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yfinance as yf
import math
import os

app = FastAPI()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DCFRequest(BaseModel):
    ticker: str
    wacc: float            # e.g. 0.10 for 10%
    terminal_growth: float # e.g. 0.025 for 2.5%
    revenue_growth: float  # e.g. 0.05 for 5%
    projection_years: int = 5

# ---------------------------------------------------------------------------
# Helper: pull financials from yfinance
# ---------------------------------------------------------------------------

def fetch_financials(ticker: str) -> dict:
    stock = yf.Ticker(ticker)

    info = stock.info
    if not info or info.get("regularMarketPrice") is None:
        # Try currentPrice as fallback
        if info.get("currentPrice") is None:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found or has no data.")

    income = stock.financials          # annual income statement
    cashflow = stock.cashflow          # annual cash flow
    balance = stock.balance_sheet      # annual balance sheet

    if income.empty or cashflow.empty or balance.empty:
        raise HTTPException(status_code=404, detail=f"No financial statements found for '{ticker}'.")

    # --- Latest year data ---
    latest_col = income.columns[0]

    def safe(df, keys, col=None):
        """Return first matching row value, or 0."""
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

    # Effective tax rate
    if pretax_income != 0:
        tax_rate = max(0, min(tax_provision / pretax_income, 0.40))
    else:
        tax_rate = 0.21  # default US corporate

    # Change in net working capital (approx)
    def wc(col):
        ca = safe(balance, ["Current Assets", "Total Current Assets"], col)
        cl = safe(balance, ["Current Liabilities", "Total Current Liabilities"], col)
        return ca - cl

    if len(balance.columns) >= 2:
        nwc_change = wc(balance.columns[0]) - wc(balance.columns[1])
    else:
        nwc_change = 0.0

    # Market data
    price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
    shares = info.get("sharesOutstanding", 0)
    cash = info.get("totalCash", 0) or 0
    debt = info.get("totalDebt", 0) or 0
    name = info.get("shortName", ticker.upper())
    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", 0) or 0
    pe = info.get("trailingPE", None)
    ev_ebitda = info.get("enterpriseToEbitda", None)

    # Historical revenue (for context)
    hist_revenue = []
    for col in reversed(income.columns):
        r = safe(income, ["Total Revenue", "Revenue"], col)
        hist_revenue.append(r)

    return {
        "name": name,
        "ticker": ticker.upper(),
        "sector": sector,
        "price": price,
        "shares_outstanding": shares,
        "market_cap": market_cap,
        "cash": cash,
        "debt": debt,
        "pe_ratio": pe,
        "ev_ebitda": ev_ebitda,
        "latest_revenue": revenue,
        "latest_ebit": ebit,
        "depreciation": depreciation,
        "capex": capex,
        "tax_rate": tax_rate,
        "nwc_change": nwc_change,
        "hist_revenue": hist_revenue,
    }

# ---------------------------------------------------------------------------
# DCF calculation
# ---------------------------------------------------------------------------

def run_dcf(data: dict, wacc: float, terminal_growth: float, revenue_growth: float, years: int) -> dict:
    revenue = data["latest_revenue"]
    ebit = data["latest_ebit"]
    depreciation = data["depreciation"]
    capex = data["capex"]
    tax_rate = data["tax_rate"]
    nwc_change = data["nwc_change"]
    shares = data["shares_outstanding"]
    cash = data["cash"]
    debt = data["debt"]

    if revenue == 0:
        raise HTTPException(status_code=400, detail="Revenue is zero — cannot project.")

    ebit_margin = ebit / revenue if revenue else 0

    # --- Project free cash flows ---
    projections = []
    for yr in range(1, years + 1):
        proj_revenue = revenue * ((1 + revenue_growth) ** yr)
        proj_ebit = proj_revenue * ebit_margin
        nopat = proj_ebit * (1 - tax_rate)
        # Scale D&A and CapEx with revenue growth
        scale = ((1 + revenue_growth) ** yr)
        proj_da = depreciation * scale
        proj_capex = capex * scale
        proj_nwc = nwc_change * (1 + revenue_growth)  # simplified
        ufcf = nopat + proj_da - proj_capex - proj_nwc
        discount_factor = 1 / ((1 + wacc) ** yr)
        pv = ufcf * discount_factor

        projections.append({
            "year": yr,
            "revenue": round(proj_revenue),
            "ebit": round(proj_ebit),
            "nopat": round(nopat),
            "da": round(proj_da),
            "capex": round(proj_capex),
            "nwc_change": round(proj_nwc),
            "ufcf": round(ufcf),
            "discount_factor": round(discount_factor, 4),
            "pv_ufcf": round(pv),
        })

    # --- Terminal value ---
    final_ufcf = projections[-1]["ufcf"]
    if wacc <= terminal_growth:
        raise HTTPException(status_code=400, detail="WACC must be greater than terminal growth rate.")

    terminal_value = (final_ufcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)

    # --- Enterprise & equity value ---
    sum_pv_ufcf = sum(p["pv_ufcf"] for p in projections)
    enterprise_value = sum_pv_ufcf + pv_terminal
    equity_value = enterprise_value + cash - debt
    implied_share_price = equity_value / shares if shares else 0

    current_price = data["price"]
    upside = ((implied_share_price - current_price) / current_price * 100) if current_price else 0

    # Recommendation
    if upside > 15:
        recommendation = "BUY"
    elif upside < -15:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    # --- Sensitivity table (WACC vs Terminal Growth) ---
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
                # Recalculate PV of UFCFs with this WACC
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
        "sensitivity": sensitivity,
        "sensitivity_tg_range": tg_range,
    }

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/dcf")
def dcf_endpoint(req: DCFRequest):
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

@app.get("/api/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Serve frontend (static files)
# ---------------------------------------------------------------------------

# Serve static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Catch-all: serve index.html for any non-API route
@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""):
    return FileResponse("static/index.html")
