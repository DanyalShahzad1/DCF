from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import math, traceback, httpx, json, os
from datetime import datetime, timezone

yf.set_tz_cache_location("/tmp/yf_cache")
app = FastAPI()
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

class TickerRequest(BaseModel):
    ticker: str

def fetch_financials(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker, session=None)
        info = stock.info
        if not info or len(info) <= 1:
            raise Exception("Empty response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for '{ticker}': {str(e)}")

    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")

    try:
        income = stock.financials
        cashflow = stock.cashflow
        balance = stock.balance_sheet
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch statements: {str(e)}")

    def safe(df, keys, col=None):
        if df is None or df.empty: return 0.0
        if col is None: col = df.columns[0]
        for k in keys:
            if k in df.index:
                val = df.loc[k, col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return float(val)
        return 0.0

    revenue = safe(income, ["Total Revenue", "Revenue"])
    ebit = safe(income, ["EBIT", "Operating Income"])
    net_income = safe(income, ["Net Income", "Net Income Common Stockholders"])
    depreciation = safe(cashflow, ["Depreciation And Amortization", "Depreciation & Amortization"])
    capex = abs(safe(cashflow, ["Capital Expenditure", "Capital Expenditures"]))
    tax_provision = safe(income, ["Tax Provision", "Income Tax Expense"])
    pretax_income = safe(income, ["Pretax Income", "Income Before Tax"])
    interest_expense = abs(safe(income, ["Interest Expense", "Interest Expense Non Operating"]))
    tax_rate = max(0, min(tax_provision / pretax_income, 0.40)) if pretax_income != 0 else 0.21

    fcf = safe(cashflow, ["Free Cash Flow"])
    if fcf == 0:
        op_cf = safe(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        fcf = op_cf - capex

    def wc(col):
        return safe(balance, ["Current Assets", "Total Current Assets"], col) - safe(balance, ["Current Liabilities", "Total Current Liabilities"], col)

    nwc_change = wc(balance.columns[0]) - wc(balance.columns[1]) if balance is not None and not balance.empty and len(balance.columns) >= 2 else 0.0

    price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
    shares = info.get("sharesOutstanding", 0) or 0
    cash = info.get("totalCash", 0) or 0
    debt = info.get("totalDebt", 0) or 0
    market_cap = info.get("marketCap", 0) or 0
    beta = info.get("beta")
    pe = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    ev_ebitda = info.get("enterpriseToEbitda")
    pb = info.get("priceToBook")
    ps = info.get("priceToSalesTrailing12Months")
    dividend_yield = info.get("dividendYield", 0) or 0
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    name = info.get("shortName", ticker.upper())
    week52_high = info.get("fiftyTwoWeekHigh", 0) or 0
    week52_low = info.get("fiftyTwoWeekLow", 0) or 0
    ma50 = info.get("fiftyDayAverage", 0) or 0
    ma200 = info.get("twoHundredDayAverage", 0) or 0
    rec_mean = info.get("recommendationMean")
    rec_key = info.get("recommendationKey", "")
    n_analysts = info.get("numberOfAnalystOpinions", 0) or 0
    target_mean = info.get("targetMeanPrice", 0) or 0
    target_high = info.get("targetHighPrice", 0) or 0
    target_low = info.get("targetLowPrice", 0) or 0

    hist_revenue, hist_ebit, hist_ni, hist_fcf, hist_years = [], [], [], [], []
    if income is not None and not income.empty:
        for i, col in enumerate(reversed(income.columns)):
            hist_years.append(str(col.year) if hasattr(col, 'year') else f"Y{i+1}")
            hist_revenue.append(safe(income, ["Total Revenue", "Revenue"], col))
            hist_ebit.append(safe(income, ["EBIT", "Operating Income"], col))
            hist_ni.append(safe(income, ["Net Income", "Net Income Common Stockholders"], col))
            cf = safe(cashflow, ["Free Cash Flow"], col) if cashflow is not None and not cashflow.empty else 0
            if cf == 0 and cashflow is not None and not cashflow.empty:
                cf = safe(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"], col) - abs(safe(cashflow, ["Capital Expenditure", "Capital Expenditures"], col))
            hist_fcf.append(cf)

    try:
        hist = stock.history(period="1y")
        step = max(1, len(hist) // 60)
        price_history = [{"date": str(hist.index[idx].date()), "close": round(float(hist.iloc[idx]["Close"]), 2)} for idx in range(0, len(hist), step)] if not hist.empty else []
    except:
        price_history = []

    rfr = 0.043; erp = 0.055
    coe = rfr + (beta if beta and beta > 0 else 1.0) * erp
    cod = interest_expense / debt if debt > 0 and interest_expense > 0 else 0.05
    evm = market_cap if market_cap > 0 else price * shares
    tc = evm + debt
    we = evm / tc if tc > 0 else 1.0
    wd = debt / tc if tc > 0 else 0.0
    wacc = max(0.05, min((we * coe) + (wd * cod * (1 - tax_rate)), 0.20))

    grs = [(hist_revenue[i] - hist_revenue[i-1]) / hist_revenue[i-1]
           for i in range(1, len(hist_revenue)) if hist_revenue[i-1] > 0 and hist_revenue[i] > 0]
    avg_growth = max(-0.10, min(sum(grs)/len(grs), 0.30)) if grs else 0.05

    da_pct = (depreciation / revenue) if revenue > 0 else 0.03
    capex_pct = (capex / revenue) if revenue > 0 else 0.04
    sector_mature = {"Technology":0.06,"Communication Services":0.07,"Consumer Cyclical":0.06,"Consumer Defensive":0.04,"Healthcare":0.05,"Financial Services":0.02,"Industrials":0.05,"Energy":0.08,"Basic Materials":0.06,"Real Estate":0.03,"Utilities":0.10}
    mature_capex_pct = sector_mature.get(sector, 0.06)
    if capex_pct <= mature_capex_pct: mature_capex_pct = capex_pct
    nwc_pct = max(-0.20, min(nwc_change / (revenue * avg_growth), 0.20)) if revenue > 0 and avg_growth > 0 else 0.02
    ebit_margin = (ebit / revenue) if revenue > 0 else 0
    da_mature = mature_capex_pct if da_pct >= mature_capex_pct else da_pct
    terminal_growth = 0.025

    dcf_implied = 0
    if revenue > 0 and shares > 0 and wacc > terminal_growth:
        proj_fcfs = []
        prev = revenue
        for yr in range(1, 11):
            r = revenue * ((1 + avg_growth) ** yr)
            e = r * ebit_margin
            nopat = e * (1 - tax_rate)
            f = (yr - 1) / 9
            da_yr = r * (da_pct + (da_mature - da_pct) * f)
            cx_yr = r * (capex_pct + (mature_capex_pct - capex_pct) * f)
            nwc_yr = (r - prev) * nwc_pct
            ufcf = nopat + da_yr - cx_yr - nwc_yr
            df = 1 / ((1 + wacc) ** yr)
            proj_fcfs.append(ufcf * df)
            prev = r
        sum_pv = sum(proj_fcfs)
        final_ufcf = revenue * ((1 + avg_growth) ** 10) * ebit_margin * (1 - tax_rate)
        tv = (final_ufcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_tv = tv / ((1 + wacc) ** 10)
        ev = sum_pv + pv_tv
        equity_val = ev + cash - debt
        dcf_implied = equity_val / shares

    return {
        "name": name, "ticker": ticker.upper(), "sector": sector, "industry": industry,
        "price": price, "shares": shares, "market_cap": market_cap,
        "cash": cash, "debt": debt, "beta": beta,
        "pe": pe, "fwd_pe": fwd_pe, "ev_ebitda": ev_ebitda, "pb": pb, "ps": ps,
        "dividend_yield": dividend_yield,
        "revenue": revenue, "ebit": ebit, "net_income": net_income, "fcf": fcf,
        "ebit_margin": ebit_margin,
        "week52_high": week52_high, "week52_low": week52_low,
        "ma50": ma50, "ma200": ma200,
        "rec_mean": rec_mean, "rec_key": rec_key,
        "n_analysts": n_analysts, "target_mean": target_mean,
        "target_high": target_high, "target_low": target_low,
        "hist_revenue": hist_revenue, "hist_ebit": hist_ebit,
        "hist_ni": hist_ni, "hist_fcf": hist_fcf, "hist_years": hist_years,
        "price_history": price_history,
        "dcf_implied": round(dcf_implied, 2),
        "wacc": round(wacc, 4),
        "avg_growth": round(avg_growth, 4),
    }

def compute_signals(d: dict) -> dict:
    signals = []

    # 1. DCF Valuation
    if d["dcf_implied"] > 0 and d["price"] > 0:
        upside = (d["dcf_implied"] - d["price"]) / d["price"] * 100
        if upside > 20:
            sig = "buy"; detail = f"Intrinsic value ~${d['dcf_implied']:.0f} vs ${d['price']:.0f} market price ({upside:+.0f}% upside)"
        elif upside < -20:
            sig = "sell"; detail = f"Intrinsic value ~${d['dcf_implied']:.0f} vs ${d['price']:.0f} market price ({upside:+.0f}% downside)"
        else:
            sig = "hold"; detail = f"Intrinsic value ~${d['dcf_implied']:.0f} roughly in line with ${d['price']:.0f} market price ({upside:+.0f}%)"
        signals.append({"name": "DCF Valuation", "signal": sig, "detail": detail})

    # 2. Analyst Consensus
    if d["n_analysts"] >= 3 and d["rec_mean"] is not None:
        rm = d["rec_mean"]
        if rm <= 2.0: sig = "buy"; label = "Strong Buy"
        elif rm <= 2.8: sig = "buy"; label = "Moderate Buy"
        elif rm <= 3.2: sig = "hold"; label = "Hold"
        elif rm <= 4.0: sig = "sell"; label = "Moderate Sell"
        else: sig = "sell"; label = "Strong Sell"
        upside_to_target = (d["target_mean"] - d["price"]) / d["price"] * 100 if d["target_mean"] and d["price"] else 0
        detail = f"{d['n_analysts']} analysts rate it {label}. Mean price target ${d['target_mean']:.0f} ({upside_to_target:+.0f}% vs current)"
        signals.append({"name": "Analyst Consensus", "signal": sig, "detail": detail})

    # 3. Financial Health
    health_score = 0; health_flags = []
    if d["fcf"] > 0: health_score += 1
    else: health_flags.append("negative free cash flow")
    if d["net_income"] > 0: health_score += 1
    else: health_flags.append("net loss")
    if d["debt"] > 0 and d["market_cap"] > 0 and (d["debt"] / d["market_cap"]) < 1.0: health_score += 1
    elif d["debt"] > d["market_cap"] * 1.5: health_flags.append("high debt load")
    if d["ebit_margin"] > 0.10: health_score += 1
    elif d["ebit_margin"] < 0: health_flags.append("negative operating margin")
    if len(d["hist_revenue"]) >= 2 and d["hist_revenue"][-1] > 0 and d["hist_revenue"][0] > d["hist_revenue"][-1]: health_score += 1
    elif len(d["hist_revenue"]) >= 2 and d["hist_revenue"][-1] > 0 and d["hist_revenue"][0] < d["hist_revenue"][-1] * 0.9: health_flags.append("declining revenue")

    if health_score >= 4: sig = "buy"; detail = "Strong financials — positive FCF, profitable, manageable debt, healthy margins"
    elif len(health_flags) >= 2: sig = "sell"; detail = f"Caution: {', '.join(health_flags)}"
    elif len(health_flags) == 1: sig = "hold"; detail = f"Mixed financials — note: {health_flags[0]}"
    else: sig = "hold"; detail = "Adequate financials with no major red flags"
    signals.append({"name": "Financial Health", "signal": sig, "detail": detail})

    # 4. Momentum
    if d["price"] > 0 and d["week52_high"] > 0 and d["week52_low"] > 0:
        week_range = d["week52_high"] - d["week52_low"]
        position = (d["price"] - d["week52_low"]) / week_range if week_range > 0 else 0.5
        above_ma200 = d["price"] > d["ma200"] if d["ma200"] > 0 else None
        above_ma50 = d["price"] > d["ma50"] if d["ma50"] > 0 else None
        if position > 0.65 and above_ma200 and above_ma50:
            sig = "buy"; detail = f"Strong uptrend — in top {(position*100):.0f}% of 52-week range, above both 50 and 200-day moving averages"
        elif position < 0.35 and above_ma200 == False:
            sig = "sell"; detail = f"Downtrend — in bottom {(position*100):.0f}% of 52-week range, below 200-day moving average"
        else:
            sig = "hold"
            ma_note = "above" if above_ma200 else "below"
            detail = f"Neutral momentum — at {(position*100):.0f}% of 52-week range, {ma_note} 200-day moving average"
        signals.append({"name": "Price Momentum", "signal": sig, "detail": detail})

    # 5. Relative Valuation
    sector_avg_pe = {"Technology": 28, "Healthcare": 22, "Consumer Cyclical": 20, "Consumer Defensive": 22,
                     "Communication Services": 20, "Industrials": 20, "Energy": 14, "Financial Services": 15,
                     "Basic Materials": 16, "Real Estate": 30, "Utilities": 18}
    avg_pe = sector_avg_pe.get(d["sector"], 20)
    multi_score = 0; multi_flags = []
    if d["pe"] and d["pe"] > 0:
        if d["pe"] < avg_pe * 0.8: multi_score += 1
        elif d["pe"] > avg_pe * 1.5: multi_flags.append(f"P/E {d['pe']:.1f}x above sector avg ~{avg_pe}x")
    if d["ev_ebitda"] and d["ev_ebitda"] > 0:
        if d["ev_ebitda"] < 15: multi_score += 1
        elif d["ev_ebitda"] > 30: multi_flags.append(f"EV/EBITDA {d['ev_ebitda']:.1f}x looks elevated")
    if d["pb"] and d["pb"] > 0:
        if d["pb"] < 3: multi_score += 1
        elif d["pb"] > 10: multi_flags.append(f"P/B {d['pb']:.1f}x looks elevated")

    if multi_score >= 2: sig = "buy"; detail = f"Trading at a discount vs peers — P/E {d['pe']:.1f}x vs sector avg ~{avg_pe}x" if d["pe"] else "Multiples look attractive vs sector peers"
    elif len(multi_flags) >= 2: sig = "sell"; detail = "; ".join(multi_flags)
    else:
        sig = "hold"
        parts = [x for x in [f"P/E {d['pe']:.1f}x" if d["pe"] else "", f"EV/EBITDA {d['ev_ebitda']:.1f}x" if d["ev_ebitda"] else ""] if x]
        detail = f"Valuation roughly in line with sector — {', '.join(parts)}" if parts else "Insufficient multiples data"
    signals.append({"name": "Relative Valuation", "signal": sig, "detail": detail})

    buy_count = sum(1 for s in signals if s["signal"] == "buy")
    sell_count = sum(1 for s in signals if s["signal"] == "sell")
    hold_count = sum(1 for s in signals if s["signal"] == "hold")
    total = len(signals)

    if buy_count > sell_count and buy_count >= total * 0.5:
        verdict = "BUY"; confidence = round((buy_count / total) * 100)
    elif sell_count > buy_count and sell_count >= total * 0.5:
        verdict = "SELL"; confidence = round((sell_count / total) * 100)
    else:
        verdict = "HOLD"; confidence = round(((hold_count + min(buy_count, sell_count)) / total) * 100)

    return {"verdict": verdict, "confidence": confidence,
            "buy_count": buy_count, "sell_count": sell_count, "hold_count": hold_count,
            "total_signals": total, "signals": signals}

async def claude_no_search(prompt: str, max_tokens: int = 400) -> str | None:
    if not CLAUDE_API_KEY: return None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json", "x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]})
            if resp.status_code != 200: return None
            result = resp.json()
            return "".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text").strip()
    except: return None

async def get_plain_english_summary(d: dict, signals: dict) -> str:
    verdict = signals["verdict"]
    buy = signals["buy_count"]; sell = signals["sell_count"]; total = signals["total_signals"]
    signal_lines = "\n".join(f"- {s['name']}: {s['signal'].upper()} — {s['detail']}" for s in signals["signals"])
    prompt = f"""You are explaining a stock analysis to a retail investor with zero finance knowledge. Be direct, plain English, no jargon.

Company: {d['name']} ({d['ticker']}) — {d['sector']}
Verdict: {verdict} ({buy}/{total} signals bullish, {sell}/{total} bearish)
Signals:
{signal_lines}

Write exactly 2-3 sentences. Lead with the bottom line verdict. Mention the strongest reason for it, then one key risk. No bullet points, no headers. Sound like a knowledgeable friend, not a financial advisor. End with: "This is not financial advice — always do your own research."
"""
    text = await claude_no_search(prompt, 250)
    if not text:
        if verdict == "BUY": return f"{d['name']} looks like a solid opportunity — most signals point to upside from here. Keep in mind that no investment is guaranteed, and markets can be unpredictable. This is not financial advice — always do your own research."
        elif verdict == "SELL": return f"{d['name']} is showing several warning signs worth paying attention to. The data suggests this may not be the best entry point right now. This is not financial advice — always do your own research."
        else: return f"{d['name']} looks fairly valued right now — not an obvious buy or sell based on available data. It might be worth watching for a clearer signal. This is not financial advice — always do your own research."
    return text

@app.post("/api/analyze")
async def analyze(req: TickerRequest):
    try:
        ticker = req.ticker.strip().upper()
        if not ticker: raise HTTPException(status_code=400, detail="Ticker required.")
        d = fetch_financials(ticker)
        signals = compute_signals(d)
        summary = await get_plain_english_summary(d, signals)

        sector_avg_pe = {"Technology": 28, "Healthcare": 22, "Consumer Cyclical": 20, "Consumer Defensive": 22,
                         "Communication Services": 20, "Industrials": 20, "Energy": 14, "Financial Services": 15,
                         "Basic Materials": 16, "Real Estate": 30, "Utilities": 18}

        def fmt_b(n):
            if not n or n == 0: return "N/A"
            if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
            if abs(n) >= 1e9: return f"${n/1e9:.2f}B"
            if abs(n) >= 1e6: return f"${n/1e6:.1f}M"
            return f"${n:,.0f}"

        return {
            "company": {
                "name": d["name"], "ticker": d["ticker"],
                "sector": d["sector"], "industry": d["industry"],
                "price": d["price"], "market_cap": fmt_b(d["market_cap"]),
                "pe": round(d["pe"], 1) if d["pe"] else None,
                "fwd_pe": round(d["fwd_pe"], 1) if d["fwd_pe"] else None,
                "ev_ebitda": round(d["ev_ebitda"], 1) if d["ev_ebitda"] else None,
                "beta": round(d["beta"], 2) if d["beta"] else None,
                "dividend_yield": round(d["dividend_yield"] * 100, 2) if d["dividend_yield"] else None,
                "week52_high": d["week52_high"], "week52_low": d["week52_low"],
                "ma50": round(d["ma50"], 2) if d["ma50"] else None,
                "ma200": round(d["ma200"], 2) if d["ma200"] else None,
                "target_mean": round(d["target_mean"], 2) if d["target_mean"] else None,
                "n_analysts": d["n_analysts"],
                "sector_pe": sector_avg_pe.get(d["sector"], 20),
            },
            "chart_data": {
                "price_history": d["price_history"],
                "hist_years": d["hist_years"],
                "hist_revenue": d["hist_revenue"],
                "hist_fcf": d["hist_fcf"],
                "hist_ni": d["hist_ni"],
            },
            "analysis": {
                "verdict": signals["verdict"],
                "confidence": signals["confidence"],
                "buy_count": signals["buy_count"],
                "sell_count": signals["sell_count"],
                "hold_count": signals["hold_count"],
                "total_signals": signals["total_signals"],
                "signals": signals["signals"],
                "summary": summary,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException: raise
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/health")
def health(): return {"status": "ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""): return FileResponse("static/index.html")
