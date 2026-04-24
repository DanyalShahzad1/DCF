from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import math, traceback, httpx, json, os, sqlite3, asyncio
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

yf.set_tz_cache_location("/tmp/yf_cache")
app = FastAPI()
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
DB_PATH = "/tmp/stockwise.db"

# ── DATABASE ──────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS watchlist (
            session_id TEXT, ticker TEXT, added_at TEXT,
            PRIMARY KEY (session_id, ticker))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS signal_history (
            ticker TEXT, verdict TEXT, confidence INTEGER,
            buy_count INTEGER, sell_count INTEGER, hold_count INTEGER,
            recorded_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS catalyst_cache (
            ticker TEXT PRIMARY KEY, data TEXT, cached_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS analysis_cache (
            ticker TEXT PRIMARY KEY, data TEXT, cached_at TEXT)""")
        conn.commit()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ── MODELS ────────────────────────────────────────────────────────────────
class TickerRequest(BaseModel):
    ticker: str

class WatchlistRequest(BaseModel):
    session_id: str
    ticker: str

class WatchlistGetRequest(BaseModel):
    session_id: str

class CompareRequest(BaseModel):
    ticker1: str
    ticker2: str

class CatalystRequest(BaseModel):
    ticker: str

# ── FINANCIALS ────────────────────────────────────────────────────────────
def fetch_financials(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker, session=None)
        info = stock.info
        if not info or len(info) <= 1: raise Exception("Empty response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for '{ticker}': {str(e)}")

    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")

    try:
        income = stock.financials; cashflow = stock.cashflow; balance = stock.balance_sheet
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

    revenue = safe(income, ["Total Revenue","Revenue"])
    ebit = safe(income, ["EBIT","Operating Income"])
    net_income = safe(income, ["Net Income","Net Income Common Stockholders"])
    depreciation = safe(cashflow, ["Depreciation And Amortization","Depreciation & Amortization"])
    capex = abs(safe(cashflow, ["Capital Expenditure","Capital Expenditures"]))
    tax_provision = safe(income, ["Tax Provision","Income Tax Expense"])
    pretax_income = safe(income, ["Pretax Income","Income Before Tax"])
    interest_expense = abs(safe(income, ["Interest Expense","Interest Expense Non Operating"]))
    tax_rate = max(0, min(tax_provision/pretax_income, 0.40)) if pretax_income != 0 else 0.21

    fcf = safe(cashflow, ["Free Cash Flow"])
    if fcf == 0:
        op_cf = safe(cashflow, ["Operating Cash Flow","Total Cash From Operating Activities"])
        fcf = op_cf - capex

    def wc(col):
        return safe(balance,["Current Assets","Total Current Assets"],col) - safe(balance,["Current Liabilities","Total Current Liabilities"],col)

    nwc_change = wc(balance.columns[0]) - wc(balance.columns[1]) if balance is not None and not balance.empty and len(balance.columns)>=2 else 0.0

    price = info.get("regularMarketPrice") or info.get("currentPrice",0)
    shares = info.get("sharesOutstanding",0) or 0
    cash = info.get("totalCash",0) or 0
    debt = info.get("totalDebt",0) or 0
    market_cap = info.get("marketCap",0) or 0
    beta = info.get("beta"); pe = info.get("trailingPE"); fwd_pe = info.get("forwardPE")
    ev_ebitda = info.get("enterpriseToEbitda"); pb = info.get("priceToBook")
    dividend_yield = info.get("dividendYield",0) or 0
    sector = info.get("sector","N/A"); industry = info.get("industry","N/A")
    name = info.get("shortName", ticker.upper())
    week52_high = info.get("fiftyTwoWeekHigh",0) or 0; week52_low = info.get("fiftyTwoWeekLow",0) or 0
    ma50 = info.get("fiftyDayAverage",0) or 0; ma200 = info.get("twoHundredDayAverage",0) or 0
    rec_mean = info.get("recommendationMean"); n_analysts = info.get("numberOfAnalystOpinions",0) or 0
    target_mean = info.get("targetMeanPrice",0) or 0

    hist_revenue,hist_ebit,hist_ni,hist_fcf,hist_years = [],[],[],[],[]
    if income is not None and not income.empty:
        for i,col in enumerate(reversed(income.columns)):
            hist_years.append(str(col.year) if hasattr(col,'year') else f"Y{i+1}")
            hist_revenue.append(safe(income,["Total Revenue","Revenue"],col))
            hist_ebit.append(safe(income,["EBIT","Operating Income"],col))
            hist_ni.append(safe(income,["Net Income","Net Income Common Stockholders"],col))
            cf = safe(cashflow,["Free Cash Flow"],col) if cashflow is not None and not cashflow.empty else 0
            if cf==0 and cashflow is not None and not cashflow.empty:
                cf = safe(cashflow,["Operating Cash Flow","Total Cash From Operating Activities"],col) - abs(safe(cashflow,["Capital Expenditure","Capital Expenditures"],col))
            hist_fcf.append(cf)

    try:
        hist = stock.history(period="1y"); step = max(1,len(hist)//60)
        price_history = [{"date":str(hist.index[i].date()),"close":round(float(hist.iloc[i]["Close"]),2)} for i in range(0,len(hist),step)] if not hist.empty else []
    except: price_history = []

    rfr=0.043; erp=0.055
    coe = rfr+(beta if beta and beta>0 else 1.0)*erp
    cod = interest_expense/debt if debt>0 and interest_expense>0 else 0.05
    evm = market_cap if market_cap>0 else price*shares; tc=evm+debt
    we=evm/tc if tc>0 else 1.0; wd=debt/tc if tc>0 else 0.0
    wacc = max(0.05,min((we*coe)+(wd*cod*(1-tax_rate)),0.20))

    grs = [(hist_revenue[i]-hist_revenue[i-1])/hist_revenue[i-1] for i in range(1,len(hist_revenue)) if hist_revenue[i-1]>0 and hist_revenue[i]>0]
    avg_growth = max(-0.10,min(sum(grs)/len(grs),0.30)) if grs else 0.05

    da_pct=(depreciation/revenue) if revenue>0 else 0.03
    capex_pct=(capex/revenue) if revenue>0 else 0.04
    s_cap={"Technology":0.06,"Communication Services":0.07,"Consumer Cyclical":0.06,"Consumer Defensive":0.04,"Healthcare":0.05,"Financial Services":0.02,"Industrials":0.05,"Energy":0.08,"Basic Materials":0.06,"Real Estate":0.03,"Utilities":0.10}
    mature_capex = s_cap.get(sector,0.06)
    if capex_pct<=mature_capex: mature_capex=capex_pct
    nwc_pct = max(-0.20,min(nwc_change/(revenue*avg_growth),0.20)) if revenue>0 and avg_growth>0 else 0.02
    ebit_margin=(ebit/revenue) if revenue>0 else 0
    da_m=mature_capex if da_pct>=mature_capex else da_pct; tg=0.025

    dcf_implied=0
    if revenue>0 and shares>0 and wacc>tg:
        pfs=[]; prev=revenue
        for yr in range(1,11):
            r=revenue*((1+avg_growth)**yr); e=r*ebit_margin; nopat=e*(1-tax_rate)
            f=(yr-1)/9; da_yr=r*(da_pct+(da_m-da_pct)*f); cx_yr=r*(capex_pct+(mature_capex-capex_pct)*f)
            nwc_yr=(r-prev)*nwc_pct; ufcf=nopat+da_yr-cx_yr-nwc_yr
            pfs.append(ufcf/((1+wacc)**yr)); prev=r
        sum_pv=sum(pfs); final_u=revenue*((1+avg_growth)**10)*ebit_margin*(1-tax_rate)
        tv=(final_u*(1+tg))/(wacc-tg); pv_tv=tv/((1+wacc)**10)
        eq=sum_pv+pv_tv+cash-debt; dcf_implied=eq/shares

    return {
        "name":name,"ticker":ticker.upper(),"sector":sector,"industry":industry,
        "price":price,"shares":shares,"market_cap":market_cap,
        "cash":cash,"debt":debt,"beta":beta,"pe":pe,"fwd_pe":fwd_pe,
        "ev_ebitda":ev_ebitda,"pb":pb,"dividend_yield":dividend_yield,
        "revenue":revenue,"ebit":ebit,"net_income":net_income,"fcf":fcf,
        "ebit_margin":ebit_margin,"week52_high":week52_high,"week52_low":week52_low,
        "ma50":ma50,"ma200":ma200,"rec_mean":rec_mean,"n_analysts":n_analysts,
        "target_mean":target_mean,"hist_revenue":hist_revenue,"hist_ebit":hist_ebit,
        "hist_ni":hist_ni,"hist_fcf":hist_fcf,"hist_years":hist_years,
        "price_history":price_history,"dcf_implied":round(dcf_implied,2),
        "wacc":round(wacc,4),"avg_growth":round(avg_growth,4),
    }

# ── SIGNALS ───────────────────────────────────────────────────────────────
SECTOR_PE = {"Technology":28,"Healthcare":22,"Consumer Cyclical":20,"Consumer Defensive":22,
             "Communication Services":20,"Industrials":20,"Energy":14,"Financial Services":15,
             "Basic Materials":16,"Real Estate":35,"Utilities":18}

def compute_signals(d: dict) -> dict:
    signals = []
    if d["dcf_implied"]>0 and d["price"]>0:
        up=(d["dcf_implied"]-d["price"])/d["price"]*100
        if up>20: sig="buy"; det=f"Intrinsic value ~${d['dcf_implied']:.0f} vs ${d['price']:.0f} ({up:+.0f}% upside)"
        elif up<-20: sig="sell"; det=f"Intrinsic value ~${d['dcf_implied']:.0f} vs ${d['price']:.0f} ({up:+.0f}% downside)"
        else: sig="hold"; det=f"Intrinsic value ~${d['dcf_implied']:.0f} roughly in line with ${d['price']:.0f} ({up:+.0f}%)"
        signals.append({"name":"DCF Valuation","signal":sig,"detail":det})

    if d["n_analysts"]>=3 and d["rec_mean"] is not None:
        rm=d["rec_mean"]
        if rm<=2.0: sig="buy"; lbl="Strong Buy"
        elif rm<=2.8: sig="buy"; lbl="Moderate Buy"
        elif rm<=3.2: sig="hold"; lbl="Hold"
        elif rm<=4.0: sig="sell"; lbl="Moderate Sell"
        else: sig="sell"; lbl="Strong Sell"
        utt=(d["target_mean"]-d["price"])/d["price"]*100 if d["target_mean"] and d["price"] else 0
        det=f"{d['n_analysts']} analysts rate it {lbl}. Mean target ${d['target_mean']:.0f} ({utt:+.0f}%)"
        signals.append({"name":"Analyst Consensus","signal":sig,"detail":det})

    hs=0; hf=[]
    if d["fcf"]>0: hs+=1
    else: hf.append("negative free cash flow")
    if d["net_income"]>0: hs+=1
    else: hf.append("net loss")
    if d["debt"]>0 and d["market_cap"]>0 and (d["debt"]/d["market_cap"])<1.0: hs+=1
    elif d["debt"]>d["market_cap"]*1.5: hf.append("high debt load")
    if d["ebit_margin"]>0.10: hs+=1
    elif d["ebit_margin"]<0: hf.append("negative operating margin")
    if len(d["hist_revenue"])>=2 and d["hist_revenue"][-1]>0 and d["hist_revenue"][0]>d["hist_revenue"][-1]: hs+=1
    elif len(d["hist_revenue"])>=2 and d["hist_revenue"][-1]>0 and d["hist_revenue"][0]<d["hist_revenue"][-1]*0.9: hf.append("declining revenue")
    if hs>=4: sig="buy"; det="Strong financials — positive FCF, profitable, manageable debt"
    elif len(hf)>=2: sig="sell"; det=f"Caution: {', '.join(hf)}"
    elif len(hf)==1: sig="hold"; det=f"Mixed financials — note: {hf[0]}"
    else: sig="hold"; det="Adequate financials with no major red flags"
    signals.append({"name":"Financial Health","signal":sig,"detail":det})

    if d["price"]>0 and d["week52_high"]>0 and d["week52_low"]>0:
        wr=d["week52_high"]-d["week52_low"]; pos=(d["price"]-d["week52_low"])/wr if wr>0 else 0.5
        am200=d["price"]>d["ma200"] if d["ma200"]>0 else None
        am50=d["price"]>d["ma50"] if d["ma50"]>0 else None
        if pos>0.65 and am200 and am50: sig="buy"; det=f"Strong uptrend — top {pos*100:.0f}% of 52w range, above 50 & 200-day MAs"
        elif pos<0.35 and am200==False: sig="sell"; det=f"Downtrend — bottom {pos*100:.0f}% of 52w range, below 200-day MA"
        else: sig="hold"; det=f"Neutral momentum — {pos*100:.0f}% of 52w range, {'above' if am200 else 'below'} 200-day MA"
        signals.append({"name":"Price Momentum","signal":sig,"detail":det})

    avg_pe=SECTOR_PE.get(d["sector"],20); ms=0; mf=[]
    if d["pe"] and d["pe"]>0:
        if d["pe"]<avg_pe*0.8: ms+=1
        elif d["pe"]>avg_pe*1.5: mf.append(f"P/E {d['pe']:.1f}x above sector ~{avg_pe}x")
    if d["ev_ebitda"] and d["ev_ebitda"]>0:
        if d["ev_ebitda"]<15: ms+=1
        elif d["ev_ebitda"]>30: mf.append(f"EV/EBITDA {d['ev_ebitda']:.1f}x elevated")
    if d["pb"] and d["pb"]>0:
        if d["pb"]<3: ms+=1
        elif d["pb"]>10: mf.append(f"P/B {d['pb']:.1f}x elevated")
    if ms>=2: sig="buy"; det=f"Trading at a discount vs peers — P/E {d['pe']:.1f}x vs sector avg ~{avg_pe}x" if d["pe"] else "Attractive multiples vs peers"
    elif len(mf)>=2: sig="sell"; det="; ".join(mf)
    else:
        sig="hold"; parts=[x for x in [f"P/E {d['pe']:.1f}x" if d["pe"] else "",f"EV/EBITDA {d['ev_ebitda']:.1f}x" if d["ev_ebitda"] else ""] if x]
        det=f"In line with sector — {', '.join(parts)}" if parts else "Insufficient multiples data"
    signals.append({"name":"Relative Valuation","signal":sig,"detail":det})

    bc=sum(1 for s in signals if s["signal"]=="buy")
    sc=sum(1 for s in signals if s["signal"]=="sell")
    hc=sum(1 for s in signals if s["signal"]=="hold"); total=len(signals)
    if bc>sc and bc>=total*0.5: verdict="BUY"; conf=round(bc/total*100)
    elif sc>bc and sc>=total*0.5: verdict="SELL"; conf=round(sc/total*100)
    else: verdict="HOLD"; conf=round((hc+min(bc,sc))/total*100)
    return {"verdict":verdict,"confidence":conf,"buy_count":bc,"sell_count":sc,"hold_count":hc,"total_signals":total,"signals":signals}

# ── CLAUDE ────────────────────────────────────────────────────────────────
async def claude_call(prompt: str, max_tokens=600, use_search=False) -> str | None:
    if not CLAUDE_API_KEY: return None
    body = {"model":"claude-sonnet-4-20250514","max_tokens":max_tokens,"messages":[{"role":"user","content":prompt}]}
    if use_search: body["tools"]=[{"type":"web_search_20250305","name":"web_search"}]
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            messages=[{"role":"user","content":prompt}]
            for _ in range(8 if use_search else 1):
                resp=await client.post("https://api.anthropic.com/v1/messages",
                    headers={"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01"},
                    json={**body,"messages":messages})
                if resp.status_code!=200: return None
                result=resp.json(); stop=result.get("stop_reason",""); blocks=result.get("content",[])
                text="".join(b.get("text","") for b in blocks if b.get("type")=="text").strip()
                if stop=="end_turn": return text if text else None
                if stop=="tool_use":
                    messages.append({"role":"assistant","content":blocks})
                    trs=[]
                    for b in blocks:
                        if b.get("type")=="server_tool_use": trs.append({"type":"server_tool_result","tool_use_id":b["id"]})
                        elif b.get("type")=="tool_use": trs.append({"type":"tool_result","tool_use_id":b["id"],"content":"OK"})
                    if trs: messages.append({"role":"user","content":trs})
                    continue
                if text: return text
        return None
    except: return None

async def get_summary(d,signals):
    v=signals["verdict"]; b=signals["buy_count"]; s=signals["sell_count"]; t=signals["total_signals"]
    sl="\n".join(f"- {x['name']}: {x['signal'].upper()} — {x['detail']}" for x in signals["signals"])
    prompt=f"""Explain this stock analysis to a retail investor with no finance background. Plain English only.
Company: {d['name']} ({d['ticker']}) — {d['sector']}
Verdict: {v} ({b}/{t} bullish, {s}/{t} bearish)
Signals:\n{sl}
Write 2-3 sentences. Lead with the verdict. Mention strongest reason and one key risk. End with: "This is not financial advice — always do your own research." """
    r=await claude_call(prompt,250)
    if not r:
        msgs={"BUY":f"{d['name']} looks attractively priced with most signals pointing to upside.","SELL":f"{d['name']} is showing multiple warning signs worth paying attention to.","HOLD":f"{d['name']} looks fairly valued right now — not an obvious buy or sell."}
        return msgs.get(v,"")+f" This is not financial advice — always do your own research."
    return r

async def get_catalyst(ticker,name,sector):
    # Check cache first (1 hour TTL)
    with get_db() as conn:
        row=conn.execute("SELECT data,cached_at FROM catalyst_cache WHERE ticker=?",(ticker,)).fetchone()
        if row:
            cached_at=datetime.fromisoformat(row["cached_at"])
            if datetime.now(timezone.utc)-cached_at<timedelta(hours=1):
                return json.loads(row["data"])

    prompt=f"""Search the web for upcoming catalysts for {name} ({ticker}) stock. Find:
1. Next earnings date and EPS/revenue estimates
2. Any analyst upgrades/downgrades in the last 2 weeks
3. Upcoming product launches, FDA decisions, regulatory events, or major conferences
4. Any recent insider buying or institutional activity
5. One key risk to watch

Return ONLY this JSON (no markdown):
{{"earnings_date":"<date or 'Unknown'>","earnings_est_eps":"<$ or null>","earnings_est_rev":"<$ or null>","analyst_activity":"<1 sentence>","upcoming_events":["<event 1>","<event 2>"],"key_risk":"<1 sentence>","last_updated":"<today's date>"}}"""

    text=await claude_call(prompt,600,use_search=True)
    default={"earnings_date":"Unknown","earnings_est_eps":None,"earnings_est_rev":None,"analyst_activity":"No recent analyst activity found.","upcoming_events":[],"key_risk":"Insufficient data to assess near-term risks.","last_updated":datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    if not text: return default
    try:
        s=text.find("{"); e=text.rfind("}")+1
        data=json.loads(text[s:e]) if s>=0 and e>s else default
        with get_db() as conn:
            conn.execute("INSERT OR REPLACE INTO catalyst_cache VALUES (?,?,?)",(ticker,json.dumps(data),datetime.now(timezone.utc).isoformat()))
        return data
    except: return default

def fmt_b(n):
    if not n or n==0: return "N/A"
    if abs(n)>=1e12: return f"${n/1e12:.2f}T"
    if abs(n)>=1e9: return f"${n/1e9:.2f}B"
    if abs(n)>=1e6: return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"

def build_response(d,signals,summary):
    return {
        "company":{"name":d["name"],"ticker":d["ticker"],"sector":d["sector"],"industry":d["industry"],
            "price":d["price"],"market_cap":fmt_b(d["market_cap"]),"pe":round(d["pe"],1) if d["pe"] else None,
            "fwd_pe":round(d["fwd_pe"],1) if d["fwd_pe"] else None,"ev_ebitda":round(d["ev_ebitda"],1) if d["ev_ebitda"] else None,
            "beta":round(d["beta"],2) if d["beta"] else None,"dividend_yield":round(d["dividend_yield"]*100,2) if d["dividend_yield"] else None,
            "week52_high":d["week52_high"],"week52_low":d["week52_low"],
            "ma50":round(d["ma50"],2) if d["ma50"] else None,"ma200":round(d["ma200"],2) if d["ma200"] else None,
            "target_mean":round(d["target_mean"],2) if d["target_mean"] else None,"n_analysts":d["n_analysts"],
            "sector_pe":SECTOR_PE.get(d["sector"],20)},
        "chart_data":{"price_history":d["price_history"],"hist_years":d["hist_years"],
            "hist_revenue":d["hist_revenue"],"hist_fcf":d["hist_fcf"],"hist_ni":d["hist_ni"]},
        "analysis":{"verdict":signals["verdict"],"confidence":signals["confidence"],
            "buy_count":signals["buy_count"],"sell_count":signals["sell_count"],"hold_count":signals["hold_count"],
            "total_signals":signals["total_signals"],"signals":signals["signals"],"summary":summary},
        "timestamp":datetime.now(timezone.utc).isoformat()
    }

# ── ROUTES ────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(req: TickerRequest):
    try:
        ticker=req.ticker.strip().upper()
        if not ticker: raise HTTPException(400,"Ticker required.")
        d=fetch_financials(ticker)
        signals=compute_signals(d)
        summary=await get_summary(d,signals)
        resp=build_response(d,signals,summary)
        # Store signal history
        with get_db() as conn:
            conn.execute("INSERT INTO signal_history VALUES (?,?,?,?,?,?,?)",
                (ticker,signals["verdict"],signals["confidence"],signals["buy_count"],
                 signals["sell_count"],signals["hold_count"],datetime.now(timezone.utc).isoformat()))
        return resp
    except HTTPException: raise
    except Exception as e:
        print(traceback.format_exc()); return JSONResponse(500,{"detail":str(e)})

@app.post("/api/catalyst")
async def catalyst_endpoint(req: CatalystRequest):
    try:
        ticker=req.ticker.strip().upper()
        d=fetch_financials(ticker)
        data=await get_catalyst(ticker,d["name"],d["sector"])
        return {"ticker":ticker,"name":d["name"],"catalyst":data,"timestamp":datetime.now(timezone.utc).isoformat()}
    except HTTPException: raise
    except Exception as e:
        print(traceback.format_exc()); return JSONResponse(500,{"detail":str(e)})

@app.post("/api/history")
async def signal_history(req: TickerRequest):
    try:
        ticker=req.ticker.strip().upper()
        with get_db() as conn:
            rows=conn.execute("SELECT verdict,confidence,buy_count,sell_count,hold_count,recorded_at FROM signal_history WHERE ticker=? ORDER BY recorded_at DESC LIMIT 30",(ticker,)).fetchall()
        return {"ticker":ticker,"history":[dict(r) for r in rows]}
    except Exception as e:
        return JSONResponse(500,{"detail":str(e)})

@app.post("/api/watchlist/add")
async def watchlist_add(req: WatchlistRequest):
    try:
        with get_db() as conn:
            conn.execute("INSERT OR IGNORE INTO watchlist VALUES (?,?,?)",
                (req.session_id,req.ticker.upper(),datetime.now(timezone.utc).isoformat()))
        return {"ok":True}
    except Exception as e: return JSONResponse(500,{"detail":str(e)})

@app.post("/api/watchlist/remove")
async def watchlist_remove(req: WatchlistRequest):
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM watchlist WHERE session_id=? AND ticker=?",
                (req.session_id,req.ticker.upper()))
        return {"ok":True}
    except Exception as e: return JSONResponse(500,{"detail":str(e)})

@app.post("/api/watchlist/get")
async def watchlist_get(req: WatchlistGetRequest):
    try:
        with get_db() as conn:
            rows=conn.execute("SELECT ticker,added_at FROM watchlist WHERE session_id=? ORDER BY added_at DESC",
                (req.session_id,)).fetchall()
        tickers=[r["ticker"] for r in rows]
        return {"tickers":tickers}
    except Exception as e: return JSONResponse(500,{"detail":str(e)})

@app.post("/api/watchlist/analyze")
async def watchlist_analyze(req: WatchlistGetRequest):
    """Fetch quick signal summary for all watchlist tickers"""
    try:
        with get_db() as conn:
            rows=conn.execute("SELECT ticker FROM watchlist WHERE session_id=? ORDER BY added_at DESC",
                (req.session_id,)).fetchall()
        tickers=[r["ticker"] for r in rows]
        if not tickers: return {"items":[]}

        async def analyze_one(ticker):
            try:
                d=fetch_financials(ticker)
                signals=compute_signals(d)
                # Get latest history entry for change detection
                with get_db() as conn:
                    prev=conn.execute("SELECT verdict FROM signal_history WHERE ticker=? ORDER BY recorded_at DESC LIMIT 2",(ticker,)).fetchall()
                prev_verdict=prev[1]["verdict"] if len(prev)>=2 else None
                changed=prev_verdict and prev_verdict!=signals["verdict"]
                return {"ticker":ticker,"name":d["name"],"price":d["price"],
                    "verdict":signals["verdict"],"confidence":signals["confidence"],
                    "buy_count":signals["buy_count"],"sell_count":signals["sell_count"],
                    "hold_count":signals["hold_count"],"changed":changed,"prev_verdict":prev_verdict,
                    "sector":d["sector"],"market_cap":fmt_b(d["market_cap"]),"error":None}
            except Exception as ex:
                return {"ticker":ticker,"error":str(ex)}

        results=await asyncio.gather(*[analyze_one(t) for t in tickers])
        return {"items":list(results)}
    except Exception as e:
        print(traceback.format_exc()); return JSONResponse(500,{"detail":str(e)})

@app.post("/api/compare")
async def compare(req: CompareRequest):
    try:
        t1=req.ticker1.strip().upper(); t2=req.ticker2.strip().upper()
        async def get_one(t):
            d=fetch_financials(t); s=compute_signals(d)
            return d,s
        (d1,s1),(d2,s2)=await asyncio.gather(get_one(t1),get_one(t2))
        # Ask Claude who wins
        prompt=f"""Compare these two stocks for a retail investor. Be direct and decisive — pick a clear winner.
{t1} ({d1['name']}): {s1['verdict']} — {s1['buy_count']}/{s1['total_signals']} bullish, confidence {s1['confidence']}%, P/E {d1['pe']}, sector {d1['sector']}
{t2} ({d2['name']}): {s2['verdict']} — {s2['buy_count']}/{s2['total_signals']} bullish, confidence {s2['confidence']}%, P/E {d2['pe']}, sector {d2['sector']}
Write 3 sentences max. State the winner first. Give the most important reason. Mention the main risk of the winner. No jargon. End with "Not financial advice." """
        verdict=await claude_call(prompt,200)
        if not verdict:
            w=t1 if s1["confidence"]>s2["confidence"] else t2
            verdict=f"{w} comes out ahead based on a stronger signal score. Not financial advice."
        return {
            "ticker1":{"ticker":t1,"data":build_response(d1,s1,"")},
            "ticker2":{"ticker":t2,"data":build_response(d2,s2,"")},
            "comparison_verdict":verdict,
            "winner":t1 if (s1["verdict"]=="BUY" and s2["verdict"]!="BUY") or s1["confidence"]>s2["confidence"] else t2
        }
    except HTTPException: raise
    except Exception as e:
        print(traceback.format_exc()); return JSONResponse(500,{"detail":str(e)})

# Sector heatmap tickers
# ── HEATMAP UNIVERSE ──────────────────────────────────────────────────────
# US: S&P 500 large + mid caps (~500), Canada: TSX 60 (.TO suffix)
# Organised by GICS sector. ~560 tickers total.

HEATMAP_TICKERS = {
    "🇺🇸 Technology": [
        # Mega-cap / large-cap
        "AAPL","MSFT","NVDA","GOOGL","GOOG","META","AMD","INTC","CRM","ORCL",
        "ADBE","QCOM","TXN","AVGO","MU","NOW","SNOW","PLTR","AMAT","LRCX",
        "KLAC","MRVL","MCHP","SWKS","QRVO","MPWR","ON","STX","WDC","NTAP",
        # Mid-cap software / cloud
        "DDOG","ZS","CRWD","NET","OKTA","TEAM","MDB","HCP","GTLB","PATH",
        "BILL","HUBS","PCTY","PAYC","COUP","APPN","ESTC","SUMO","FROG","AI",
        # Semis & hardware
        "TSM","ASML","NXPI","TER","COHR","RMBS","MTSI","ACLS","UCTT","ONTO",
    ],
    "🇺🇸 Healthcare": [
        "JNJ","UNH","PFE","ABBV","MRK","LLY","TMO","ABT","BMY","AMGN",
        "GILD","ISRG","VRTX","REGN","ZTS","DHR","IQV","HCA","CVS","CI",
        "HUM","MCK","ABC","CAH","ANTM","MOH","CNC","WBA","DXCM","ALGN",
        "HOLX","IDXX","MTD","WAT","A","BIO","TECH","PODD","INSP","NVCR",
        "MRNA","BNTX","RARE","ALNY","IONS","SRPT","BMRN","BLUE","ACAD","EXAS",
    ],
    "🇺🇸 Financial Services": [
        "JPM","BAC","WFC","GS","MS","BLK","AXP","C","USB","TFC",
        "SCHW","COF","PGR","CB","ICE","CME","SPGI","MCO","AON","MMC",
        "MET","PRU","AFL","ALL","TRV","HIG","EG","CINF","GL","RNR",
        "FITB","HBAN","KEY","RF","MTB","CFG","PBCT","FHN","SNV","IBOC",
        "V","MA","PYPL","SQ","FIS","FI","GPN","WEX","FLYW","PAYO",
    ],
    "🇺🇸 Consumer Cyclical": [
        "AMZN","TSLA","HD","NKE","MCD","SBUX","TGT","LOW","TJX","BKNG",
        "ABNB","UBER","LYFT","F","GM","RIVN","ETSY","EBAY","MAR","HLT",
        "RCL","CCL","NCLH","LVS","MGM","WYNN","CZR","PENN","DKNG","RSI",
        "ORLY","AZO","AAP","GPC","LKQ","BWA","LEA","ALV","VC","MGA",
        "ROST","BURL","FIVE","OLLI","DG","DLTR","KSS","M","JWN","ANF",
    ],
    "🇺🇸 Industrials": [
        "CAT","HON","UPS","BA","GE","MMM","RTX","DE","EMR","ITW",
        "LMT","NOC","GD","FDX","CSX","NSC","WM","RSG","VRSK","FAST",
        "PCAR","CMI","PH","ROK","AME","XYL","ROP","FTV","GNRC","BLDR",
        "URI","TREX","MAS","OC","SWK","SNA","GWW","MSC","WIRE","ATKR",
        "DAL","UAL","AAL","LUV","ALK","JBLU","CHRW","EXPD","XPO","SAIA",
    ],
    "🇺🇸 Energy": [
        "XOM","CVX","COP","SLB","EOG","MPC","VLO","OXY","PSX","PXD",
        "HAL","BKR","WMB","KMI","OKE","DVN","FANG","HES","MRO","APA",
        "CTRA","SM","MTDR","CRGY","NOG","VTLE","CHRD","ESTE","CIVI","RRC",
        "EQT","AR","CNX","GPOR","SWN","CRK","TELL","MNRL","PHX","DINO",
    ],
    "🇺🇸 Communication Services": [
        "NFLX","DIS","T","VZ","CMCSA","CHTR","SNAP","PINS","SPOT","WBD",
        "MTCH","EA","TTWO","ROKU","PARA","FOX","NYT","IAC","ZM","GOOGL",
        "LUMN","TMUS","DISH","SIRI","IACI","FOXA","LYV","MSG","MSGS","WWE",
        "RBLX","U","PLTK","SKLZ","GMBL","PENN","RSI","DKNG","GENI","SGHC",
    ],
    "🇺🇸 Consumer Defensive": [
        "WMT","PG","KO","PEP","COST","CL","GIS","K","CPB","HRL",
        "MO","PM","STZ","TAP","MNST","KHC","SYY","CAG","TSN","HSY",
        "BG","ADM","INGR","MKC","SJM","THS","SMPL","CENT","FLNG","NOMD",
        "EL","COTY","REV","SPB","CHD","ENR","CLX","KMB","PPC","SAFM",
    ],
    "🇺🇸 Real Estate": [
        "PLD","AMT","EQIX","CCI","PSA","WELL","AVB","EQR","DLR","SPG",
        "O","VICI","WY","EXR","IRM","MAA","UDR","ESS","CPT","KIM",
        "VTR","PEAK","OHI","HR","NNN","SRC","STOR","ADC","EPRT","NTST",
        "BXP","VNO","SLG","KRC","HIW","CUZ","PDM","JBGS","DEI","WRE",
    ],
    "🇺🇸 Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","ETR",
        "WEC","ES","AWK","PPL","FE","DTE","CMS","NI","AES","CNP",
        "PNW","EVRG","OGE","AVA","POR","IDA","NWE","SPWR","RUN","NOVA",
        "BEP","BEPC","CWEN","AY","NEP","HASI","AMPX","ARRY","FSLR","ENPH",
    ],
    "🇺🇸 Basic Materials": [
        "LIN","APD","ECL","SHW","FCX","NEM","NUE","VMC","MLM","CF",
        "MOS","ALB","CTVA","FMC","CE","PPG","EMN","HUN","OLN","AA",
        "MP","LTHM","SQM","LAC","PLL","LTBR","CCJ","NXE","URG","DNN",
        "X","CLF","STLD","CMC","RS","WOR","ZEUS","CSTM","ATI","CRS",
    ],
    # ── CANADA (TSX 60) ──────────────────────────────────────────────────
    "🇨🇦 Financials": [
        "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO",
        "SLF.TO","MFC.TO","IAG.TO","GWO.TO","FFH.TO","IFC.TO",
    ],
    "🇨🇦 Energy": [
        "CNQ.TO","SU.TO","CVE.TO","IMO.TO","MEG.TO","ARX.TO",
        "BTE.TO","CPG.TO","ERF.TO","TVE.TO","TOU.TO","PEY.TO",
    ],
    "🇨🇦 Materials & Mining": [
        "ABX.TO","AEM.TO","AGI.TO","K.TO","IMG.TO","FM.TO",
        "LUN.TO","CS.TO","TCS.TO","HBM.TO","ERO.TO","WPM.TO",
    ],
    "🇨🇦 Technology & Telecom": [
        "SHOP.TO","CSU.TO","OTEX.TO","BB.TO","KXS.TO","ENGH.TO",
        "BCE.TO","T.TO","RCI-B.TO","QBR-B.TO","MBT.TO","TCI.TO",
    ],
    "🇨🇦 Industrials & REITs": [
        "CNR.TO","CP.TO","TFI.TO","STN.TO","WSP.TO","TFII.TO",
        "REI-UN.TO","CAR-UN.TO","DIR-UN.TO","HR-UN.TO","AP-UN.TO","GRT-UN.TO",
    ],
    "🇨🇦 Consumer & Healthcare": [
        "ATD.TO","L.TO","MRU.TO","DOL.TO","EMP-A.TO","PBH.TO",
        "CTC-A.TO","GIL.TO","NWC.TO","WN.TO","BPF-UN.TO","QSR.TO",
    ],
}

# ── NIGHTLY CACHE LOGIC ───────────────────────────────────────────────────
HEATMAP_CACHE_TTL_HOURS = 6   # serve cached results for up to 6 hours
HEATMAP_CACHE_KEY = "heatmap_full"

def get_heatmap_cache():
    """Return cached heatmap data if fresh, else None."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT data, cached_at FROM analysis_cache WHERE ticker=?",
                (HEATMAP_CACHE_KEY,)
            ).fetchone()
        if row:
            cached_at = datetime.fromisoformat(row["cached_at"])
            age = datetime.now(timezone.utc) - cached_at
            if age < timedelta(hours=HEATMAP_CACHE_TTL_HOURS):
                data = json.loads(row["data"])
                data["cached"] = True
                data["cache_age_minutes"] = int(age.total_seconds() / 60)
                return data
    except: pass
    return None

def set_heatmap_cache(data: dict):
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_cache VALUES (?,?,?)",
                (HEATMAP_CACHE_KEY, json.dumps(data), datetime.now(timezone.utc).isoformat())
            )
    except: pass

async def run_heatmap_scan(market_filter: str = "all") -> dict:
    """
    Scan all tickers. market_filter: 'us' | 'ca' | 'all'
    Batches requests in chunks of 30 to avoid hammering yfinance.
    """
    BATCH = 30

    # Filter sectors by market
    if market_filter == "us":
        sectors_to_scan = {k: v for k, v in HEATMAP_TICKERS.items() if "🇺🇸" in k}
    elif market_filter == "ca":
        sectors_to_scan = {k: v for k, v in HEATMAP_TICKERS.items() if "🇨🇦" in k}
    else:
        sectors_to_scan = HEATMAP_TICKERS

    all_tickers = [t for ts in sectors_to_scan.values() for t in ts]

    async def scan_one(ticker):
        try:
            d = fetch_financials(ticker)
            s = compute_signals(d)
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO signal_history VALUES (?,?,?,?,?,?,?)",
                    (ticker, s["verdict"], s["confidence"], s["buy_count"],
                     s["sell_count"], s["hold_count"], datetime.now(timezone.utc).isoformat())
                )
            return {
                "ticker": ticker, "name": d["name"], "price": d["price"],
                "verdict": s["verdict"], "confidence": s["confidence"],
                "sector": d["sector"], "market_cap": fmt_b(d["market_cap"]),
                "error": None
            }
        except Exception as ex:
            return {"ticker": ticker, "verdict": "HOLD", "confidence": 50, "name": ticker, "price": None, "error": str(ex)[:60]}

    # Process in batches to be gentle on rate limits
    results = []
    for i in range(0, len(all_tickers), BATCH):
        batch = all_tickers[i:i+BATCH]
        batch_results = await asyncio.gather(*[scan_one(t) for t in batch])
        results.extend(batch_results)
        if i + BATCH < len(all_tickers):
            await asyncio.sleep(1)  # brief pause between batches

    result_map = {r["ticker"]: r for r in results}

    sectors_out = {}
    for sector, tickers in sectors_to_scan.items():
        items = [result_map[t] for t in tickers if t in result_map]
        valid = [i for i in items if not i.get("error")]
        bc = sum(1 for i in valid if i["verdict"] == "BUY")
        sc = sum(1 for i in valid if i["verdict"] == "SELL")
        hc = sum(1 for i in valid if i["verdict"] == "HOLD")
        total = len(valid)
        if total == 0:
            sector_verdict = "MIXED"
        elif bc > sc and bc >= total * 0.4:
            sector_verdict = "BULLISH"
        elif sc > bc and sc >= total * 0.4:
            sector_verdict = "BEARISH"
        else:
            sector_verdict = "MIXED"
        sectors_out[sector] = {
            "verdict": sector_verdict, "buy": bc, "sell": sc, "hold": hc,
            "total": total, "scanned": len(items), "tickers": valid
        }

    total_tickers = sum(s["scanned"] for s in sectors_out.values())
    total_buy = sum(s["buy"] for s in sectors_out.values())
    total_sell = sum(s["sell"] for s in sectors_out.values())
    total_hold = sum(s["hold"] for s in sectors_out.values())

    return {
        "sectors": sectors_out,
        "summary": {"total": total_tickers, "buy": total_buy, "sell": total_sell, "hold": total_hold},
        "market_filter": market_filter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "cache_age_minutes": 0,
    }

# Background nightly refresh task
async def nightly_refresh_loop():
    """Refresh the heatmap cache every 6 hours in the background."""
    await asyncio.sleep(30)  # wait for server to fully start
    while True:
        try:
            print("[Stockwise] Running background heatmap refresh...")
            data = await run_heatmap_scan("all")
            set_heatmap_cache(data)
            print(f"[Stockwise] Heatmap cache updated: {data['summary']['total']} tickers scanned.")
        except Exception as e:
            print(f"[Stockwise] Background refresh error: {e}")
        await asyncio.sleep(HEATMAP_CACHE_TTL_HOURS * 3600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(nightly_refresh_loop())

@app.get("/api/heatmap")
async def heatmap(market: str = "all", force: bool = False):
    """
    market: 'us' | 'ca' | 'all'
    force=true: bypass cache and rescan live
    """
    try:
        # Serve from cache if available and not forcing
        if not force:
            cached = get_heatmap_cache()
            if cached:
                # If market filter differs, recompute sector subset from cache
                if market != "all" and market != cached.get("market_filter", "all"):
                    flag = "🇺🇸" if market == "us" else "🇨🇦"
                    filtered = {k: v for k, v in cached["sectors"].items() if flag in k}
                    cached["sectors"] = filtered
                return cached

        # No cache or forced — run live scan
        data = await run_heatmap_scan(market)
        if market == "all":
            set_heatmap_cache(data)
        return data
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(500, {"detail": str(e)})

@app.get("/api/heatmap/status")
def heatmap_status():
    """Returns cache age so the frontend can show when data was last refreshed."""
    cached = get_heatmap_cache()
    if cached:
        return {"cached": True, "age_minutes": cached.get("cache_age_minutes", 0),
                "timestamp": cached.get("timestamp"), "total_tickers": cached.get("summary", {}).get("total", 0)}
    return {"cached": False, "age_minutes": None, "timestamp": None, "total_tickers": 0}

@app.get("/api/health")
def health(): return {"status":"ok"}

app.mount("/static",StaticFiles(directory="static"),name="static")

@app.get("/{full_path:path}")
def serve(full_path:str=""): return FileResponse("static/index.html")
