from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import math, traceback, httpx, json, os, io
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from fastapi.responses import StreamingResponse

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

class MemoRequest(BaseModel):
    ticker: str

class PDFRequest(BaseModel):
    ticker: str
    wacc: float
    terminal_growth: float
    revenue_growth: float
    projection_years: int = 10

# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------
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
        income = stock.financials; cashflow = stock.cashflow; balance = stock.balance_sheet
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch statements for '{ticker}': {str(e)}")

    for name, df in [("income", income), ("cashflow", cashflow), ("balance", balance)]:
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No {name} statement for '{ticker}'.")

    def safe(df, keys, col=None):
        if col is None: col = df.columns[0]
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
    tax_rate = max(0, min(tax_provision / pretax_income, 0.40)) if pretax_income != 0 else 0.21

    def wc(col):
        return safe(balance, ["Current Assets", "Total Current Assets"], col) - safe(balance, ["Current Liabilities", "Total Current Liabilities"], col)

    nwc_change_abs = wc(balance.columns[0]) - wc(balance.columns[1]) if len(balance.columns) >= 2 else 0.0
    rev_prior = safe(income, ["Total Revenue", "Revenue"], income.columns[1]) if len(income.columns) >= 2 else 0.0
    rev_delta = revenue - rev_prior if rev_prior > 0 else revenue
    nwc_pct = max(-0.20, min(nwc_change_abs / rev_delta, 0.20)) if rev_delta != 0 and abs(rev_delta) > 1e6 else 0.02
    da_pct = (depreciation / revenue) if revenue > 0 else 0.03
    capex_pct = (capex / revenue) if revenue > 0 else 0.04

    sector = info.get("sector", "N/A"); industry = info.get("industry", "N/A")
    mature_capex = {"Technology":0.06,"Communication Services":0.07,"Consumer Cyclical":0.06,"Consumer Defensive":0.04,"Healthcare":0.05,"Financial Services":0.02,"Industrials":0.05,"Energy":0.08,"Basic Materials":0.06,"Real Estate":0.03,"Utilities":0.10}
    mature_capex_pct = mature_capex.get(sector, 0.06)
    if capex_pct <= mature_capex_pct: mature_capex_pct = capex_pct

    price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
    shares = info.get("sharesOutstanding", 0) or 0
    cash = info.get("totalCash", 0) or 0; debt = info.get("totalDebt", 0) or 0
    name = info.get("shortName", ticker.upper()); market_cap = info.get("marketCap", 0) or 0
    pe = info.get("trailingPE"); ev_ebitda = info.get("enterpriseToEbitda"); beta = info.get("beta")

    hist_revenue, hist_ebit, hist_ni, hist_fcf, hist_years = [], [], [], [], []
    for i, col in enumerate(reversed(income.columns)):
        hist_years.append(str(col.year) if hasattr(col, 'year') else f"Y{i+1}")
        hist_revenue.append(safe(income, ["Total Revenue", "Revenue"], col))
        hist_ebit.append(safe(income, ["EBIT", "Operating Income"], col))
        hist_ni.append(safe(income, ["Net Income", "Net Income Common Stockholders"], col))
        cf = safe(cashflow, ["Free Cash Flow"], col)
        if cf == 0: cf = safe(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"], col) - abs(safe(cashflow, ["Capital Expenditure", "Capital Expenditures"], col))
        hist_fcf.append(cf)

    hist_margins = [{"year": hist_years[i], "ebit": round(hist_ebit[i]/r*100, 1), "net": round(hist_ni[i]/r*100, 1)} for i, r in enumerate(hist_revenue) if r > 0]

    try:
        hist = stock.history(period="1y")
        step = max(1, len(hist) // 60)
        price_history = [{"date": str(hist.index[idx].date()), "close": round(float(hist.iloc[idx]["Close"]), 2)} for idx in range(0, len(hist), step)] if not hist.empty else []
    except: price_history = []

    if shares == 0: raise HTTPException(status_code=400, detail=f"No shares data for '{ticker}'.")

    rfr = 0.043; erp = 0.055
    coe = rfr + (beta if beta and beta > 0 else 1.0) * erp
    cod = interest_expense / debt if debt > 0 and interest_expense > 0 else 0.05
    evm = market_cap if market_cap > 0 else price * shares
    tc = evm + debt; we = evm / tc if tc > 0 else 1.0; wd = debt / tc if tc > 0 else 0.0
    calc_wacc = max(0.05, min((we * coe) + (wd * cod * (1 - tax_rate)), 0.20))

    grs = [(hist_revenue[i] - hist_revenue[i-1]) / hist_revenue[i-1] for i in range(1, len(hist_revenue)) if hist_revenue[i-1] > 0 and hist_revenue[i] > 0]
    avg_gr = max(-0.10, min(sum(grs)/len(grs), 0.30)) if grs else 0.05

    return {
        "name":name,"ticker":ticker.upper(),"sector":sector,"industry":industry,
        "price":price,"shares_outstanding":shares,"market_cap":market_cap,
        "cash":cash,"debt":debt,"pe_ratio":pe,"ev_ebitda":ev_ebitda,"beta":beta,
        "latest_revenue":revenue,"latest_ebit":ebit,"net_income":net_income,
        "depreciation":depreciation,"capex":capex,
        "da_pct":da_pct,"capex_pct":capex_pct,"mature_capex_pct":mature_capex_pct,
        "tax_rate":tax_rate,"interest_expense":interest_expense,"nwc_pct_of_rev_change":nwc_pct,
        "hist_revenue":hist_revenue,"hist_ebit":hist_ebit,"hist_net_income":hist_ni,
        "hist_fcf":hist_fcf,"hist_years":hist_years,"hist_margins":hist_margins,"price_history":price_history,
        "ebit_margin":(ebit/revenue*100) if revenue else 0,
        "calc_wacc":round(calc_wacc,4),"calc_revenue_growth":round(avg_gr,4),"calc_terminal_growth":0.025,
        "cost_of_equity":round(coe,4),"cost_of_debt":round(cod,4),
        "weight_equity":round(we,4),"weight_debt":round(wd,4),"risk_free_rate":rfr,"equity_risk_premium":erp,
    }

# ---------------------------------------------------------------------------
# Claude multi-turn helper
# ---------------------------------------------------------------------------
async def claude_with_search(prompt: str, max_tokens: int = 1000, max_turns: int = 8) -> str | None:
    if not CLAUDE_API_KEY: return None
    messages = [{"role": "user", "content": prompt}]
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for turn in range(max_turns):
                resp = await client.post("https://api.anthropic.com/v1/messages",
                    headers={"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01"},
                    json={"model":"claude-sonnet-4-20250514","max_tokens":max_tokens,
                          "tools":[{"type":"web_search_20250305","name":"web_search"}],
                          "messages":messages})
                if resp.status_code != 200:
                    print(f"Claude error turn {turn}: {resp.status_code} {resp.text}")
                    return None
                result = resp.json()
                stop = result.get("stop_reason","")
                content_blocks = result.get("content", [])
                print(f"Turn {turn}: stop={stop}, blocks={[b.get('type') for b in content_blocks]}")

                # Collect any text from this response
                text = "".join(b.get("text","") for b in content_blocks if b.get("type")=="text").strip()

                if stop == "end_turn":
                    return text if text else None

                # Handle tool use - need to continue conversation
                if stop == "tool_use":
                    messages.append({"role":"assistant","content":content_blocks})
                    tool_results = []
                    for b in content_blocks:
                        if b.get("type") == "server_tool_use":
                            tool_results.append({"type":"server_tool_result","tool_use_id":b["id"]})
                        elif b.get("type") == "tool_use":
                            tool_results.append({"type":"tool_result","tool_use_id":b["id"],"content":"OK"})
                    if tool_results:
                        messages.append({"role":"user","content":tool_results})
                    continue

                # Unknown stop reason but has text
                if text: return text
        return None
    except Exception as e:
        print(f"Claude error: {traceback.format_exc()}"); return None

async def claude_no_search(prompt: str, max_tokens: int = 1000) -> str | None:
    """Fallback: call Claude without web search."""
    if not CLAUDE_API_KEY: return None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages",
                headers={"Content-Type":"application/json","x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01"},
                json={"model":"claude-sonnet-4-20250514","max_tokens":max_tokens,
                      "messages":[{"role":"user","content":prompt}]})
            if resp.status_code != 200:
                print(f"Claude no-search error: {resp.status_code}"); return None
            result = resp.json()
            return "".join(b.get("text","") for b in result.get("content",[]) if b.get("type")=="text").strip()
    except Exception as e:
        print(f"Claude no-search error: {traceback.format_exc()}"); return None

def extract_json_obj(text):
    s = text.find("{"); e = text.rfind("}") + 1
    return json.loads(text[s:e]) if s >= 0 and e > s else None

def extract_json_arr(text):
    s = text.find("["); e = text.rfind("]") + 1
    return json.loads(text[s:e]) if s >= 0 and e > s else None

# ---------------------------------------------------------------------------
# AI: assumptions, news, memo
# ---------------------------------------------------------------------------
async def get_ai_assumptions(data: dict) -> dict | None:
    hr_str = "".join(f"  Yr{i+1}: ${r/1e9:.1f}B\n" for i,r in enumerate(data["hist_revenue"]))
    gr_str = "".join(f"  {i}->{i+1}: {(data['hist_revenue'][i]-data['hist_revenue'][i-1])/data['hist_revenue'][i-1]*100:.1f}%\n" for i in range(1,len(data["hist_revenue"])) if data["hist_revenue"][i-1]>0)
    prompt = f"""You are a senior equity research analyst. First, search the web for the latest news, earnings, analyst estimates, and recent developments for {data['name']} ({data['ticker']}). Then recommend DCF assumptions using BOTH financials AND current news.

DATA: Sector:{data['sector']}, Industry:{data['industry']}, MCap:${data['market_cap']/1e9:.1f}B, Price:${data['price']:.2f}, Beta:{data['beta'] or 'N/A'}, P/E:{f"{data['pe_ratio']:.1f}x" if data['pe_ratio'] else 'N/A'}, EV/EBITDA:{f"{data['ev_ebitda']:.1f}x" if data['ev_ebitda'] else 'N/A'}
Revenue:${data['latest_revenue']/1e9:.1f}B, EBIT Margin:{data['ebit_margin']:.1f}%, CapEx:{data['capex_pct']*100:.1f}% rev
Revenue History:\n{hr_str}Growth Rates:\n{gr_str}Cash:${data['cash']/1e9:.1f}B, Debt:${data['debt']/1e9:.1f}B, Tax:{data['tax_rate']*100:.1f}%
CoE:{data['cost_of_equity']*100:.1f}%, CoD:{data['cost_of_debt']*100:.1f}%, EqWt:{data['weight_equity']*100:.1f}%, DtWt:{data['weight_debt']*100:.1f}%
Model uses CapEx fade {data['capex_pct']*100:.1f}% -> {data['mature_capex_pct']*100:.1f}%.

Factor in recent news, earnings, guidance, analyst consensus. Reference specific events.
RESPOND ONLY JSON (no markdown): {{"wacc":<num>,"wacc_reasoning":"<1-2 sent>","revenue_growth":<num>,"revenue_growth_reasoning":"<1-2 sent>","terminal_growth":<num>,"terminal_growth_reasoning":"<1-2 sent>","projection_years":<int>,"projection_years_reasoning":"<1 sent>","overall_analysis":"<2-3 sent>"}}
Rules: WACC 6-15%, Terminal 1.5-3.5%, Years 5-15."""

    text = await claude_with_search(prompt, 1000)
    if not text: return None
    try:
        text = text.replace("```json","").replace("```","").strip()
        p = extract_json_obj(text)
        if not p: return None
        return {"wacc":round(max(5,min(float(p["wacc"]),20)),2),"wacc_reasoning":p.get("wacc_reasoning",""),
                "revenue_growth":round(max(-10,min(float(p["revenue_growth"]),35)),2),"revenue_growth_reasoning":p.get("revenue_growth_reasoning",""),
                "terminal_growth":round(max(1,min(float(p["terminal_growth"]),4)),2),"terminal_growth_reasoning":p.get("terminal_growth_reasoning",""),
                "projection_years":max(5,min(int(p.get("projection_years",10)),15)),"projection_years_reasoning":p.get("projection_years_reasoning",""),
                "overall_analysis":p.get("overall_analysis","")}
    except: return None

async def get_ai_news(ticker, name) -> list | None:
    prompt = f"""Search the web for the latest news about {name} ({ticker}) stock. Find recent earnings, analyst ratings, product news, regulatory developments, major events.
Provide 6 items as JSON array: [{{"headline":"...","summary":"2-3 sent with investment implications","sentiment":"positive/negative/neutral","category":"earnings/product/regulatory/market/management/macro/competitive/partnership"}}]
Return ONLY the JSON array."""
    text = await claude_with_search(prompt, 4096)
    if not text: return None
    try:
        text = text.replace("```json","").replace("```","").strip()
        arr = extract_json_arr(text)
        return arr[:8] if arr and isinstance(arr, list) else None
    except: return None

async def get_ai_memo(ticker, name, data) -> dict | None:
    prompt = f"""You are a senior equity research analyst writing an investment memo for {name} ({ticker}).

Company: {name} ({ticker}), Sector: {data['sector']}, Industry: {data['industry']}
Market Cap: ${data['market_cap']/1e9:.1f}B, Price: ${data['price']:.2f}
Revenue: ${data['latest_revenue']/1e9:.1f}B, EBIT Margin: {data['ebit_margin']:.1f}%
P/E: {f"{data['pe_ratio']:.1f}x" if data['pe_ratio'] else 'N/A'}, Beta: {data['beta'] or 'N/A'}
Cash: ${data['cash']/1e9:.1f}B, Debt: ${data['debt']/1e9:.1f}B

Search the web for recent news and developments, then write the memo.

RESPOND WITH ONLY THIS JSON (no markdown, no backticks, no extra text):
{{"thesis":"<3-4 sentence investment thesis>","bull_case":"<2-3 sentences on upside scenario>","bear_case":"<2-3 sentences on downside scenario>","catalysts":["<specific catalyst 1>","<specific catalyst 2>","<specific catalyst 3>"],"risks":["<specific risk 1>","<specific risk 2>","<specific risk 3>"],"key_metrics_to_watch":["<metric 1>","<metric 2>","<metric 3>"]}}"""

    # Try with web search first
    text = await claude_with_search(prompt, 2000)
    if not text:
        # Fallback: try without web search
        print(f"Memo web search failed for {ticker}, trying without search")
        text = await claude_no_search(prompt, 1500)
    if not text: return None
    try:
        text = text.replace("```json","").replace("```","").strip()
        result = extract_json_obj(text)
        if result and "thesis" in result:
            return result
        return None
    except Exception as e:
        print(f"Memo parse error: {e}, text was: {text[:200]}")
        return None

# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------
def run_dcf(data, wacc, terminal_growth, revenue_growth, years):
    revenue=data["latest_revenue"]; ebit=data["latest_ebit"]
    da_s=data["da_pct"]; cx_s=data["capex_pct"]; cx_m=data["mature_capex_pct"]
    tr=data["tax_rate"]; nwc_pct=data["nwc_pct_of_rev_change"]
    shares=data["shares_outstanding"]; cash=data["cash"]; debt=data["debt"]
    if revenue==0: raise HTTPException(status_code=400, detail="Revenue is zero.")
    em = ebit/revenue; da_m = cx_m if da_s >= cx_m else da_s
    proj=[]; prev=revenue
    for yr in range(1, years+1):
        r=revenue*((1+revenue_growth)**yr); e=r*em; nopat=e*(1-tr)
        f=(yr-1)/max(years-1,1)
        da=r*(da_s+(da_m-da_s)*f); cx=r*(cx_s+(cx_m-cx_s)*f)
        nwc=(r-prev)*nwc_pct; ufcf=nopat+da-cx-nwc; df=1/((1+wacc)**yr)
        proj.append({"year":yr,"revenue":round(r),"ebit":round(e),"nopat":round(nopat),
            "da":round(da),"capex":round(cx),"nwc_change":round(nwc),"ufcf":round(ufcf),
            "discount_factor":round(df,4),"pv_ufcf":round(ufcf*df),"capex_pct":round((cx_s+(cx_m-cx_s)*f)*100,1)})
        prev=r
    fufcf=proj[-1]["ufcf"]
    if wacc<=terminal_growth: raise HTTPException(status_code=400, detail="WACC must exceed terminal growth.")
    tv=(fufcf*(1+terminal_growth))/(wacc-terminal_growth); pvtv=tv/((1+wacc)**years)
    spv=sum(p["pv_ufcf"] for p in proj); ev=spv+pvtv; eq=ev+cash-debt
    imp=eq/shares if shares else 0; cp=data["price"]
    up=((imp-cp)/cp*100) if cp else 0
    rec="BUY" if up>15 else "SELL" if up<-15 else "HOLD"
    wr=[round(wacc+d,4) for d in [-.02,-.01,0,.01,.02]]
    tgr=[round(terminal_growth+d,4) for d in [-.01,-.005,0,.005,.01]]
    sens=[]
    for w in wr:
        row={"wacc":w,"values":[]}
        for tg in tgr:
            if w<=tg or w<=0: row["values"].append(None)
            else:
                t2=(fufcf*(1+tg))/(w-tg); pt=t2/((1+w)**years)
                s2=sum(p["ufcf"]/((1+w)**p["year"]) for p in proj)
                row["values"].append(round((s2+pt+cash-debt)/shares,2) if shares else None)
        sens.append(row)
    return {"projections":proj,"terminal_value":round(tv),"pv_terminal":round(pvtv),"sum_pv_ufcf":round(spv),
        "enterprise_value":round(ev),"equity_value":round(eq),"implied_share_price":round(imp,2),
        "current_price":cp,"upside_pct":round(up,2),"recommendation":rec,
        "ebit_margin":round(em*100,2),"tax_rate_used":round(tr*100,2),
        "capex_pct_start":round(cx_s*100,2),"capex_pct_mature":round(cx_m*100,2),
        "sensitivity":sens,"sensitivity_tg_range":tgr}

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def generate_pdf(company, assumptions, valuation):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.5*inch, leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallGray', parent=styles['Normal'], fontSize=8, textColor=colors.gray))
    styles.add(ParagraphStyle(name='SectionHead', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0e7490'), spaceAfter=6))
    styles.add(ParagraphStyle(name='MetricLabel', parent=styles['Normal'], fontSize=8, textColor=colors.gray))
    styles.add(ParagraphStyle(name='MetricVal', parent=styles['Normal'], fontSize=11, textColor=colors.black))
    story = []
    # Header
    story.append(Paragraph(f"DCF Valuation Report — {company['name']} ({company['ticker']})", styles['Title']))
    story.append(Paragraph(f"Generated {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')} | DCFengine", styles['SmallGray']))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#22d3ee')))
    story.append(Spacer(1, 12))
    # Recommendation
    v = valuation; rec_color = {"BUY":colors.green,"SELL":colors.red,"HOLD":colors.orange}.get(v["recommendation"], colors.gray)
    story.append(Paragraph(f"<font size=22 color='{rec_color.hexval()}'><b>{v['recommendation']}</b></font>", styles['Normal']))
    story.append(Paragraph(f"Market Price: ${v['current_price']:.2f}  |  Implied Value: ${v['implied_share_price']:.2f}  |  Upside: {v['upside_pct']:+.1f}%", styles['Normal']))
    story.append(Spacer(1, 16))
    # Assumptions
    story.append(Paragraph("Assumptions", styles['SectionHead']))
    a = assumptions
    adata = [["WACC", f"{a['wacc']*100:.2f}%", "Revenue Growth", f"{a['revenue_growth']*100:.2f}%"],
             ["Terminal Growth", f"{a['terminal_growth']*100:.2f}%", "Projection Years", str(a['projection_years'])]]
    at = Table(adata, colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 1.2*inch])
    at.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),9),('TEXTCOLOR',(0,0),(0,-1),colors.gray),('TEXTCOLOR',(2,0),(2,-1),colors.gray),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(at); story.append(Spacer(1, 12))
    # Valuation
    story.append(Paragraph("Valuation Summary", styles['SectionHead']))
    def fmtb(n):
        if abs(n)>=1e12: return f"${n/1e12:.2f}T"
        if abs(n)>=1e9: return f"${n/1e9:.2f}B"
        if abs(n)>=1e6: return f"${n/1e6:.1f}M"
        return f"${n:,.0f}"
    vrows = [["Sum PV (UFCF)", fmtb(v["sum_pv_ufcf"])],["PV Terminal Value", fmtb(v["pv_terminal"])],
             ["Enterprise Value", fmtb(v["enterprise_value"])],["+ Cash", fmtb(company["cash"])],
             ["- Debt", fmtb(company["debt"])],["Equity Value", fmtb(v["equity_value"])],
             ["Implied Share Price", f"${v['implied_share_price']:.2f}"],["Market Price", f"${v['current_price']:.2f}"]]
    vt = Table(vrows, colWidths=[3*inch, 2*inch])
    vt.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f0f9ff')),('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4)]))
    story.append(vt); story.append(Spacer(1, 12))
    # Projections
    story.append(Paragraph("Projected Free Cash Flows", styles['SectionHead']))
    pheader = ["Year","Revenue","EBIT","UFCF","PV"]
    prows = [pheader] + [[f"Yr {p['year']}", fmtb(p["revenue"]), fmtb(p["ebit"]), fmtb(p["ufcf"]), fmtb(p["pv_ufcf"])] for p in v["projections"]]
    pt = Table(prows, colWidths=[0.6*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    pt.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0e7490')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3)]))
    story.append(pt); story.append(Spacer(1, 16))
    story.append(Paragraph("Disclaimer: This is an educational tool, not financial advice. DCF models are sensitive to assumptions.", styles['SmallGray']))
    doc.build(story)
    buf.seek(0)
    return buf

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/api/lookup")
async def lookup_endpoint(req: TickerLookup):
    try:
        ticker = req.ticker.strip().upper()
        if not ticker: raise HTTPException(status_code=400, detail="Ticker required.")
        data = fetch_financials(ticker)
        ai = await get_ai_assumptions(data)
        now = datetime.now(timezone.utc).isoformat()
        if ai:
            assumptions = {"wacc":ai["wacc"],"revenue_growth":ai["revenue_growth"],"terminal_growth":ai["terminal_growth"],"projection_years":ai["projection_years"]}
            reasoning = {"wacc":ai["wacc_reasoning"],"revenue_growth":ai["revenue_growth_reasoning"],"terminal_growth":ai["terminal_growth_reasoning"],"projection_years":ai["projection_years_reasoning"],"overall":ai["overall_analysis"],"powered_by_ai":True}
        else:
            assumptions = {"wacc":round(data["calc_wacc"]*100,2),"revenue_growth":round(data["calc_revenue_growth"]*100,2),"terminal_growth":2.5,"projection_years":10}
            reasoning = {"wacc":"CAPM-based.","revenue_growth":"Historical average.","terminal_growth":"GDP proxy.","projection_years":"Standard 10yr.","overall":"AI unavailable.","powered_by_ai":False}
        return {"company":{"name":data["name"],"ticker":data["ticker"],"sector":data["sector"],"industry":data["industry"],
            "price":data["price"],"shares_outstanding":data["shares_outstanding"],"market_cap":data["market_cap"],
            "cash":data["cash"],"debt":data["debt"],"pe_ratio":data["pe_ratio"],"ev_ebitda":data["ev_ebitda"],"beta":data["beta"],
            "hist_revenue":data["hist_revenue"],"hist_ebit":data["hist_ebit"],"hist_net_income":data["hist_net_income"],
            "hist_fcf":data["hist_fcf"],"hist_years":data["hist_years"],"hist_margins":data["hist_margins"],"price_history":data["price_history"]},
            "auto_assumptions":assumptions,"ai_reasoning":reasoning,
            "wacc_breakdown":{"risk_free_rate":round(data["risk_free_rate"]*100,2),"equity_risk_premium":round(data["equity_risk_premium"]*100,2),
                "beta":data["beta"],"cost_of_equity":round(data["cost_of_equity"]*100,2),"cost_of_debt":round(data["cost_of_debt"]*100,2),
                "weight_equity":round(data["weight_equity"]*100,2),"weight_debt":round(data["weight_debt"]*100,2),"tax_rate":round(data["tax_rate"]*100,2)},
            "model_inputs":{"ebit_margin":round(data["ebit_margin"],2),"da_pct":round(data["da_pct"]*100,2),
                "capex_pct_current":round(data["capex_pct"]*100,2),"capex_pct_mature":round(data["mature_capex_pct"]*100,2),
                "nwc_pct":round(data["nwc_pct_of_rev_change"]*100,2)},
            "timestamp":now}
    except HTTPException: raise
    except Exception as e:
        print(f"ERROR lookup: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail":str(e)})

@app.post("/api/dcf")
def dcf_endpoint(req: DCFRequest):
    try:
        ticker=req.ticker.strip().upper()
        data=fetch_financials(ticker)
        result=run_dcf(data,req.wacc,req.terminal_growth,req.revenue_growth,req.projection_years)
        now=datetime.now(timezone.utc).isoformat()
        return {"company":{"name":data["name"],"ticker":data["ticker"],"sector":data["sector"],"price":data["price"],
            "shares_outstanding":data["shares_outstanding"],"market_cap":data["market_cap"],"cash":data["cash"],"debt":data["debt"],
            "pe_ratio":data["pe_ratio"],"ev_ebitda":data["ev_ebitda"],"beta":data["beta"],"hist_revenue":data["hist_revenue"]},
            "assumptions":{"wacc":req.wacc,"terminal_growth":req.terminal_growth,"revenue_growth":req.revenue_growth,"projection_years":req.projection_years},
            "valuation":result,"timestamp":now}
    except HTTPException: raise
    except Exception as e:
        print(f"ERROR dcf: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail":str(e)})



@app.post("/api/memo")
async def memo_endpoint(req: MemoRequest):
    try:
        ticker=req.ticker.strip().upper()
        data=fetch_financials(ticker)
        memo=await get_ai_memo(ticker,data["name"],data)
        return {"ticker":ticker,"company":data["name"],"memo":memo or {},"timestamp":datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        print(f"ERROR memo: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail":str(e)})

@app.post("/api/pdf")
def pdf_endpoint(req: PDFRequest):
    try:
        ticker=req.ticker.strip().upper()
        data=fetch_financials(ticker)
        result=run_dcf(data,req.wacc,req.terminal_growth,req.revenue_growth,req.projection_years)
        company={"name":data["name"],"ticker":data["ticker"],"cash":data["cash"],"debt":data["debt"]}
        assumptions={"wacc":req.wacc,"terminal_growth":req.terminal_growth,"revenue_growth":req.revenue_growth,"projection_years":req.projection_years}
        buf=generate_pdf(company,assumptions,result)
        return StreamingResponse(buf, media_type="application/pdf",
            headers={"Content-Disposition":f'attachment; filename="DCF_{ticker}.pdf"'})
    except HTTPException: raise
    except Exception as e:
        print(f"ERROR pdf: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail":str(e)})

@app.get("/api/health")
def health(): return {"status":"ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/{full_path:path}")
def serve_frontend(full_path: str = ""): return FileResponse("static/index.html")
