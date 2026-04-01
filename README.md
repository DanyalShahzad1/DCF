# DCF Valuation Platform

A web-based DCF valuation tool. Enter any public company ticker, set your assumptions (WACC, growth rates), and get a Buy / Hold / Sell recommendation with full model breakdown and Excel export.

## How It Works

```
User enters ticker + assumptions
        ↓
FastAPI backend calls yfinance
        ↓
Backend returns clean financials
        ↓
Backend runs DCF model
        ↓
Frontend displays: recommendation, projections, sensitivity table
        ↓
User can download full Excel model
```

## Tech Stack

- **Backend**: Python, FastAPI, yfinance
- **Frontend**: HTML/CSS/JS (served by FastAPI)
- **Hosting**: Railway
- **Excel Export**: SheetJS (client-side)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000

# Open browser
# http://localhost:8000
```

## Deploy to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app)
3. Click **New Project** → **Deploy from GitHub Repo**
4. Select this repo
5. Railway auto-detects Python and uses the `Procfile`
6. It will deploy automatically — you'll get a public URL

That's it. No environment variables needed. No API keys.

## Project Structure

```
dcf-platform/
├── main.py              # FastAPI app (API + serves frontend)
├── requirements.txt     # Python dependencies
├── Procfile             # Railway start command
├── railway.toml         # Railway config
├── .python-version      # Python version
├── .gitignore
├── static/
│   └── index.html       # Full frontend (single file)
└── README.md
```

## Features

- **Any ticker**: Search any publicly traded company
- **Adjustable assumptions**: WACC, revenue growth, terminal growth
- **Full DCF breakdown**: Revenue → EBIT → NOPAT → UFCF → PV
- **Valuation bridge**: Enterprise value → equity value → implied share price
- **Sensitivity analysis**: WACC vs terminal growth matrix
- **Buy / Hold / Sell**: Recommendation based on upside/downside vs market price
- **Excel export**: Download the full model as .xlsx

## How the DCF Works

1. Pulls historical financials via yfinance
2. Calculates EBIT margin from latest data
3. Projects revenue using user-defined growth rate
4. Applies margin to get projected EBIT
5. Computes NOPAT (EBIT × (1 - tax rate))
6. Adds back D&A, subtracts CapEx and change in NWC → Unlevered FCF
7. Discounts UFCFs at user-defined WACC
8. Calculates terminal value using perpetuity growth method
9. Sums PV of UFCFs + PV of terminal value = Enterprise Value
10. Enterprise Value + Cash - Debt = Equity Value
11. Equity Value ÷ Shares Outstanding = Implied Share Price
12. Compares to market price → recommendation

## Recommendation Logic

- **BUY**: Implied value is 15%+ above market price
- **SELL**: Implied value is 15%+ below market price
- **HOLD**: Within 15% range

## Important Disclaimers

- This is a learning project, not financial advice
- DCF models are highly sensitive to assumptions
- Terminal value often drives a large portion of total valuation
- yfinance data may not always be perfectly accurate
- Different industries may require different modeling approaches
