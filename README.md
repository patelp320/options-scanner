# Options Scanner Prototype

This repository contains a prototype Python script that illustrates how one might
build an automated options scanner to generate candidate option trades.  The
project was created in response to a request to design a scanner that can
identify profitable trades twice each trading day (shortly after the market
opens and again an hour later), incorporate news and social sentiment, and
continuously learn from new data.

> ⚠️ **Disclaimer:**  This code is provided for educational and research
> purposes only.  It does not constitute financial advice.  Options trading
> involves significant risk and is not suitable for every investor.  There are
> no guarantees of profit.  Always consult a qualified financial advisor
> before making trading decisions.

## Contents

* `options_scanner.py` – Main program implementing the scanner.  Contains
  strategy definitions, stub functions for fetching data, scheduling logic
  and a simple selection mechanism for vertical spreads **plus a prototype
  reinforcement‑learning agent**.  The RL component builds a Q‑table over
  discretised market states and actions (strategy identifiers) and uses
  ε‑greedy exploration/exploitation to recommend one trade per symbol.  A
  lightweight logging system records trade proposals for later analysis and
  learning.
* `README.md` – This document.

## How it Works

1. **Scheduling:**  When run, the script performs one scan immediately and then
   sleeps until the next scheduled time.  Scheduled times default to
   09:35 AM and 10:30 AM America/New_York but can be changed via the
   `SCAN_TIMES` environment variable (comma‑separated list of `HH:MM` values).
2. **Data Retrieval:**  The functions `fetch_options_chain` and
   `fetch_underlying_price` are integrated with the optional
   `yfinance` library, which pulls stock and option data from Yahoo Finance
   without requiring API keys.  To enable this, install `yfinance` (`pip
   install yfinance`) and set `USE_YFINANCE=true` in your environment.  Note
   that Yahoo Finance data may be delayed or incomplete, so it should be
   used for research and prototyping rather than production trading.
3. **Strategy Selection:**  The prototype includes simple data classes for
   covered calls, protective puts, vertical spreads, straddles, strangles and
   iron condors.  Only vertical spreads are used by default.  You can extend
   the `identify_candidate_trades` function to build more complex structures or
   to filter trades by probability of profit (POP), implied volatility rank
   (IVR), delta or any other metric.  The latest version annotates each
   candidate trade with a **risk/reward ratio** and a **risk level** (Low,
   Medium, High) based on the strategy’s maximum gain and loss, helping
   prioritise trades according to risk tolerance.

   The current version also includes a **reinforcement‑learning module**.
   When enabled, the scanner derives a simplified market state for each
   ticker (currently just the rounded underlying price) and consults a
   tabular Q‑learning agent to choose one candidate strategy.  After each
   scan, the chosen trade proposal is appended to a local `trade_log.json`.
   Once the trade is closed and a profit or loss is known, the record can
   be updated with the final result and next state.  At the start of each
   day, the agent loads existing trade records and updates its Q‑table
   accordingly using a binary reward (positive for profit, negative for
   loss)【291481126490656†L270-L279】.  This allows the scanner to gradually favour
   strategies that historically performed well and avoid those that did not.
   **Warning:** reinforcement learning in trading is highly experimental and
   susceptible to noise【291481126490656†L655-L668】.  You should backtest and
   validate any learned policy before using it with real capital, and always
   maintain human oversight【848725721887257†L94-L122】.
4. **Scanning Universe:**  By default the scanner processes the tickers
   specified in `SYMBOL_LIST` (or defaults to the Dow 30).  If you set
   `SCAN_ALL=true`, it will attempt to call `fetch_all_symbols()`.  Without
   external data this function returns an empty list; you can populate it
   manually or load a CSV of optionable stocks if desired.
5. **Execution:**  The script prints proposed trades to the console.  At
   startup it performs a **pre‑market scan** using free `yfinance` data to
   compute simple momentum and relative‑volume metrics, assigns a
   rudimentary confidence score and prints a list of "GO" cards.  Each
   card shows the trade description, confidence score, risk level and
   estimated max gain/loss.  This approximates the "Best‑of‑Best"
   watchlist described in the outline.  After printing the GO cards, it
   runs the standard scan and reinforcement‑learning update.  The script does
   *not* execute any trades; it merely surfaces ideas for manual review.  If
   you decide to integrate with a broker API to place trades automatically,
   ensure you implement appropriate safeguards and obtain explicit
   confirmation before execution to comply with your broker’s terms and
   relevant regulations.
   *not* execute any trades.  If you decide to integrate with a broker API to
   place trades automatically, ensure you implement appropriate safeguards and
   obtain explicit confirmation before execution to comply with your broker’s
   terms and relevant regulations.

## Running the Scanner

1. Install the required dependencies.  The script uses only the Python
   standard library, so no additional packages are required to run the default
   version.  However, you will need external libraries or API clients for
   retrieving real‑time options data.
2. Clone the repository and navigate into it:

   ```sh
   git clone <repo_url>
   cd <repo_name>
   ```

3. Set any desired environment variables (optional):

   ```sh
   export SYMBOL_LIST="AAPL,TSLA,MSFT"   # watch list
   export SCAN_TIMES="09:35,10:30"       # when to run the scanner
   export USE_YFINANCE=true             # enable Yahoo Finance data
   ```
   
   If you later integrate your own data sources or sentiment feeds, you can
   set additional environment variables (e.g. `NEWS_API_KEY`, `SOCIAL_API_KEY`,
   `DATA_API_KEY`) and modify the corresponding functions in
   `options_scanner.py`.  For now, these are not required.

4. Run the script:

   ```sh
   python options_scanner.py
   ```

The program will perform a scan immediately and then wait until the next
scheduled run time.  To stop the program press `Ctrl+C`.

## Limitations and Future Work

* **Live data:**  The scanner’s pre‑market scan relies solely on free
  `yfinance` data.  This data is delayed and may omit key fields such as
  live bid/ask, IV surfaces and intraday volume.  To realise the full
  capabilities outlined (DataGuard, NewsPulse, ShockGuard, etc.) you would
  need streaming data from a vendor or broker API that includes options
  quotes, greeks, implied volatility surfaces and news.
* **Machine learning and reinforcement learning:**  A basic tabular
  reinforcement‑learning agent is provided as a proof of concept.  It
  discretises the market state and learns Q‑values for each action.  This
  implementation is intentionally simple and may not capture the
  complexities of real markets.  In practice you might replace it with a
  neural network or ensemble model that takes more features (e.g. technical
  indicators, implied volatility, news sentiment) and outputs action
  probabilities.  Even with advanced models, reinforcement learning faces
  challenges such as delayed rewards and non‑stationary data【291481126490656†L203-L215】.
  Thorough backtesting, cross‑validation and robust risk management are
  essential before relying on any ML algorithm for live trading.
* **Backtesting:**  Before running any automated strategy live you should
  backtest it on historical data to evaluate performance, risk and drawdowns.
* **Risk management:**  The example trade selection is simplistic.  In
  practice you should consider risk metrics such as maximum drawdown,
  volatility, correlation between positions and portfolio allocation.

By using this prototype as a starting point you can incrementally develop a
more sophisticated options scanner tailored to your trading style and risk
tolerance.