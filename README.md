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
  strategy definitions, stub functions for fetching data, scheduling logic and
  a simple selection mechanism for vertical spreads.
* `README.md` – This document.

## How it Works

1. **Scheduling:**  When run, the script performs one scan immediately and then
   sleeps until the next scheduled time.  Scheduled times default to
   09:35 AM and 10:30 AM America/New_York but can be changed via the
   `SCAN_TIMES` environment variable (comma‑separated list of `HH:MM` values).
2. **Data Retrieval:**  The functions `fetch_trending_symbols`,
   `fetch_options_chain` and `fetch_underlying_price` are stubs.  To make the
   scanner functional you must supply implementations that return real news
   data, social sentiment and options chains.  See the docstrings in
   `options_scanner.py` for guidance.  API keys can be provided via
   environment variables (`NEWS_API_KEY`, `SOCIAL_API_KEY`, `DATA_API_KEY`).
3. **Strategy Selection:**  The prototype includes simple data classes for
   covered calls, protective puts, vertical spreads, straddles, strangles and
   iron condors.  Only vertical spreads are used by default.  You can extend
   the `identify_candidate_trades` function to build more complex structures or
   to filter trades by probability of profit (POP), implied volatility rank
   (IVR), delta or any other metric.  The latest version annotates each
   candidate trade with a **risk/reward ratio** and a **risk level** (Low,
   Medium, High) based on the strategy’s maximum gain and loss.  This helps
   prioritise trades according to risk tolerance.
4. **Scanning Universe:**  By setting the environment variable `SCAN_ALL=true`
   the scanner will call `fetch_all_symbols()` (which you must implement) to
   return a universe of optionable stocks.  Otherwise it scans the tickers
   specified in `SYMBOL_LIST` (or defaults to the Dow 30).
4. **Execution:**  The script prints proposed trades to the console.  It does
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
   export SYMBOL_LIST="AAPL,TSLA,MSFT"
   export SCAN_TIMES="09:35,10:30"
   export NEWS_API_KEY="your_news_api_key"
   export SOCIAL_API_KEY="your_social_api_key"
   export DATA_API_KEY="your_market_data_api_key"
   ```

4. Run the script:

   ```sh
   python options_scanner.py
   ```

The program will perform a scan immediately and then wait until the next
scheduled run time.  To stop the program press `Ctrl+C`.

## Limitations and Future Work

* **Live data:**  The scanner is not useful without real market data.
  Implement data retrieval using a service like Finnhub, AlphaVantage or your
  broker’s API.  Note that many free APIs do not include options data.
* **Machine learning and reinforcement learning:**  The current version does
  not incorporate any learning algorithm.  To make the scanner “learn” from
  past trades you would need to capture data, label outcomes and train a
  model.  Reinforcement learning for options trading is an advanced topic
  beyond the scope of this prototype.
* **Backtesting:**  Before running any automated strategy live you should
  backtest it on historical data to evaluate performance, risk and drawdowns.
* **Risk management:**  The example trade selection is simplistic.  In
  practice you should consider risk metrics such as maximum drawdown,
  volatility, correlation between positions and portfolio allocation.

By using this prototype as a starting point you can incrementally develop a
more sophisticated options scanner tailored to your trading style and risk
tolerance.