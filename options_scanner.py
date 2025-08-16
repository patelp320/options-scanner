"""
options_scanner.py
===================

This module implements a prototype of an options‑trading scanner that fetches
market data, identifies potential trading opportunities and prints trade ideas
at scheduled times.  It is designed to run on a MacBook (or any machine with
Python 3.9+ installed) and illustrates how different components of an automated
options strategy could be stitched together.  **Important:** this code is
provided for educational purposes only.  It does not constitute financial
advice, does not guarantee any returns and should not be used to place live
trades without proper due diligence and risk management.  You are solely
responsible for any trading decisions you make.

Key Features
------------

* **Scheduling:** The scanner runs twice per trading day – once shortly after
  the U.S. markets open (default 9:35 AM America/New_York) and again at
  10:30 AM.  The times can be adjusted via configuration.
* **Strategy templates:** The module defines data classes for common options
  strategies (covered call, protective put, vertical spreads, straddles,
  strangles, iron condor, etc.) and provides simple P&L calculations.  These
  templates can be extended to compute theoretical payoffs, risk metrics and
  probability of profit (POP).
* **Pluggable data sources:** The scanner reads live market data from
  configurable functions.  By default, these functions contain placeholders
  because online API access requires credentials.  You can supply your own
  implementations via environment variables or by editing the `fetch_*`
  functions.
* **News and social sentiment:** The scanner can integrate with news APIs
  (e.g. NewsAPI.org) and social media sentiment providers to identify
  trending tickers.  In this prototype these functions return empty lists by
  default.  Add your own API keys and parsing logic to enable them.
* **Risk management:** The strategy selection process considers max gain,
  max loss and required capital.  You can extend the ranking logic to
  incorporate volatility, Greeks (delta, gamma, theta) and other metrics.

Usage
-----

Run the script directly from the command line.  The scanner will
immediately perform one scan and then enter an infinite loop, waking up at
the scheduled times.  Use `Ctrl+C` to exit.

```sh
python options_scanner.py
```

Environment Variables
---------------------

Several optional environment variables can be set to configure the scanner:

```
NEWS_API_KEY        API key for a news service (e.g. newsapi.org)
SOCIAL_API_KEY      API key for a social sentiment service (e.g. Twitter)
DATA_API_KEY        API key for a market data service (e.g. Finnhub, IEX Cloud)
SYMBOL_LIST         Comma‑separated list of tickers to monitor (default: the
                    30 stocks of the Dow Jones Industrial Average)
TIMEZONE            Time zone in which to schedule scans (default: America/New_York)
SCAN_TIMES          Comma‑separated local times at which to run the scanner
                    (default: "09:35,10:30")
```

Notes
-----

1. The scanner relies on external APIs for real‑time options and underlying
   data.  Without valid API keys the `fetch_*` functions will return empty
   structures and the scanner will not generate trade ideas.  You can
   implement these functions to retrieve data from sources such as Finnhub,
   AlphaVantage, IEX Cloud, Yahoo Finance or your broker’s API.
2. Scheduling uses the built‑in `time` module rather than external
   dependencies.  This keeps the requirements minimal but also means the
   script must run continuously to trigger at the right times.
3. The examples of strategies and selection criteria are greatly simplified.
   They serve as a starting point for further development and are not
   guaranteed to be profitable.  Extend them with your own analytics to
   reflect your risk tolerance and market outlook.

"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


###############################################################################
# Strategy definitions
###############################################################################

@dataclass
class OptionLeg:
    """Represents a single option leg."""

    kind: str  # 'call' or 'put'
    strike: float
    premium: float  # price paid (debit) or received (credit) per share
    quantity: int   # positive for long, negative for short

    def payoff(self, price: float) -> float:
        """Calculate payoff of this leg at expiration per share.

        Positive numbers represent profit, negative numbers represent loss.

        Args:
            price: Underlying asset price at expiration.

        Returns:
            Payoff per share (premium included).
        """
        intrinsic = 0.0
        if self.kind == 'call':
            intrinsic = max(price - self.strike, 0.0)
        elif self.kind == 'put':
            intrinsic = max(self.strike - price, 0.0)
        else:
            raise ValueError(f"Invalid option kind: {self.kind}")
        # For short positions, quantity will be negative and premium should be
        # received; for long positions, quantity positive and premium paid.
        return self.quantity * (intrinsic - self.premium)


@dataclass
class Strategy:
    """Base class for options strategies."""

    legs: List[OptionLeg]
    underlying_quantity: int = 0  # positive for long stock, negative for short

    def payoff(self, price: float) -> float:
        """Calculate total payoff (per share of stock) at expiration."""
        stock_payoff = self.underlying_quantity * (price - self.entry_price)
        option_payoff = sum(leg.payoff(price) for leg in self.legs)
        return stock_payoff + option_payoff

    @property
    def entry_price(self) -> float:
        """Override in subclasses if needed to track entry price of underlying."""
        return 0.0

    def max_gain(self) -> Optional[float]:
        """Estimate maximum possible gain.  Returns None if unlimited."""
        return None

    def max_loss(self) -> Optional[float]:
        """Estimate maximum possible loss.  Returns None if unlimited."""
        return None


@dataclass
class CoveredCall(Strategy):
    """Implements a covered call strategy.

    The trader owns 100 shares of the underlying and sells a call against it.

    """

    underlying_quantity: int = 100  # long 100 shares
    entry_price: float = 0.0        # purchase price per share

    def max_gain(self) -> float:
        if not self.legs:
            return None
        call = self.legs[0]
        # Max gain occurs if stock settles at the short call strike or above.
        return (call.strike - self.entry_price + call.premium) * call.quantity

    def max_loss(self) -> float:
        # Loss limited to full purchase cost minus premium received
        call = self.legs[0]
        return (self.entry_price - call.premium) * abs(call.quantity)


@dataclass
class ProtectivePut(Strategy):
    """A married put/protective put strategy.

    The trader owns stock and buys a put to protect downside.

    """

    underlying_quantity: int = 100
    entry_price: float = 0.0

    def max_gain(self) -> Optional[float]:
        # Unlimited upside
        return None

    def max_loss(self) -> float:
        put = self.legs[0]
        # Max loss occurs if stock goes to zero
        return (self.entry_price - put.strike + put.premium) * self.underlying_quantity


@dataclass
class VerticalSpread(Strategy):
    """Represents both bull call and bear put spreads."""

    def max_gain(self) -> float:
        long_leg, short_leg = self.legs
        width = abs(short_leg.strike - long_leg.strike)
        net_premium = -sum(leg.premium * leg.quantity for leg in self.legs)
        return width - net_premium

    def max_loss(self) -> float:
        net_premium = -sum(leg.premium * leg.quantity for leg in self.legs)
        return net_premium


@dataclass
class Straddle(Strategy):
    """Long straddle: long call + long put at same strike."""

    def max_gain(self) -> Optional[float]:
        # Unlimited upside if underlying moves significantly
        return None

    def max_loss(self) -> float:
        return sum(leg.premium * leg.quantity for leg in self.legs)


@dataclass
class Strangle(Strategy):
    """Long strangle: long OTM call + long OTM put."""

    def max_gain(self) -> Optional[float]:
        return None

    def max_loss(self) -> float:
        return sum(leg.premium * leg.quantity for leg in self.legs)


@dataclass
class IronCondor(Strategy):
    """Iron condor constructed with four legs."""

    def max_gain(self) -> float:
        # Premium received minus width of spreads on losing side
        premium = -sum(leg.premium * leg.quantity for leg in self.legs)
        short_put, long_put, short_call, long_call = sorted(self.legs, key=lambda x: x.strike)
        width = min(long_call.strike - short_call.strike, short_put.strike - long_put.strike)
        return premium

    def max_loss(self) -> float:
        premium = -sum(leg.premium * leg.quantity for leg in self.legs)
        short_put, long_put, short_call, long_call = sorted(self.legs, key=lambda x: x.strike)
        width = min(long_call.strike - short_call.strike, short_put.strike - long_put.strike)
        return width - premium


###############################################################################
# Data retrieval stubs
###############################################################################

def fetch_trending_symbols() -> List[str]:
    """Fetch a list of trending symbols based on news and social sentiment.

    Returns an empty list by default.  To enable, implement calls to a news
    API (e.g. NewsAPI.org) and a social sentiment API (e.g. Twitter) and
    return a deduplicated list of tickers (e.g. ['AAPL', 'TSLA']).
    """
    return []


def fetch_options_chain(symbol: str) -> List[Dict[str, object]]:
    """Retrieve the options chain for a given symbol.

    Each entry in the returned list should contain at least the following keys:
    ``expiry``, ``strike``, ``option_type`` ('call' or 'put'), ``bid``, ``ask``,
    ``last``, ``volume``, ``open_interest``.  Data should be for the nearest
    weekly expiry (or 0DTE if scanning intraday).  To implement this function
    you can use a third‑party service (Finnhub, AlphaVantage, IEX Cloud, etc.)
    or broker APIs.  Without a data source the function returns an empty
    list.
    """
    return []


def fetch_underlying_price(symbol: str) -> Optional[float]:
    """Fetch the current price of the underlying stock.  Returns None if no
    data is available.  Implement this with your preferred market data API.
    """
    return None


###############################################################################
# Scanner logic
###############################################################################

def identify_candidate_trades(symbol: str) -> List[Tuple[str, Strategy]]:
    """Scan the options chain for a given symbol and generate trade ideas.

    The function returns a list of tuples ``(description, strategy)``.  The
    description should be a human‑readable string summarising the trade (e.g.
    "Bull call spread: buy 50 call @1.00, sell 55 call @0.50, expiring 2025‑08‑29")
    and `strategy` is an instance of a Strategy subclass representing the
    structure.  This prototype selects high‑volume contracts and constructs
    simple vertical spreads as an example.  Extend this logic to include more
    sophisticated filters (volatility, delta, risk/reward ratio, etc.).
    """
    options_chain = fetch_options_chain(symbol)
    if not options_chain:
        return []
    # Group by option type for demonstration
    calls = [opt for opt in options_chain if opt['option_type'] == 'call']
    puts = [opt for opt in options_chain if opt['option_type'] == 'put']
    # Sort by volume descending
    calls.sort(key=lambda x: x.get('volume', 0), reverse=True)
    puts.sort(key=lambda x: x.get('volume', 0), reverse=True)
    candidates: List[Tuple[str, Strategy]] = []
    # Example: create a bull call spread from top two calls with ascending strikes
    if len(calls) >= 2:
        long = calls[0]
        short = next((c for c in calls[1:] if c['strike'] > long['strike']), None)
        if short:
            long_leg = OptionLeg('call', long['strike'], (long['ask'] + long['bid']) / 2, +1)
            short_leg = OptionLeg('call', short['strike'], (short['ask'] + short['bid']) / 2, -1)
            spread = VerticalSpread(legs=[long_leg, short_leg])
            desc = (
                f"Bull call spread on {symbol}: buy {long_leg.strike} call @~{long_leg.premium:.2f}, "
                f"sell {short_leg.strike} call @~{short_leg.premium:.2f}, expiring {long['expiry']}"
            )
            candidates.append((desc, spread))
    # Example: create a bear put spread from top two puts with ascending strikes
    if len(puts) >= 2:
        long = puts[0]
        short = next((p for p in puts[1:] if p['strike'] < long['strike']), None)
        if short:
            long_leg = OptionLeg('put', long['strike'], (long['ask'] + long['bid']) / 2, +1)
            short_leg = OptionLeg('put', short['strike'], (short['ask'] + short['bid']) / 2, -1)
            spread = VerticalSpread(legs=[long_leg, short_leg])
            desc = (
                f"Bear put spread on {symbol}: buy {long_leg.strike} put @~{long_leg.premium:.2f}, "
                f"sell {short_leg.strike} put @~{short_leg.premium:.2f}, expiring {long['expiry']}"
            )
            candidates.append((desc, spread))
    return candidates


def scan_once(symbols: List[str]) -> None:
    """Perform a single scan across a list of symbols and print results."""
    for symbol in symbols:
        try:
            ideas = identify_candidate_trades(symbol)
        except Exception as exc:
            print(f"Error scanning {symbol}: {exc}")
            continue
        if not ideas:
            print(f"No trade ideas for {symbol}.")
            continue
        print(f"\nTrade ideas for {symbol}:")
        for desc, strat in ideas:
            max_gain = strat.max_gain()
            max_loss = strat.max_loss()
            print(f"  {desc}")
            print(f"    Max gain: {max_gain if max_gain is not None else 'Unlimited'}")
            print(f"    Max loss: {max_loss if max_loss is not None else 'Unlimited'}")


def parse_symbols() -> List[str]:
    """Parse the SYMBOL_LIST environment variable into a list."""
    env = os.getenv('SYMBOL_LIST')
    if env:
        return [s.strip().upper() for s in env.split(',') if s.strip()]
    # Default to the Dow 30 constituents
    return [
        'AAPL', 'MSFT', 'IBM', 'JPM', 'XOM', 'V', 'JNJ', 'WMT', 'PG', 'UNH',
        'KO', 'HD', 'INTC', 'CRM', 'MCD', 'CAT', 'HON', 'GS', 'DIS', 'VZ',
        'NKE', 'TRV', 'AXP', 'BA', 'MRK', 'CSCO', 'CVX', 'DOW', 'AMGN', 'MMM',
    ]


def get_scan_times() -> List[_dt.time]:
    """Return a list of times (today) at which to run the scanner."""
    tz = os.getenv('TIMEZONE', 'America/New_York')
    times_str = os.getenv('SCAN_TIMES', '09:35,10:30')
    times = []
    for tstr in times_str.split(','):
        try:
            hour, minute = map(int, tstr.strip().split(':'))
            times.append(_dt.time(hour, minute))
        except ValueError:
            print(f"Invalid time format in SCAN_TIMES: {tstr}")
    # Remove duplicates and sort
    return sorted(set(times))


def wait_until_next_scan(scan_times: List[_dt.time]) -> None:
    """Sleep until the next scheduled scan time."""
    now = _dt.datetime.now()
    today = now.date()
    # Build list of datetime objects for today or tomorrow
    schedule: List[_dt.datetime] = []
    for t in scan_times:
        dt_today = _dt.datetime.combine(today, t)
        if dt_today > now:
            schedule.append(dt_today)
        else:
            # schedule for next day
            dt_next = _dt.datetime.combine(today + _dt.timedelta(days=1), t)
            schedule.append(dt_next)
    next_run = min(schedule)
    delta = next_run - now
    seconds = delta.total_seconds()
    if seconds > 0:
        print(f"Sleeping {seconds/60:.1f} minutes until {next_run.strftime('%Y-%m-%d %H:%M')} …")
        time.sleep(seconds)


def main() -> None:
    print("Starting options scanner …")
    symbols = parse_symbols()
    scan_times = get_scan_times()
    # Immediately run one scan on start
    print(f"Initial scan at { _dt.datetime.now().strftime('%Y-%m-%d %H:%M') }")
    scan_once(symbols)
    while True:
        wait_until_next_scan(scan_times)
        print(f"\nRunning scheduled scan at { _dt.datetime.now().strftime('%Y-%m-%d %H:%M') }")
        scan_once(symbols)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Scanner terminated by user.")