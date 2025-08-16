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
USE_YFINANCE        If "true", use the `yfinance` library to pull stock
                    prices and options chains from Yahoo Finance (default: false).
SYMBOL_LIST         Comma‑separated list of tickers to monitor (default: the
                    30 stocks of the Dow Jones Industrial Average)
TIMEZONE            Time zone in which to schedule scans (default: America/New_York)
SCAN_TIMES          Comma‑separated local times at which to run the scanner
                    (default: "09:35,10:30")
```

You may also set `NEWS_API_KEY`, `SOCIAL_API_KEY` and `DATA_API_KEY` if you
decide to integrate third‑party news, sentiment or market data services in
the future.  These are not required when using `yfinance`.

Notes
-----

1. By default the scanner uses Yahoo Finance via `yfinance` for market
   data when `USE_YFINANCE=true`.  This data is provided without API keys but
   may be delayed or incomplete.  If you choose to integrate another data
   source or broker API, implement the `fetch_*` functions accordingly and
   set the appropriate API keys via environment variables.
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

# Attempt to import yfinance for free market data.  This optional
# dependency allows the scanner to retrieve stock prices and options
# chains without requiring a paid API key.  If not installed, the
# functions will fall back to returning None or an empty list.  Install
# via `pip install yfinance`.
try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False

# ===========================================================================
# NOTE ON RESPONSIBLE USE
#
# The additions below introduce a simple reinforcement‑learning (RL) framework
# and performance logging infrastructure.  RL can help discover patterns
# through trial and error, learning to favour actions that historically led to
# positive rewards.  However, financial markets are noisy and non‑stationary,
# which makes RL particularly challenging【291481126490656†L623-L651】.  Random noise
# or one‑off events may be mistaken for meaningful signals【291481126490656†L655-L668】,
# and the agent may overfit to past conditions.  Implementers must be
# cautious: backtest thoroughly, validate models out‑of‑sample, monitor
# performance and incorporate human oversight【848725721887257†L94-L122】.  These
# examples are for educational purposes and do not guarantee any profits.



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
    """Base class for options strategies.

    A strategy consists of one or more option legs and an optional
    underlying stock position.  Concrete subclasses implement
    `max_gain` and `max_loss` for payoff analysis.
    """

    legs: List[OptionLeg]
    underlying_quantity: int = 0  # positive for long stock, negative for short

    def payoff(self, price: float) -> float:
        """Calculate total payoff (per share of stock) at expiration.

        Payoff includes both option legs and any underlying stock position.
        """
        stock_payoff = self.underlying_quantity * (price - self.entry_price)
        option_payoff = sum(leg.payoff(price) for leg in self.legs)
        return stock_payoff + option_payoff

    @property
    def entry_price(self) -> float:
        """Entry price of the underlying stock.

        Subclasses may override if they need to track an entry price.  By
        default we assume zero, which effectively ignores stock payoff.
        """
        return 0.0

    def max_gain(self) -> Optional[float]:
        """Estimate maximum possible gain in dollars.

        Returns None if unlimited or not easily quantified.  Subclasses
        should override this method to provide a meaningful value.
        """
        return None

    def max_loss(self) -> Optional[float]:
        """Estimate maximum possible loss in dollars.

        Returns None if unlimited or not easily quantified.  Subclasses
        should override this method to provide a meaningful value.
        """
        return None

    def risk_reward_ratio(self) -> Optional[float]:
        """Compute the ratio of maximum loss to maximum gain.

        A lower ratio indicates a more favourable reward relative to the
        potential risk.  Returns None if either max gain or max loss is
        None (i.e. unlimited).
        """
        max_gain = self.max_gain()
        max_loss = self.max_loss()
        if max_gain is None or max_loss is None or max_gain == 0:
            return None
        return abs(max_loss) / max_gain

    def risk_level(self) -> str:
        """Categorise the strategy's risk level based on the risk/reward ratio.

        - 'Low'   : ratio ≤ 0.5
        - 'Medium': 0.5 < ratio ≤ 1.0
        - 'High'  : ratio > 1.0 or undefined

        Unlimited risk or unlimited reward results in 'High'.
        """
        ratio = self.risk_reward_ratio()
        if ratio is None:
            return 'High'
        if ratio <= 0.5:
            return 'Low'
        if ratio <= 1.0:
            return 'Medium'
        return 'High'


# ---------------------------------------------------------------------------
# Trade logging and reinforcement learning
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Represents the outcome of a historical trade.

    This data structure is used for self‑learning.  Each record stores
    metadata about the trade (symbol, strategy type, entry/exit prices and
    timestamps) along with the realised profit or loss.  These records
    populate the RL agent’s experience buffer.
    """
    symbol: str
    strategy_type: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    max_gain: Optional[float]
    max_loss: Optional[float]
    result: Optional[float]  # profit/loss; positive for profit, negative for loss
    state: Tuple  # representation of the market state when trade was opened
    next_state: Optional[Tuple]


class ReinforcementLearner:
    """Simple tabular Q‑learning agent for trade recommendation.

    The agent learns to map discrete market "states" to actions (strategy
    identifiers) based on realised rewards.  It uses an ε‑greedy policy
    during action selection and updates its Q‑table using the Bellman
    equation.  In practice, continuous state spaces require function
    approximation (e.g. neural networks), but this prototype keeps things
    simple for illustration.
    """

    def __init__(self) -> None:
        # Q‑table: {state: {action: value}}
        self.q_table: Dict[Tuple, Dict[str, float]] = {}

    def get_action(self, state: Tuple, possible_actions: List[str], epsilon: float = 0.1) -> str:
        """Choose an action using ε‑greedy selection.

        With probability ε, select a random action to encourage exploration;
        otherwise, select the action with the highest Q‑value for the given
        state.  If the state is unseen, it initialises Q‑values to zero.
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in possible_actions}
        # Exploration
        import random
        if random.random() < epsilon:
            return random.choice(possible_actions)
        # Exploitation
        action_values = self.q_table[state]
        return max(action_values, key=action_values.get)

    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple, alpha: float = 0.1, gamma: float = 0.9) -> None:
        """Update the Q‑table based on an observed transition.

        Args:
            state: The previous state.
            action: The action taken in the previous state.
            reward: The reward obtained after taking the action.
            next_state: The state after taking the action.
            alpha: Learning rate (0 < alpha ≤ 1).
            gamma: Discount factor (0 ≤ gamma < 1) determining future reward weight.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0}
        old_value = self.q_table[state].get(action, 0.0)
        next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        # Q‑learning update rule
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        self.q_table[state][action] = new_value


def extract_state(symbol: str) -> Tuple:
    """Derive a simplified market state representation for RL.

    In a real implementation, the state could include technical indicators
    (e.g. moving averages, RSI), volatility, implied volatility skew, news
    sentiment and macro factors.  Here we use a placeholder: the current
    underlying price rounded to the nearest whole number.  Extend this
    function to incorporate your preferred features.
    """
    price = fetch_underlying_price(symbol)
    if price is None:
        # Unknown price becomes a generic state
        return ('unknown',)
    # Round to reduce dimensionality (binning)
    return (round(price),)


def compute_reward(record: TradeRecord) -> float:
    """Compute a binary reward based on the trade outcome.

    Reinforcement learning in trading often benefits from simple rewards
    (positive for profitable trades, negative for losses)【291481126490656†L270-L279】.  This
    encourages consistent positive returns rather than chasing outsized gains.
    """
    if record.result is None:
        return 0.0
    return 1.0 if record.result > 0 else -1.0


def load_trade_records(path: str = 'trade_log.json') -> List[TradeRecord]:
    """Load historical trade records from disk for learning.

    The file should contain a list of dictionaries matching the TradeRecord
    fields.  Returns an empty list if no file exists.  Errors are ignored.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        records: List[TradeRecord] = []
        for item in data:
            records.append(TradeRecord(**item))
        return records
    except Exception:
        return []


def save_trade_records(records: List[TradeRecord], path: str = 'trade_log.json') -> None:
    """Persist trade records to a JSON file."""
    try:
        serialisable = [dataclasses.asdict(r) for r in records]
        with open(path, 'w') as f:
            json.dump(serialisable, f, indent=2)
    except Exception as exc:
        print(f"Could not save trade log: {exc}")


def learn_from_records(learner: ReinforcementLearner, records: List[TradeRecord]) -> None:
    """Update the RL agent using historical trade records.

    For each record where the trade has been closed (i.e. `result` and
    `next_state` are defined), compute a simple binary reward and update
    the Q‑table accordingly.  This method can be extended to use more
    sophisticated reward structures or discounting based on trade length.
    """
    for record in records:
        if record.result is not None and record.next_state is not None:
            reward = compute_reward(record)
            # Use the strategy type as the action identifier
            action = record.strategy_type
            learner.update(record.state, action, reward, record.next_state)



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
    """Return a list of trending symbols.

    Without third‑party APIs this function simply returns an empty list.
    You can manually populate a list here if you wish to monitor specific
    tickers beyond the static `SYMBOL_LIST`.  If you have access to a
    sentiment data feed, you can implement a call here.  For simplicity,
    this prototype relies solely on the watch list provided via
    `SYMBOL_LIST` or `USE_YFINANCE` when scanning the market.
    """
    return []


def fetch_options_chain(symbol: str) -> List[Dict[str, object]]:
    """Retrieve the options chain for a given symbol.

    The scanner uses Yahoo Finance via the optional `yfinance` library to
    obtain the nearest expiry options chain when `USE_YFINANCE=true` and
    `yfinance` is installed.  Each entry in the returned list contains
    the fields ``expiry``, ``strike``, ``option_type`` ('call' or 'put'),
    ``bid``, ``ask``, ``last``, ``volume`` and ``open_interest``.  If
    `yfinance` is unavailable or disabled, the function returns an
    empty list.
    """
    # First attempt to use yfinance if available and enabled via environment
    use_yf = os.getenv('USE_YFINANCE', 'false').lower() in {'1', 'true', 'yes'}
    if use_yf and _HAS_YFINANCE:
        try:
            ticker = yf.Ticker(symbol)
            expiries = ticker.options
            if not expiries:
                return []
            # Choose the nearest expiry
            expiry = sorted(expiries)[0]
            chain = ticker.option_chain(expiry)
            records: List[Dict[str, object]] = []
            for opt_df, opt_type in [(chain.calls, 'call'), (chain.puts, 'put')]:
                for _, row in opt_df.iterrows():
                    records.append({
                        'expiry': expiry,
                        'strike': float(row['strike']),
                        'option_type': opt_type,
                        'bid': float(row.get('bid', 0.0) or 0.0),
                        'ask': float(row.get('ask', 0.0) or 0.0),
                        'last': float(row.get('lastPrice', 0.0) or 0.0),
                        'volume': int(row.get('volume', 0) or 0),
                        'open_interest': int(row.get('openInterest', 0) or 0),
                    })
            return records
        except Exception:
            pass
    return []


def fetch_underlying_price(symbol: str) -> Optional[float]:
    """Fetch the current price of the underlying stock.

    When `USE_YFINANCE=true` and the optional `yfinance` library is
    available, this function returns the most recent close price from
    Yahoo Finance.  Otherwise it returns `None`.
    """
    # Attempt to use yfinance for a real price if enabled
    use_yf = os.getenv('USE_YFINANCE', 'false').lower() in {'1', 'true', 'yes'}
    if use_yf and _HAS_YFINANCE:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if not hist.empty:
                # Use the last close price
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
    return None

def fetch_all_symbols() -> List[str]:
    """Return a broad universe of tradable symbols.

    Without external data sources this function returns an empty list.  To
    scan the entire market you can manually load a list of optionable stocks
    from a CSV file or package it into the repository.  Alternatively,
    leave `SCAN_ALL=false` and provide a custom watch list via the
    `SYMBOL_LIST` environment variable.
    """
    return []


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
            # Compute risk/reward ratio and descriptive string
            ratio = spread.risk_reward_ratio()
            risk_label = spread.risk_level()
            ratio_str = f"{ratio:.2f}" if ratio is not None else '∞'
            desc = (
                f"Bull call spread on {symbol}: buy {long_leg.strike} call @~{long_leg.premium:.2f}, "
                f"sell {short_leg.strike} call @~{short_leg.premium:.2f}, exp {long['expiry']}. "
                f"Risk/Reward={ratio_str}, Risk Level={risk_label}"
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
            ratio = spread.risk_reward_ratio()
            risk_label = spread.risk_level()
            ratio_str = f"{ratio:.2f}" if ratio is not None else '∞'
            desc = (
                f"Bear put spread on {symbol}: buy {long_leg.strike} put @~{long_leg.premium:.2f}, "
                f"sell {short_leg.strike} put @~{short_leg.premium:.2f}, exp {long['expiry']}. "
                f"Risk/Reward={ratio_str}, Risk Level={risk_label}"
            )
            candidates.append((desc, spread))
    return candidates


def select_trade(symbol: str, learner: Optional[ReinforcementLearner] = None) -> Optional[Tuple[str, Strategy]]:
    """Select a single trade idea for a symbol using RL (if provided).

    This wrapper uses `identify_candidate_trades` to generate candidate
    strategies and then either returns the first candidate (if no learner) or
    employs ε‑greedy selection via the provided `ReinforcementLearner`.  It
    returns a tuple (description, strategy) or None if no candidates.
    """
    candidates = identify_candidate_trades(symbol)
    if not candidates:
        return None
    # If no RL learner, simply choose the first candidate
    if learner is None:
        return candidates[0]
    # Build the state and actions list
    state = extract_state(symbol)
    actions = [desc for desc, _ in candidates]
    chosen_desc = learner.get_action(state, actions)
    # Find the matching strategy
    for desc, strat in candidates:
        if desc == chosen_desc:
            return (desc, strat)
    # Fallback
    return candidates[0]


def scan_once(symbols: List[str], learner: Optional[ReinforcementLearner] = None, trade_log: Optional[List[TradeRecord]] = None) -> None:
    """Perform a single scan across a list of symbols and print results.

    If a `ReinforcementLearner` is supplied, the scanner will use it to
    prioritise one trade per symbol.  All proposed trades can be stored in
    `trade_log` for later analysis and learning.  In the absence of a
    learner, the function behaves like the original scanner and prints all
    candidate trades.
    """
    for symbol in symbols:
        try:
            candidates = identify_candidate_trades(symbol)
        except Exception as exc:
            print(f"Error scanning {symbol}: {exc}")
            continue
        if not candidates:
            print(f"No trade ideas for {symbol}.")
            continue
        # Use RL selection if a learner is provided
        if learner:
            selection = select_trade(symbol, learner)
            if selection is None:
                print(f"No trade ideas for {symbol}.")
                continue
            desc, strat = selection
            print(f"\nRecommended trade for {symbol}:")
            print(f"  {desc}")
            max_gain = strat.max_gain()
            max_loss = strat.max_loss()
            print(f"    Max gain: {max_gain if max_gain is not None else 'Unlimited'}")
            print(f"    Max loss: {max_loss if max_loss is not None else 'Unlimited'}")
            # Record the recommendation for learning
            if trade_log is not None:
                record = TradeRecord(
                    symbol=symbol,
                    strategy_type=type(strat).__name__,
                    entry_date=_dt.datetime.now().strftime('%Y-%m-%d'),
                    exit_date=None,
                    entry_price=fetch_underlying_price(symbol) or 0.0,
                    exit_price=None,
                    max_gain=max_gain,
                    max_loss=max_loss,
                    result=None,
                    state=extract_state(symbol),
                    next_state=None,
                )
                trade_log.append(record)
        else:
            # Default behaviour: list all candidate trades
            print(f"\nTrade ideas for {symbol}:")
            for desc, strat in candidates:
                max_gain = strat.max_gain()
                max_loss = strat.max_loss()
                print(f"  {desc}")
                print(f"    Max gain: {max_gain if max_gain is not None else 'Unlimited'}")
                print(f"    Max loss: {max_loss if max_loss is not None else 'Unlimited'}")


def parse_symbols() -> List[str]:
    """Determine the list of symbols to scan.

    If the environment variable `SCAN_ALL` is set to "true" (case‑insensitive),
    the function returns the full universe from `fetch_all_symbols()`.  Otherwise
    it parses `SYMBOL_LIST` (a comma‑separated list).  If `SYMBOL_LIST` is not
    provided, it defaults to the 30 Dow Jones Industrial Average constituents.

    When `USE_YFINANCE` is the only data source, scanning the full universe
    can be slow.  Consider specifying a shorter `SYMBOL_LIST` to reduce
    runtime.  This helper hides these details from callers.
    """
    scan_all = os.getenv('SCAN_ALL', 'false').lower() in {'1', 'true', 'yes'}
    if scan_all:
        syms = fetch_all_symbols()
        if syms:
            return syms
    env = os.getenv('SYMBOL_LIST')
    if env:
        return [s.strip().upper() for s in env.split(',') if s.strip()]
    # Default to the Dow 30 constituents
    return [
        'AAPL', 'MSFT', 'IBM', 'JPM', 'XOM', 'V', 'JNJ', 'WMT', 'PG', 'UNH',
        'KO', 'HD', 'INTC', 'CRM', 'MCD', 'CAT', 'HON', 'GS', 'DIS', 'VZ',
        'NKE', 'TRV', 'AXP', 'BA', 'MRK', 'CSCO', 'CVX', 'DOW', 'AMGN', 'MMM',
    ]


# ============================================================================
# Advanced scanning helpers (yfinance‑only implementation)
#
# The functions below implement a simplified version of the "Best‑of‑Best"
# scanning process described in the outline.  Without access to real‑time news
# and dual‑feed market data, these helpers rely on the free `yfinance` data
# source to estimate volume, momentum and volatility.  They assign a
# rudimentary confidence score to each symbol and build candidate spreads
# accordingly.  These features do not guarantee profitability and are
# illustrative only.
# ==========================================================================

def compute_intraday_metrics(symbol: str) -> Optional[Dict[str, float]]:
    """Compute simple intraday metrics for a symbol using `yfinance`.

    This function fetches 1‑minute bars for the current trading day (if
    available) and returns a dictionary with:

    - 'price_change': percentage change from previous close to the latest
      bar (as a float between -1 and 1).
    - 'rvol': relative volume (volume today vs average volume over the past
      five trading days).  Values >1 indicate higher‑than‑normal activity.
    - 'iv': average implied volatility of near‑term options (if available).

    Returns None if data cannot be retrieved or computed.
    """
    if not _HAS_YFINANCE:
        return None
    try:
        ticker = yf.Ticker(symbol)
        # Daily history for last 7 days to compute average volume
        hist_daily = ticker.history(period="7d", interval="1d")
        if hist_daily.empty or len(hist_daily) < 2:
            return None
        prev_close = float(hist_daily['Close'].iloc[-2])
        avg_vol = float(hist_daily['Volume'].iloc[-6:-1].mean()) if len(hist_daily) >= 6 else float(hist_daily['Volume'].mean())
        # Intraday 1‑minute bars for today
        hist_min = ticker.history(period="1d", interval="1m")
        if hist_min.empty:
            return None
        last_row = hist_min.iloc[-1]
        latest_close = float(last_row['Close'])
        cum_vol = float(hist_min['Volume'].sum())
        price_change = (latest_close - prev_close) / prev_close
        rvol = cum_vol / avg_vol if avg_vol else 0.0
        # Fetch options chain to estimate an implied volatility average
        iv = None
        if use_yfinance := (os.getenv('USE_YFINANCE', 'false').lower() in {'1', 'true', 'yes'}):
            try:
                opts = ticker.options
                if opts:
                    # Use nearest expiration to estimate IV
                    exp = opts[0]
                    chain = ticker.option_chain(exp)
                    vols = []
                    for df in [chain.calls, chain.puts]:
                        # filter out invalid values
                        vols.extend([iv for iv in df.get('impliedVolatility', []) if iv == iv])
                    if vols:
                        iv = sum(vols) / len(vols)
            except Exception:
                pass
        return {
            'price_change': price_change,
            'rvol': rvol,
            'iv': iv or 0.0,
        }
    except Exception:
        return None


def compute_confidence(metrics: Dict[str, float]) -> float:
    """Compute a rudimentary confidence score (0–100) from intraday metrics.

    The score weighs momentum (price change), relative volume (RVOL) and
    implied volatility (IV).  A higher RVOL and positive price change increase
    the score; extremely high IV reduces the score to reflect heightened
    uncertainty.  This heuristic is simplistic and intended as a stand‑in
    for more sophisticated signal blending.
    """
    score = 0.0
    # Momentum component: scale 50 points across ±5% price change
    momentum = max(min(metrics.get('price_change', 0.0), 0.05), -0.05)
    score += (momentum / 0.05) * 30.0  # ±30 pts
    # Relative volume: up to 40 points at rvol ≥ 5
    rvol = metrics.get('rvol', 1.0)
    rvol_points = min(rvol / 5.0, 1.0) * 40.0
    score += rvol_points
    # Implied volatility penalty: subtract up to 20 points when IV > 0.6
    iv = metrics.get('iv', 0.0)
    if iv > 0.6:
        penalty = min((iv - 0.6) / 0.4, 1.0) * 20.0
        score -= penalty
    # Clip to 0–100
    return max(0.0, min(100.0, 50.0 + score))


def pre_market_scan(symbols: List[str], max_candidates: int = 50) -> List[Tuple[str, Strategy, float]]:
    """Perform a simplified pre‑market scan and return top candidates.

    For each symbol, this function computes intraday metrics, derives a
    confidence score, and generates a candidate vertical spread using
    `identify_candidate_trades`.  It returns a list of tuples
    (description, strategy, confidence) sorted by descending confidence.

    The number of returned candidates is capped by `max_candidates`.  When
    metrics cannot be computed or no trades are available, the symbol is
    skipped.
    """
    results: List[Tuple[str, Strategy, float]] = []
    for sym in symbols:
        metrics = compute_intraday_metrics(sym)
        if not metrics:
            continue
        conf = compute_confidence(metrics)
        trades = identify_candidate_trades(sym)
        if not trades:
            continue
        # Use the first trade idea as a representative; additional ideas
        # could be considered in a more exhaustive implementation.
        desc, strat = trades[0]
        results.append((desc, strat, conf))
    # Sort by confidence descending and truncate
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:max_candidates]


def print_go_cards(candidates: List[Tuple[str, Strategy, float]]) -> None:
    """Pretty‑print the selected trade ideas as "GO" cards.

    Each candidate is displayed with its description, confidence score,
    estimated max gain/loss and risk level.  In a more feature‑rich system
    this function could write to a spreadsheet or broker ticket template.
    """
    print("\n===== GO CARDS =====")
    for idx, (desc, strat, conf) in enumerate(candidates, start=1):
        mg = strat.max_gain()
        ml = strat.max_loss()
        print(f"{idx}. {desc}")
        print(f"   Confidence: {conf:.1f}/100  Risk Level: {strat.risk_level()}")
        print(f"   Max Gain: {mg if mg is not None else 'Unlimited'}  Max Loss: {ml if ml is not None else 'Unlimited'}")
    print("====================\n")


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
    # Load historical trades and initialise the learner
    learner = ReinforcementLearner()
    trade_log = load_trade_records()
    # Learn from previous records
    if trade_log:
        learn_from_records(learner, trade_log)
    # Immediately run one pre‑market scan on start using intraday metrics.
    print(f"Initial pre‑market scan at { _dt.datetime.now().strftime('%Y-%m-%d %H:%M') }")
    try:
        candidates = pre_market_scan(symbols)
        if candidates:
            print_go_cards(candidates)
    except Exception as exc:
        print(f"Pre‑market scan failed: {exc}")
    # Run a conventional scan to update RL and trade log
    scan_once(symbols, learner, trade_log)
    save_trade_records(trade_log)
    while True:
        wait_until_next_scan(scan_times)
        print(f"\nRunning scheduled scan at { _dt.datetime.now().strftime('%Y-%m-%d %H:%M') }")
        scan_once(symbols, learner, trade_log)
        save_trade_records(trade_log)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Scanner terminated by user.")