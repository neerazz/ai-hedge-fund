from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json

from tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades with retry logic
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from insider trades with improved error handling
        insider_signals = []
        if insider_trades:
            transaction_shares = pd.Series([t.transaction_shares for t in insider_trades if t.transaction_shares is not None])
            if not transaction_shares.empty:
                insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news with improved retry logic
        company_news = get_company_news(
            ticker=ticker,
            end_date=end_date,
            limit=100,
            max_retries=6,  # Increased retries
            wait_time=60    # Increased wait time
        )

        # Get sentiment from news with fallback logic
        news_signals = []
        if company_news:
            sentiment = pd.Series([n.sentiment for n in company_news if n.sentiment is not None])
            if not sentiment.empty:
                news_signals = np.where(sentiment == "negative", "bearish",
                                      np.where(sentiment == "positive", "bullish", "neutral")).tolist()

        progress.update_status("sentiment_agent", ticker, "Combining signals")

        # Dynamically adjust weights based on available data
        if len(insider_signals) > 0 and len(news_signals) > 0:
            # Both data sources available - use standard weights
            insider_weight = 0.3
            news_weight = 0.7
        elif len(insider_signals) > 0:
            # Only insider data available
            insider_weight = 1.0
            news_weight = 0.0
        elif len(news_signals) > 0:
            # Only news data available
            insider_weight = 0.0
            news_weight = 1.0
        else:
            # No data available - default to neutral with low confidence
            sentiment_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": "Insufficient data: no valid insider trades or news sentiment available"
            }
            continue

        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )
        neutral_signals = news_signals.count("neutral") * news_weight

        # Calculate total weighted signals for confidence
        total_weighted_signals = (
            (len(insider_signals) * insider_weight if len(insider_signals) > 0 else 0) +
            (len(news_signals) * news_weight if len(news_signals) > 0 else 0)
        )

        # Determine overall signal
        if bullish_signals > bearish_signals and bullish_signals > neutral_signals:
            overall_signal = "bullish"
            signal_strength = bullish_signals
        elif bearish_signals > bullish_signals and bearish_signals > neutral_signals:
            overall_signal = "bearish"
            signal_strength = bearish_signals
        else:
            overall_signal = "neutral"
            signal_strength = max(bullish_signals, bearish_signals, neutral_signals)

        # Calculate confidence level
        confidence = 0 if total_weighted_signals == 0 else round((signal_strength / total_weighted_signals) * 100, 2)

        # Generate detailed reasoning
        reasoning_parts = []
        if insider_signals:
            insider_summary = f"Insider sentiment: {insider_signals.count('bullish')} bullish vs {insider_signals.count('bearish')} bearish trades"
            reasoning_parts.append(insider_summary)
        if news_signals:
            news_summary = f"News sentiment: {news_signals.count('bullish')} bullish, {news_signals.count('bearish')} bearish, {news_signals.count('neutral')} neutral articles"
            reasoning_parts.append(news_summary)
        
        reasoning = "; ".join(reasoning_parts)

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    return {
        "messages": [message],
        "data": data,
    }
