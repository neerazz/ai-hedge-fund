import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    has_valid_signals = False

    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        valid_signals = 0
        total_signals = 0

        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                signal = signals[ticker].get("signal")
                confidence = signals[ticker].get("confidence", 0)
                
                # Only count signals with real confidence values
                if signal and confidence > 0:
                    valid_signals += 1
                total_signals += 1
                
                ticker_signals[agent] = {
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": signals[ticker].get("reasoning", "")
                }

        # Track if we have enough valid signals
        if valid_signals > 0:
            has_valid_signals = True

        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_management_agent", None, "Making trading decisions")

    # Generate trading decisions only if we have valid signals
    if has_valid_signals:
        result = generate_trading_decision(
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            max_shares=max_shares,
            portfolio=portfolio,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
    else:
        # Create a default hold decision with explanation
        result = PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=0.0,
                    reasoning="Insufficient valid signals from analysts to make trading decisions"
                ) for ticker in tickers
            }
        )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_management_agent"  # Fixed from "portfolio_management" to match other agents
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Management Agent")

    # Update the analyst signals in the state with the portfolio decisions
    state["data"]["analyst_signals"]["portfolio_management_agent"] = {
        ticker: decision.model_dump() for ticker, decision in result.decisions.items()
    }

    progress.update_status("portfolio_management_agent", None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""
    
    # Pre-process signals to aggregate confidence levels and determine overall sentiment
    aggregated_signals = {}
    for ticker in tickers:
        ticker_signals = signals_by_ticker.get(ticker, {})
        
        if not ticker_signals:
            continue
            
        bullish_confidence = 0.0
        bearish_confidence = 0.0
        total_valid_signals = 0
        
        for agent, signal_data in ticker_signals.items():
            signal = signal_data.get("signal", "").lower()
            confidence = signal_data.get("confidence", 0.0)
            
            if signal and confidence > 0:
                if signal == "bullish":
                    bullish_confidence += confidence
                elif signal == "bearish":
                    bearish_confidence += confidence
                total_valid_signals += 1
        
        if total_valid_signals > 0:
            avg_bullish = bullish_confidence / total_valid_signals
            avg_bearish = bearish_confidence / total_valid_signals
            
            # Determine dominant sentiment and overall confidence
            if avg_bullish > avg_bearish and avg_bullish > 30:  # Minimum confidence threshold
                dominant_signal = "bullish"
                overall_confidence = avg_bullish
            elif avg_bearish > avg_bullish and avg_bearish > 30:
                dominant_signal = "bearish"
                overall_confidence = avg_bearish
            else:
                dominant_signal = "neutral"
                overall_confidence = max(avg_bullish, avg_bearish)
                
            aggregated_signals[ticker] = {
                "signal": dominant_signal,
                "confidence": overall_confidence,
                "valid_signals": total_valid_signals
            }

    # Create an enhanced prompt template that includes aggregated signal information
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a portfolio manager making final trading decisions based on multiple tickers.

            Trading Rules:
            - For long positions:
              * Only buy if you have available cash
              * Only sell if you currently hold long shares of that ticker
              * Sell quantity must be ≤ current long position shares
              * Buy quantity must be ≤ max_shares for that ticker
            
            - For short positions:
              * Only short if you have available margin (position value × margin requirement)
              * Only cover if you currently have short shares of that ticker
              * Cover quantity must be ≤ current short position shares
              * Short quantity must respect margin requirements
            
            - The max_shares values are pre-calculated to respect position limits
            - Consider both long and short opportunities based on signals
            - Maintain appropriate risk management with both long and short exposure
            - For signals with low confidence (< 50), prefer holding current positions
            - Require stronger conviction (> 60) for opening new positions vs adjusting existing ones

            Available Actions:
            - "buy": Open or add to long position (requires strong bullish signals)
            - "sell": Close or reduce long position (requires bearish signals or risk management)
            - "short": Open or add to short position (requires strong bearish signals)
            - "cover": Close or reduce short position (requires bullish signals or risk management)
            - "hold": No action (default when conviction is low)

            Decision Making Guidelines:
            1. Prioritize high-confidence signals (> 70) from multiple analysts
            2. Consider current positions when sizing trades
            3. Be more conservative with new positions vs managing existing ones
            4. Default to hold when signals are mixed or low confidence
            """,
        ),
        (
            "human",
            """Make trading decisions based on the following information:

            Aggregated Analyst Signals:
            {aggregated_signals}

            Raw Signals by Ticker:
            {signals_by_ticker}

            Current Prices:
            {current_prices}

            Maximum Shares Allowed:
            {max_shares}

            Portfolio Cash: {portfolio_cash}
            Current Positions: {portfolio_positions}
            Margin Requirement: {margin_requirement}
            Total Margin Used: {total_margin_used}

            Output strictly in JSON with the following structure:
            {{
                "decisions": {{
                    "TICKER1": {{
                        "action": "buy/sell/short/cover/hold",
                        "quantity": integer,
                        "confidence": float between 0 and 100,
                        "reasoning": "string"
                    }},
                    "TICKER2": {{
                        ...
                    }},
                    ...
                }}
            }}
            """,
        ),
    ])

    # Generate the prompt with enhanced context
    prompt = template.invoke({
        "aggregated_signals": json.dumps(aggregated_signals, indent=2),
        "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
        "current_prices": json.dumps(current_prices, indent=2),
        "max_shares": json.dumps(max_shares, indent=2),
        "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
        "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
        "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
        "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",
    })

    def create_default_portfolio_output():
        decisions = {}
        for ticker in tickers:
            # Create more informative default decisions based on aggregated signals
            agg_signal = aggregated_signals.get(ticker, {})
            if agg_signal:
                reasoning = f"Default hold due to {agg_signal.get('signal', 'neutral')} signal with {agg_signal.get('confidence', 0):.1f}% confidence from {agg_signal.get('valid_signals', 0)} valid signals"
            else:
                reasoning = "Default hold due to insufficient analyst signals"
                
            decisions[ticker] = PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=agg_signal.get('confidence', 0.0) if agg_signal else 0.0,
                reasoning=reasoning
            )
        return PortfolioManagerOutput(decisions=decisions)

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_management_agent",
        default_factory=create_default_portfolio_output
    )
