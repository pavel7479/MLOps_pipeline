"""All-in LONG/FLAT portfolio accounting."""

from dataclasses import dataclass

from .models import BacktestParameters


@dataclass(frozen=True)
class BuyExecution:
    execution_price: float
    btc_quantity: float
    notional: float
    fee: float
    total_cost: float


@dataclass(frozen=True)
class SellExecution:
    execution_price: float
    btc_quantity: float
    gross_proceeds: float
    fee: float
    net_proceeds: float


class Portfolio:
    """Hold either all capital in quote cash or all capital in BTC."""

    def __init__(self, parameters: BacktestParameters) -> None:
        parameters.validate()
        self.parameters = parameters
        self.cash = float(parameters.initial_cash)
        self.btc_quantity = 0.0

    @property
    def position(self) -> str:
        return "LONG" if self.btc_quantity > 0 else "FLAT"

    def buy(self, market_price: float) -> BuyExecution:
        if self.position != "FLAT":
            raise ValueError("Cannot buy while already LONG")
        if market_price <= 0:
            raise ValueError("Buy price must be positive")
        execution_price = float(market_price) * (1.0 + self.parameters.slippage_rate)
        available_cash = self.cash
        notional = available_cash / (1.0 + self.parameters.commission_rate)
        fee = notional * self.parameters.commission_rate
        quantity = notional / execution_price
        total_cost = notional + fee
        self.cash = max(0.0, available_cash - total_cost)
        self.btc_quantity = quantity
        return BuyExecution(execution_price, quantity, notional, fee, total_cost)

    def sell(self, market_price: float) -> SellExecution:
        if self.position != "LONG":
            raise ValueError("Cannot sell while FLAT")
        if market_price <= 0:
            raise ValueError("Sell price must be positive")
        execution_price = float(market_price) * (1.0 - self.parameters.slippage_rate)
        quantity = self.btc_quantity
        gross_proceeds = quantity * execution_price
        fee = gross_proceeds * self.parameters.commission_rate
        net_proceeds = gross_proceeds - fee
        self.cash += net_proceeds
        self.btc_quantity = 0.0
        return SellExecution(execution_price, quantity, gross_proceeds, fee, net_proceeds)

    def equity(self, mark_price: float) -> float:
        if mark_price <= 0:
            raise ValueError("Mark price must be positive")
        return float(self.cash + self.btc_quantity * float(mark_price))
