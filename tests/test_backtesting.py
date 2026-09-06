import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    ArtifactPredictor, BacktestEngine, BacktestParameters, BacktestingPipeline, Portfolio,
    maximum_drawdown_pct, win_rate,
)
from src.config import load_settings
from src.config.settings import BacktestingSettings, BacktestStrategySettings, MLDatasetSettings


def market_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 10:00", periods=rows, freq="h", tz="UTC"),
        "open": [100.0 + 10 * index for index in range(rows)],
        "close": [105.0 + 10 * index for index in range(rows)],
    })


def parameters(commission: float = 0.0) -> BacktestParameters:
    return BacktestParameters(
        initial_cash=1000.0,
        commission_rate=commission,
        slippage_rate=0.0,
        allow_short=False,
        leverage=1.0,
        force_close_at_end=True,
    )


def test_portfolio_buy_uses_cash_without_going_negative() -> None:
    portfolio = Portfolio(parameters(commission=0.01))
    execution = portfolio.buy(100.0)
    expected_notional = 1000.0 / 1.01
    assert execution.notional == pytest.approx(expected_notional)
    assert execution.fee == pytest.approx(expected_notional * 0.01)
    assert execution.btc_quantity == pytest.approx(expected_notional / 100.0)
    assert portfolio.cash == pytest.approx(0.0, abs=1e-12)
    assert portfolio.position == "LONG"


def test_portfolio_sale_accounts_for_fee_and_profit() -> None:
    portfolio = Portfolio(parameters(commission=0.01))
    buy = portfolio.buy(100.0)
    sell = portfolio.sell(120.0)
    expected_gross = buy.btc_quantity * 120.0
    assert sell.gross_proceeds == pytest.approx(expected_gross)
    assert sell.fee == pytest.approx(expected_gross * 0.01)
    assert portfolio.cash == pytest.approx(expected_gross * 0.99)
    assert portfolio.position == "FLAT"


def test_buy_hold_sell_creates_one_completed_trade_at_next_opens() -> None:
    engine = BacktestEngine(parameters(), pd.Timedelta(1, unit="h"))
    result = engine.run(
        market_frame(), ["BUY", "HOLD", "SELL", "HOLD"], "test", "Test"
    )
    assert result.buy_executions == 1
    assert result.sell_executions == 1
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_signal_time"] == pd.Timestamp("2025-01-01 10:00", tz="UTC")
    assert trade["entry_time"] == pd.Timestamp("2025-01-01 11:00", tz="UTC")
    assert trade["entry_price"] == pytest.approx(110.0)
    assert trade["exit_signal_time"] == pd.Timestamp("2025-01-01 12:00", tz="UTC")
    assert trade["exit_time"] == pd.Timestamp("2025-01-01 13:00", tz="UTC")
    assert trade["exit_price"] == pytest.approx(130.0)
    assert trade["holding_hours"] == pytest.approx(2.0)
    assert trade["exit_reason"] == "sell_signal"
    expected_quantity = 1000.0 / 110.0
    assert trade["btc_quantity"] == pytest.approx(expected_quantity)
    assert trade["gross_pnl"] == pytest.approx(expected_quantity * 20.0)
    assert trade["net_pnl"] == pytest.approx(expected_quantity * 20.0)
    assert trade["return_pct"] == pytest.approx(expected_quantity * 20.0 / 1000.0 * 100.0)


def test_buy_and_hold_uses_first_open_and_last_close() -> None:
    result = BacktestEngine(parameters(), pd.Timedelta(1, unit="h")).run_buy_and_hold(
        market_frame(3)
    )
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(125.0)
    assert trade["exit_reason"] == "end_of_backtest"
    assert result.buy_executions == 1 and result.sell_executions == 1


def test_slippage_worsens_buy_and_sell_execution_prices() -> None:
    configured = replace(parameters(), slippage_rate=0.01)
    result = BacktestEngine(configured, pd.Timedelta(1, unit="h")).run(
        market_frame(2), ["BUY", "HOLD"], "test", "Test"
    )
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(110.0 * 1.01)
    assert trade["exit_price"] == pytest.approx(115.0 * 0.99)




def test_repeated_buy_does_not_stack_positions() -> None:
    result = BacktestEngine(parameters(), pd.Timedelta(1, unit="h")).run(
        market_frame(), ["BUY", "BUY", "BUY", "HOLD"], "test", "Test"
    )
    assert result.buy_executions == 1
    assert result.sell_executions == 1
    assert len(result.trades) == 1


def test_sell_while_flat_does_not_short_or_change_cash() -> None:
    result = BacktestEngine(parameters(), pd.Timedelta(1, unit="h")).run(
        market_frame(3), ["SELL", "SELL", "HOLD"], "test", "Test"
    )
    assert result.buy_executions == 0
    assert result.sell_executions == 0
    assert result.trades.empty
    assert result.metrics["final_equity"] == pytest.approx(1000.0)
    assert (result.equity_curve["position"] == "FLAT").all()


def test_signal_executes_at_next_open_not_current_close() -> None:
    market = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 10:00", periods=2, freq="h", tz="UTC"),
        "open": [99.0, 110.0],
        "close": [100.0, 120.0],
    })
    result = BacktestEngine(parameters(), pd.Timedelta(1, unit="h")).run(
        market, ["BUY", "HOLD"], "test", "Test"
    )
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(110.0)
    assert result.trades.iloc[0]["entry_price"] != 100.0


def test_last_prediction_is_not_executed() -> None:
    result = BacktestEngine(parameters(), pd.Timedelta(1, unit="h")).run(
        market_frame(2), ["HOLD", "BUY"], "test", "Test"
    )
    assert result.buy_executions == 0
    assert result.sell_executions == 0
    assert result.trades.empty


def test_force_close_uses_last_close_and_records_reason_and_fee() -> None:
    result = BacktestEngine(parameters(commission=0.01), pd.Timedelta(1, unit="h")).run(
        market_frame(2), ["BUY", "HOLD"], "test", "Test"
    )
    trade = result.trades.iloc[0]
    assert trade["exit_price"] == pytest.approx(115.0)
    assert trade["exit_reason"] == "end_of_backtest"
    assert pd.isna(trade["exit_signal_time"])
    assert trade["exit_fee"] > 0
    assert result.equity_curve.iloc[-1]["position"] == "FLAT"


def test_commission_reduces_equity_for_same_operations() -> None:
    signals = ["BUY", "SELL", "HOLD"]
    without_fee = BacktestEngine(parameters(0.0), pd.Timedelta(1, unit="h")).run(
        market_frame(3), signals, "free", "Free"
    )
    with_fee = BacktestEngine(parameters(0.01), pd.Timedelta(1, unit="h")).run(
        market_frame(3), signals, "fee", "Fee"
    )
    assert with_fee.metrics["final_equity"] < without_fee.metrics["final_equity"]
    assert with_fee.metrics["total_fees"] > 0


def test_drawdown_and_win_rate_match_manual_examples() -> None:
    assert maximum_drawdown_pct([100.0, 120.0, 90.0, 110.0]) == pytest.approx(-25.0)
    trades = pd.DataFrame({"net_pnl": [10.0, 2.0, -1.0]})
    assert win_rate(trades) == pytest.approx(2 / 3)


def test_engine_does_not_need_target_or_future_return() -> None:
    market = market_frame()
    assert "target" not in market and "future_return" not in market
    result = BacktestEngine(parameters(), pd.Timedelta(1, unit="h")).run(
        market, ["BUY", "HOLD", "SELL", "HOLD"], "test", "Test"
    )
    assert result.metrics["number_of_trades"] == 1


class FakeArtifactModel:
    name = "fake_model"
    parameters = {"source": "test"}
    n_features_in_ = 2
    feature_names_in_ = np.asarray(["f1", "f2"])

    def __init__(self, predictions: list[str]) -> None:
        self.predictions = np.asarray(predictions)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == ["f1", "f2"]
        return self.predictions[:len(features)]


def test_predictor_rejects_feature_manifest_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    joblib.dump(FakeArtifactModel(["HOLD"]), model_path)
    with pytest.raises(ValueError, match="expects 2 features"):
        ArtifactPredictor().predict(
            model_path, pd.DataFrame({"f1": [1.0]}), ["f1"]
        )



def test_pipeline_never_reads_test_and_saves_complete_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()
    validation = market_frame()
    validation["f1"] = [1.0, 2.0, 3.0, 4.0]
    validation["f2"] = [4.0, 3.0, 2.0, 1.0]
    validation.to_parquet(ml_dir / "BTCUSDT_1h_validation.parquet", index=False)
    validation.to_parquet(ml_dir / "BTCUSDT_1h_test.parquet", index=False)
    (ml_dir / "feature_manifest.json").write_text(
        json.dumps({"features": ["f1", "f2"]}), encoding="utf-8"
    )
    first_model = tmp_path / "first.joblib"
    second_model = tmp_path / "second.joblib"
    joblib.dump(FakeArtifactModel(["BUY", "HOLD", "SELL", "HOLD"]), first_model)
    joblib.dump(FakeArtifactModel(["SELL", "BUY", "HOLD", "SELL"]), second_model)

    base = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    settings = replace(
        base,
        ml_dataset=MLDatasetSettings(ml_dir),
        backtesting=BacktestingSettings(
            initial_cash=1000.0,
            commission_rate=0.01,
            slippage_rate=0.0,
            allow_short=False,
            leverage=1.0,
            force_close_at_end=True,
            output_dir=tmp_path / "backtesting",
            strategies=(
                BacktestStrategySettings("first", "First", first_model),
                BacktestStrategySettings("second", "Second", second_model),
            ),
        ),
    )
    original_read = pd.read_parquet
    reads: list[str] = []

    def guarded_read(path, *args, **kwargs):
        name = Path(path).name
        assert "_test.parquet" not in name
        reads.append(name)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    report = BacktestingPipeline(settings).run()

    assert reads == ["BTCUSDT_1h_validation.parquet"]
    assert len(report.results) == 3
    assert all(path.is_file() for path in report.artifact_paths.values())
    metadata = json.loads(report.artifact_paths["backtest_metadata"].read_text(encoding="utf-8"))
    assert metadata["test_dataset_used"] is False
    assert metadata["execution_rule"] == "prediction_after_close_t_executes_at_open_t_plus_1"
    assert metadata["features"] == ["f1", "f2"]
    assert (tmp_path / "backtesting" / "equity_curve_first.csv").is_file()
    assert (tmp_path / "backtesting" / "trades_second.csv").is_file()
