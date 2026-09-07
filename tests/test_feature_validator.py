from datetime import datetime, timezone
import math
import random

import pytest

from src.inference import FeatureValidationError, FeatureValidator

FEATURE_NAMES = (
    "return_1h", "return_3h", "return_6h", "return_12h", "return_24h",
    "close_sma_6", "close_to_sma_6", "close_sma_12", "close_to_sma_12",
    "close_sma_24", "close_to_sma_24", "volatility_6", "volatility_12",
    "volatility_24", "volume_change_1h", "volume_change_6h", "volume_mean_6",
    "volume_to_mean_6", "volume_mean_24", "volume_to_mean_24",
    "high_low_range", "body_size", "rsi_14", "macd", "macd_signal",
    "macd_histogram", "atr_14", "atr_relative",
)


def _features() -> dict[str, float]:
    return {name: float(index) / 100 for index, name in enumerate(FEATURE_NAMES)}


def _validator() -> FeatureValidator:
    return FeatureValidator(FEATURE_NAMES, "BTCUSDT", "1h")


def test_exact_28_features_are_accepted_and_ordered() -> None:
    values = list(_features().items())
    random.Random(42).shuffle(values)
    frame = _validator().validate_and_prepare(
        symbol="BTCUSDT",
        timeframe="1h",
        feature_timestamp=datetime(2026, 9, 7, 12, tzinfo=timezone.utc),
        features=dict(values),
    )
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert frame.shape == (1, 28)


def test_missing_feature_is_rejected() -> None:
    features = _features()
    del features["rsi_14"]
    with pytest.raises(FeatureValidationError, match="missing features.*rsi_14"):
        _validator().validate_and_prepare(
            symbol="BTCUSDT", timeframe="1h",
            feature_timestamp=datetime.now(timezone.utc), features=features,
        )


def test_future_return_extra_feature_is_rejected() -> None:
    features = {**_features(), "future_return": 0.1}
    with pytest.raises(FeatureValidationError, match="unexpected features.*future_return"):
        _validator().validate_and_prepare(
            symbol="BTCUSDT", timeframe="1h",
            feature_timestamp=datetime.now(timezone.utc), features=features,
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_feature_is_rejected(invalid: float) -> None:
    features = _features()
    features["return_1h"] = invalid
    with pytest.raises(FeatureValidationError, match="must be finite"):
        _validator().validate_and_prepare(
            symbol="BTCUSDT", timeframe="1h",
            feature_timestamp=datetime.now(timezone.utc), features=features,
        )


def test_string_and_boolean_features_are_rejected() -> None:
    for invalid in ("0.1", True):
        features = _features()
        features["return_1h"] = invalid  # type: ignore[assignment]
        with pytest.raises(FeatureValidationError, match="must be numeric"):
            _validator().validate_and_prepare(
                symbol="BTCUSDT", timeframe="1h",
                feature_timestamp=datetime.now(timezone.utc), features=features,
            )


def test_wrong_symbol_and_timeframe_are_rejected() -> None:
    with pytest.raises(FeatureValidationError, match="symbol must be BTCUSDT"):
        _validator().validate_and_prepare(
            symbol="ETHUSDT", timeframe="1h",
            feature_timestamp=datetime.now(timezone.utc), features=_features(),
        )
    with pytest.raises(FeatureValidationError, match="timeframe must be 1h"):
        _validator().validate_and_prepare(
            symbol="BTCUSDT", timeframe="5m",
            feature_timestamp=datetime.now(timezone.utc), features=_features(),
        )


def test_timezone_is_required() -> None:
    with pytest.raises(FeatureValidationError, match="must include timezone"):
        _validator().validate_and_prepare(
            symbol="BTCUSDT", timeframe="1h",
            feature_timestamp=datetime(2026, 9, 7, 12), features=_features(),
        )


def test_fingerprint_is_independent_of_json_feature_order() -> None:
    timestamp = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    first = _features()
    second = dict(reversed(list(first.items())))
    assert FeatureValidator.fingerprint(
        symbol="BTCUSDT", timeframe="1h",
        feature_timestamp=timestamp, features=first,
    ) == FeatureValidator.fingerprint(
        symbol="BTCUSDT", timeframe="1h",
        feature_timestamp=timestamp, features=second,
    )
