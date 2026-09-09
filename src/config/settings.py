"""Typed application configuration loaded from one YAML file."""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class MarketDataSettings:
    provider: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime | None


@dataclass(frozen=True)
class StorageSettings:
    raw_dir: Path
    processed_dir: Path
    format: str


@dataclass(frozen=True)
class HttpSettings:
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float


@dataclass(frozen=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True)
class FeaturesSettings:
    return_periods: tuple[int, ...] = (1, 3, 6, 12, 24)
    rolling_mean_windows: tuple[int, ...] = (6, 12, 24)
    rolling_std_windows: tuple[int, ...] = (6, 12, 24)
    volume_change_periods: tuple[int, ...] = (1, 6)
    volume_rolling_windows: tuple[int, ...] = (6, 24)
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    atr_period: int = 14


@dataclass(frozen=True)
class TargetSettings:
    horizon_hours: int = 3
    buy_threshold: float = 0.003
    sell_threshold: float = -0.003
    rare_class_warning_threshold: float = 0.05


@dataclass(frozen=True)
class SplitSettings:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(frozen=True)
class MLDatasetSettings:
    output_dir: Path = Path("data/ml")


@dataclass(frozen=True)
class DummySettings:
    strategy: str = "most_frequent"


@dataclass(frozen=True)
class LogisticRegressionSettings:
    max_iter: int = 2000
    random_state: int = 42
    class_weight: str | None = None


@dataclass(frozen=True)
class CatBoostSettings:
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.05
    random_seed: int = 42
    verbose: bool = False


@dataclass(frozen=True)
class XGBoostSettings:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42


@dataclass(frozen=True)
class LightGBMSettings:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42


@dataclass(frozen=True)
class ModelsSettings:
    dummy: DummySettings = DummySettings()
    logistic_regression: LogisticRegressionSettings = LogisticRegressionSettings()
    catboost: CatBoostSettings = CatBoostSettings()
    xgboost: XGBoostSettings = XGBoostSettings()
    lightgbm: LightGBMSettings = LightGBMSettings()


@dataclass(frozen=True)
class EvaluationSettings:
    primary_metric: str = "macro_f1"
    models_dir: Path = Path("artifacts/models")
    results_dir: Path = Path("artifacts/model_evaluation")


@dataclass(frozen=True)
class TimeSeriesValidationSettings:
    n_splits: int = 5


@dataclass(frozen=True)
class HyperparameterSearchSettings:
    n_iter: int = 20
    random_state: int = 42
    scoring: str = "macro_f1"
    models_dir: Path = Path("artifacts/tuned_models")
    results_dir: Path = Path("artifacts/hyperparameter_search")


@dataclass(frozen=True)
class CatBoostSearchSpace:
    iterations: tuple[int, ...] = (200, 300, 500, 700)
    depth: tuple[int, ...] = (4, 5, 6, 7, 8)
    learning_rate: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1)
    l2_leaf_reg: tuple[int, ...] = (1, 3, 5, 7, 10)


@dataclass(frozen=True)
class XGBoostSearchSpace:
    n_estimators: tuple[int, ...] = (200, 300, 500, 700)
    max_depth: tuple[int, ...] = (3, 4, 5, 6, 8)
    learning_rate: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1)
    subsample: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    colsample_bytree: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    min_child_weight: tuple[int, ...] = (1, 3, 5, 7)


@dataclass(frozen=True)
class LightGBMSearchSpace:
    n_estimators: tuple[int, ...] = (200, 300, 500, 700)
    max_depth: tuple[int, ...] = (-1, 4, 6, 8, 10)
    learning_rate: tuple[float, ...] = (0.01, 0.03, 0.05, 0.1)
    num_leaves: tuple[int, ...] = (15, 31, 63, 127)
    subsample: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    colsample_bytree: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0)
    min_child_samples: tuple[int, ...] = (10, 20, 30, 50)


@dataclass(frozen=True)
class SearchSpacesSettings:
    catboost: CatBoostSearchSpace = CatBoostSearchSpace()
    xgboost: XGBoostSearchSpace = XGBoostSearchSpace()
    lightgbm: LightGBMSearchSpace = LightGBMSearchSpace()

@dataclass(frozen=True)
class BacktestStrategySettings:
    slug: str
    display_name: str
    model_path: Path
    model_version: str = "unspecified"


@dataclass(frozen=True)
class BacktestingSettings:
    initial_cash: float = 10000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0
    allow_short: bool = False
    leverage: float = 1.0
    force_close_at_end: bool = True
    output_dir: Path = Path("artifacts/backtesting")
    strategies: tuple[BacktestStrategySettings, ...] = ()


@dataclass(frozen=True)
class MLflowSettings:
    """Local MLflow tracking and registry configuration."""

    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "crypto_direction_classification"
    registered_model_name: str = "crypto_direction_classifier"
    artifact_location: str | None = None
    enabled: bool = False


@dataclass(frozen=True)
class APISettings:
    host: str = "127.0.0.1"
    port: int = 8000
    title: str = "Crypto ML Inference API"
    version: str = "1.0.0"


@dataclass(frozen=True)
class InferenceSettings:
    registered_model_name: str = "crypto_direction_classifier"
    model_alias: str = "champion"
    expected_symbol: str = "BTCUSDT"
    expected_timeframe: str = "1h"


@dataclass(frozen=True)
class DatabaseSettings:
    url_env_variable: str = "DATABASE_URL"
    pool_pre_ping: bool = True


@dataclass(frozen=True)
class MonitoringSettings:
    """Local Prometheus worker and drift thresholds."""

    reference_path: Path = Path("monitoring/reference/BTCUSDT_1h_train_reference.json")
    window_size: int = 500
    min_samples: int = 100
    poll_interval_seconds: float = 60.0
    metrics_host: str = "0.0.0.0"
    metrics_port: int = 9101
    warning_threshold: float = 0.10
    critical_threshold: float = 0.25




@dataclass(frozen=True)
class AppSettings:
    market_data: MarketDataSettings
    storage: StorageSettings
    http: HttpSettings
    logging: LoggingSettings
    features: FeaturesSettings = FeaturesSettings()
    target: TargetSettings = TargetSettings()
    split: SplitSettings = SplitSettings()
    ml_dataset: MLDatasetSettings = MLDatasetSettings()
    models: ModelsSettings = ModelsSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    time_series_validation: TimeSeriesValidationSettings = TimeSeriesValidationSettings()
    hyperparameter_search: HyperparameterSearchSettings = HyperparameterSearchSettings()
    search_spaces: SearchSpacesSettings = SearchSpacesSettings()
    backtesting: BacktestingSettings = BacktestingSettings()
    mlflow: MLflowSettings = MLflowSettings()
    api: APISettings = APISettings()
    inference: InferenceSettings = InferenceSettings()
    database: DatabaseSettings = DatabaseSettings()
    monitoring: MonitoringSettings = MonitoringSettings()


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigurationError(f"Missing required setting: {section}.{key}")
    return mapping[key]


def _utc_datetime(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationError(f"Invalid datetime for {name}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)



def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a list of positive integers") from exc
    if not result or any(item <= 0 for item in result):
        raise ConfigurationError(f"{name} must be a non-empty list of positive integers")
    return result


def _number_tuple(value: Any, name: str, cast: type = float) -> tuple[Any, ...]:
    try:
        result = tuple(cast(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a non-empty numeric list") from exc
    if not result:
        raise ConfigurationError(f"{name} must be a non-empty numeric list")
    return result


def load_settings(path: str | Path = "config.yaml") -> AppSettings:
    """Load, validate, and resolve application settings."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        market = raw["market_data"]
        storage = raw["storage"]
        http = raw["http"]
        logging_cfg = raw["logging"]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"Missing configuration section: {exc}") from exc
    base = config_path.parent
    start = _utc_datetime(_required(market, "start_date", "market_data"), "start_date")
    end_value = market.get("end_date")
    end = _utc_datetime(end_value, "end_date") if end_value is not None else None
    if end is not None and end <= start:
        raise ConfigurationError("market_data.end_date must be after start_date")
    fmt = str(_required(storage, "format", "storage")).lower()
    if fmt != "parquet":
        raise ConfigurationError("Only parquet storage is supported")
    retries = int(_required(http, "max_retries", "http"))
    if retries < 0:
        raise ConfigurationError("http.max_retries cannot be negative")
    feature_raw = raw.get("features", {})
    defaults = FeaturesSettings()
    returns_raw = feature_raw.get("returns", {})
    mean_raw = feature_raw.get("rolling_mean", {})
    std_raw = feature_raw.get("rolling_std", {})
    volume_raw = feature_raw.get("volume", {})
    rsi_raw = feature_raw.get("rsi", {})
    macd_raw = feature_raw.get("macd", {})
    atr_raw = feature_raw.get("atr", {})
    features = FeaturesSettings(
        return_periods=_positive_int_tuple(returns_raw.get("periods", defaults.return_periods), "features.returns.periods"),
        rolling_mean_windows=_positive_int_tuple(mean_raw.get("windows", defaults.rolling_mean_windows), "features.rolling_mean.windows"),
        rolling_std_windows=_positive_int_tuple(std_raw.get("windows", defaults.rolling_std_windows), "features.rolling_std.windows"),
        volume_change_periods=_positive_int_tuple(volume_raw.get("change_periods", defaults.volume_change_periods), "features.volume.change_periods"),
        volume_rolling_windows=_positive_int_tuple(volume_raw.get("rolling_windows", defaults.volume_rolling_windows), "features.volume.rolling_windows"),
        rsi_period=int(rsi_raw.get("period", defaults.rsi_period)),
        macd_fast_period=int(macd_raw.get("fast_period", defaults.macd_fast_period)),
        macd_slow_period=int(macd_raw.get("slow_period", defaults.macd_slow_period)),
        macd_signal_period=int(macd_raw.get("signal_period", defaults.macd_signal_period)),
        atr_period=int(atr_raw.get("period", defaults.atr_period)),
    )
    if min(features.rsi_period, features.macd_fast_period, features.macd_slow_period, features.macd_signal_period, features.atr_period) <= 0:
        raise ConfigurationError("Indicator periods must be positive")
    if features.macd_fast_period >= features.macd_slow_period:
        raise ConfigurationError("features.macd.fast_period must be below slow_period")
    target_raw = raw.get("target", {})
    target = TargetSettings(**{key: target_raw.get(key, getattr(TargetSettings(), key)) for key in TargetSettings.__dataclass_fields__})
    if target.horizon_hours <= 0 or target.sell_threshold >= target.buy_threshold:
        raise ConfigurationError("Target horizon must be positive and sell_threshold below buy_threshold")
    split_raw = raw.get("split", {})
    split = SplitSettings(**{key: float(split_raw.get(key, getattr(SplitSettings(), key))) for key in SplitSettings.__dataclass_fields__})
    if any(value <= 0 for value in (split.train_ratio, split.validation_ratio, split.test_ratio)) or abs(sum((split.train_ratio, split.validation_ratio, split.test_ratio)) - 1.0) > 1e-9:
        raise ConfigurationError("Split ratios must be positive and sum to 1.0")
    output_dir = (base / raw.get("ml_dataset", {}).get("output_dir", "data/ml")).resolve()
    models_raw = raw.get("models", {})
    model_defaults = ModelsSettings()
    dummy_raw = models_raw.get("dummy", {})
    logistic_raw = models_raw.get("logistic_regression", {})
    catboost_raw = models_raw.get("catboost", {})
    xgboost_raw = models_raw.get("xgboost", {})
    lightgbm_raw = models_raw.get("lightgbm", {})
    models = ModelsSettings(
        dummy=DummySettings(strategy=str(dummy_raw.get("strategy", model_defaults.dummy.strategy))),
        logistic_regression=LogisticRegressionSettings(
            max_iter=int(logistic_raw.get("max_iter", model_defaults.logistic_regression.max_iter)),
            random_state=int(logistic_raw.get("random_state", model_defaults.logistic_regression.random_state)),
            class_weight=logistic_raw.get("class_weight", model_defaults.logistic_regression.class_weight),
        ),
        catboost=CatBoostSettings(
            iterations=int(catboost_raw.get("iterations", model_defaults.catboost.iterations)),
            depth=int(catboost_raw.get("depth", model_defaults.catboost.depth)),
            learning_rate=float(catboost_raw.get("learning_rate", model_defaults.catboost.learning_rate)),
            random_seed=int(catboost_raw.get("random_seed", model_defaults.catboost.random_seed)),
            verbose=bool(catboost_raw.get("verbose", model_defaults.catboost.verbose)),
        ),
        xgboost=XGBoostSettings(
            n_estimators=int(xgboost_raw.get("n_estimators", model_defaults.xgboost.n_estimators)),
            max_depth=int(xgboost_raw.get("max_depth", model_defaults.xgboost.max_depth)),
            learning_rate=float(xgboost_raw.get("learning_rate", model_defaults.xgboost.learning_rate)),
            random_state=int(xgboost_raw.get("random_state", model_defaults.xgboost.random_state)),
        ),
        lightgbm=LightGBMSettings(
            n_estimators=int(lightgbm_raw.get("n_estimators", model_defaults.lightgbm.n_estimators)),
            max_depth=int(lightgbm_raw.get("max_depth", model_defaults.lightgbm.max_depth)),
            learning_rate=float(lightgbm_raw.get("learning_rate", model_defaults.lightgbm.learning_rate)),
            random_state=int(lightgbm_raw.get("random_state", model_defaults.lightgbm.random_state)),
        ),
    )
    positive_parameters = (
        models.logistic_regression.max_iter, models.catboost.iterations, models.catboost.depth,
        models.xgboost.n_estimators, models.xgboost.max_depth,
        models.lightgbm.n_estimators, models.lightgbm.max_depth,
    )
    if any(value <= 0 for value in positive_parameters):
        raise ConfigurationError("Model iteration/depth parameters must be positive")
    evaluation_raw = raw.get("evaluation", {})
    evaluation = EvaluationSettings(
        primary_metric=str(evaluation_raw.get("primary_metric", "macro_f1")),
        models_dir=(base / evaluation_raw.get("models_dir", "artifacts/models")).resolve(),
        results_dir=(base / evaluation_raw.get("results_dir", "artifacts/model_evaluation")).resolve(),
    )
    if evaluation.primary_metric != "macro_f1":
        raise ConfigurationError("Block 3 supports macro_f1 as the primary metric")
    time_series_raw = raw.get("time_series_validation", {})
    time_series_validation = TimeSeriesValidationSettings(
        n_splits=int(time_series_raw.get("n_splits", 5))
    )
    search_raw = raw.get("hyperparameter_search", {})
    hyperparameter_search = HyperparameterSearchSettings(
        n_iter=int(search_raw.get("n_iter", 20)),
        random_state=int(search_raw.get("random_state", 42)),
        scoring=str(search_raw.get("scoring", "macro_f1")),
        models_dir=(base / search_raw.get("models_dir", "artifacts/tuned_models")).resolve(),
        results_dir=(base / search_raw.get("results_dir", "artifacts/hyperparameter_search")).resolve(),
    )
    if time_series_validation.n_splits < 2:
        raise ConfigurationError("time_series_validation.n_splits must be at least 2")
    if hyperparameter_search.n_iter <= 0:
        raise ConfigurationError("hyperparameter_search.n_iter must be positive")
    if hyperparameter_search.scoring != "macro_f1":
        raise ConfigurationError("Block 4 supports macro_f1 as the search metric")
    spaces_raw = raw.get("search_spaces", {})
    cb_defaults = CatBoostSearchSpace()
    xgb_defaults = XGBoostSearchSpace()
    lgb_defaults = LightGBMSearchSpace()
    cb_raw = spaces_raw.get("catboost", {})
    xgb_raw = spaces_raw.get("xgboost", {})
    lgb_raw = spaces_raw.get("lightgbm", {})
    search_spaces = SearchSpacesSettings(
        catboost=CatBoostSearchSpace(
            iterations=_number_tuple(cb_raw.get("iterations", cb_defaults.iterations), "search_spaces.catboost.iterations", int),
            depth=_number_tuple(cb_raw.get("depth", cb_defaults.depth), "search_spaces.catboost.depth", int),
            learning_rate=_number_tuple(cb_raw.get("learning_rate", cb_defaults.learning_rate), "search_spaces.catboost.learning_rate"),
            l2_leaf_reg=_number_tuple(cb_raw.get("l2_leaf_reg", cb_defaults.l2_leaf_reg), "search_spaces.catboost.l2_leaf_reg", int),
        ),
        xgboost=XGBoostSearchSpace(
            n_estimators=_number_tuple(xgb_raw.get("n_estimators", xgb_defaults.n_estimators), "search_spaces.xgboost.n_estimators", int),
            max_depth=_number_tuple(xgb_raw.get("max_depth", xgb_defaults.max_depth), "search_spaces.xgboost.max_depth", int),
            learning_rate=_number_tuple(xgb_raw.get("learning_rate", xgb_defaults.learning_rate), "search_spaces.xgboost.learning_rate"),
            subsample=_number_tuple(xgb_raw.get("subsample", xgb_defaults.subsample), "search_spaces.xgboost.subsample"),
            colsample_bytree=_number_tuple(xgb_raw.get("colsample_bytree", xgb_defaults.colsample_bytree), "search_spaces.xgboost.colsample_bytree"),
            min_child_weight=_number_tuple(xgb_raw.get("min_child_weight", xgb_defaults.min_child_weight), "search_spaces.xgboost.min_child_weight", int),
        ),
        lightgbm=LightGBMSearchSpace(
            n_estimators=_number_tuple(lgb_raw.get("n_estimators", lgb_defaults.n_estimators), "search_spaces.lightgbm.n_estimators", int),
            max_depth=_number_tuple(lgb_raw.get("max_depth", lgb_defaults.max_depth), "search_spaces.lightgbm.max_depth", int),
            learning_rate=_number_tuple(lgb_raw.get("learning_rate", lgb_defaults.learning_rate), "search_spaces.lightgbm.learning_rate"),
            num_leaves=_number_tuple(lgb_raw.get("num_leaves", lgb_defaults.num_leaves), "search_spaces.lightgbm.num_leaves", int),
            subsample=_number_tuple(lgb_raw.get("subsample", lgb_defaults.subsample), "search_spaces.lightgbm.subsample"),
            colsample_bytree=_number_tuple(lgb_raw.get("colsample_bytree", lgb_defaults.colsample_bytree), "search_spaces.lightgbm.colsample_bytree"),
            min_child_samples=_number_tuple(lgb_raw.get("min_child_samples", lgb_defaults.min_child_samples), "search_spaces.lightgbm.min_child_samples", int),
        ),
    )

    backtesting_raw = raw.get("backtesting", {})
    strategies_raw = backtesting_raw.get("strategies", {
        "lightgbm_block3": {
            "display_name": "LightGBM Block 3",
            "model_path": "artifacts/models/lightgbm.joblib",
        },
        "lightgbm_tuned": {
            "display_name": "Tuned LightGBM",
            "model_path": "artifacts/tuned_models/lightgbm_tuned.joblib",
        },
    })
    if not isinstance(strategies_raw, dict) or not strategies_raw:
        raise ConfigurationError("backtesting.strategies must be a non-empty mapping")
    strategies = tuple(
        BacktestStrategySettings(
            slug=str(slug),
            display_name=str(_required(strategy, "display_name", f"backtesting.strategies.{slug}")),
            model_path=(base / _required(strategy, "model_path", f"backtesting.strategies.{slug}")).resolve(),
            model_version=str(strategy.get("model_version", "unspecified")),
        )
        for slug, strategy in strategies_raw.items()
    )
    if len({strategy.slug for strategy in strategies}) != len(strategies):
        raise ConfigurationError("Backtesting strategy slugs must be unique")
    if any(not strategy.slug.replace("_", "").isalnum() for strategy in strategies):
        raise ConfigurationError("Backtesting strategy slugs must contain only letters, digits, and underscores")
    backtesting = BacktestingSettings(
        initial_cash=float(backtesting_raw.get("initial_cash", 10000.0)),
        commission_rate=float(backtesting_raw.get("commission_rate", 0.001)),
        slippage_rate=float(backtesting_raw.get("slippage_rate", 0.0)),
        allow_short=bool(backtesting_raw.get("allow_short", False)),
        leverage=float(backtesting_raw.get("leverage", 1.0)),
        force_close_at_end=bool(backtesting_raw.get("force_close_at_end", True)),
        output_dir=(base / backtesting_raw.get("output_dir", "artifacts/backtesting")).resolve(),
        strategies=strategies,
    )
    if backtesting.initial_cash <= 0:
        raise ConfigurationError("backtesting.initial_cash must be positive")
    if not 0 <= backtesting.commission_rate < 1:
        raise ConfigurationError("backtesting.commission_rate must be in [0, 1)")
    if not 0 <= backtesting.slippage_rate < 1:
        raise ConfigurationError("backtesting.slippage_rate must be in [0, 1)")
    if backtesting.allow_short or backtesting.leverage != 1.0:
        raise ConfigurationError("Block 5 requires allow_short=false and leverage=1.0")
    if not backtesting.force_close_at_end:
        raise ConfigurationError("Block 5 requires force_close_at_end=true")

    mlflow_raw = raw.get("mlflow", {})
    mlflow_defaults = MLflowSettings()
    artifact_location = mlflow_raw.get("artifact_location", mlflow_defaults.artifact_location)
    mlflow = MLflowSettings(
        tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI",
            str(mlflow_raw.get("tracking_uri", mlflow_defaults.tracking_uri)),
        ).strip(),
        experiment_name=str(
            mlflow_raw.get("experiment_name", mlflow_defaults.experiment_name)
        ).strip(),
        registered_model_name=str(
            mlflow_raw.get("registered_model_name", mlflow_defaults.registered_model_name)
        ).strip(),
        artifact_location=None if artifact_location is None else str(artifact_location).strip(),
        enabled=bool(mlflow_raw.get("enabled", mlflow_defaults.enabled)),
    )
    if not mlflow.tracking_uri:
        raise ConfigurationError("mlflow.tracking_uri cannot be empty")
    if not mlflow.experiment_name:
        raise ConfigurationError("mlflow.experiment_name cannot be empty")
    if not mlflow.registered_model_name:
        raise ConfigurationError("mlflow.registered_model_name cannot be empty")

    api_raw = raw.get("api", {})
    api = APISettings(
        host=str(api_raw.get("host", "127.0.0.1")).strip(),
        port=int(api_raw.get("port", 8000)),
        title=str(api_raw.get("title", "Crypto ML Inference API")).strip(),
        version=str(api_raw.get("version", "1.0.0")).strip(),
    )
    if not api.host or not api.title or not api.version:
        raise ConfigurationError("api host, title, and version cannot be empty")
    if not 1 <= api.port <= 65535:
        raise ConfigurationError("api.port must be in [1, 65535]")

    inference_raw = raw.get("inference", {})
    inference = InferenceSettings(
        registered_model_name=str(
            inference_raw.get("registered_model_name", mlflow.registered_model_name)
        ).strip(),
        model_alias=str(inference_raw.get("model_alias", "champion")).strip(),
        expected_symbol=str(
            inference_raw.get("expected_symbol", market.get("symbol", "BTCUSDT"))
        ).strip().upper(),
        expected_timeframe=str(
            inference_raw.get("expected_timeframe", market.get("timeframe", "1h"))
        ).strip(),
    )
    if not all((
        inference.registered_model_name,
        inference.model_alias,
        inference.expected_symbol,
        inference.expected_timeframe,
    )):
        raise ConfigurationError("inference settings cannot be empty")

    database_raw = raw.get("database", {})
    database = DatabaseSettings(
        url_env_variable=str(
            database_raw.get("url_env_variable", "DATABASE_URL")
        ).strip(),
        pool_pre_ping=bool(database_raw.get("pool_pre_ping", True)),
    )
    if not database.url_env_variable:
        raise ConfigurationError("database.url_env_variable cannot be empty")

    monitoring_raw = raw.get("monitoring", {})
    monitoring_defaults = MonitoringSettings()
    monitoring = MonitoringSettings(
        reference_path=(base / os.getenv(
            "MONITORING_REFERENCE_PATH",
            str(monitoring_raw.get("reference_path", monitoring_defaults.reference_path)),
        )).resolve(),
        window_size=int(os.getenv(
            "MONITORING_WINDOW_SIZE",
            monitoring_raw.get("window_size", monitoring_defaults.window_size),
        )),
        min_samples=int(os.getenv(
            "MONITORING_MIN_SAMPLES",
            monitoring_raw.get("min_samples", monitoring_defaults.min_samples),
        )),
        poll_interval_seconds=float(os.getenv(
            "MONITORING_POLL_INTERVAL_SECONDS",
            monitoring_raw.get(
                "poll_interval_seconds", monitoring_defaults.poll_interval_seconds
            ),
        )),
        metrics_host=str(os.getenv(
            "MONITORING_METRICS_HOST",
            monitoring_raw.get("metrics_host", monitoring_defaults.metrics_host),
        )).strip(),
        metrics_port=int(os.getenv(
            "MONITORING_METRICS_PORT",
            monitoring_raw.get("metrics_port", monitoring_defaults.metrics_port),
        )),
        warning_threshold=float(monitoring_raw.get(
            "warning_threshold", monitoring_defaults.warning_threshold
        )),
        critical_threshold=float(monitoring_raw.get(
            "critical_threshold", monitoring_defaults.critical_threshold
        )),
    )
    if monitoring.window_size <= 0 or monitoring.min_samples <= 0:
        raise ConfigurationError("monitoring window_size and min_samples must be positive")
    if monitoring.min_samples > monitoring.window_size:
        raise ConfigurationError("monitoring.min_samples cannot exceed window_size")
    if monitoring.poll_interval_seconds <= 0:
        raise ConfigurationError("monitoring.poll_interval_seconds must be positive")
    if not 1 <= monitoring.metrics_port <= 65535 or not monitoring.metrics_host:
        raise ConfigurationError("monitoring metrics endpoint is invalid")
    if not 0 < monitoring.warning_threshold < monitoring.critical_threshold:
        raise ConfigurationError(
            "monitoring thresholds must satisfy 0 < warning < critical"
        )


    return AppSettings(
        market_data=MarketDataSettings(
            provider=str(_required(market, "provider", "market_data")).lower(),
            symbol=str(_required(market, "symbol", "market_data")).upper(),
            timeframe=str(_required(market, "timeframe", "market_data")),
            start_date=start,
            end_date=end,
        ),
        storage=StorageSettings(
            raw_dir=(base / _required(storage, "raw_dir", "storage")).resolve(),
            processed_dir=(base / _required(storage, "processed_dir", "storage")).resolve(),
            format=fmt,
        ),
        http=HttpSettings(
            timeout_seconds=float(_required(http, "timeout_seconds", "http")),
            max_retries=retries,
            retry_delay_seconds=float(_required(http, "retry_delay_seconds", "http")),
        ),
        logging=LoggingSettings(level=str(_required(logging_cfg, "level", "logging")).upper()),
        features=features,
        target=target,
        split=split,
        ml_dataset=MLDatasetSettings(output_dir),
        models=models,
        evaluation=evaluation,
        time_series_validation=time_series_validation,
        hyperparameter_search=hyperparameter_search,
        search_spaces=search_spaces,
        backtesting=backtesting,
        mlflow=mlflow,
        api=api,
        inference=inference,
        database=database,
        monitoring=monitoring,
    )
