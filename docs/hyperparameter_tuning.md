# Подбор гиперпараметров с временной валидацией

## Границы данных

Подбор CatBoost, XGBoost и LightGBM выполняется только на **data/ml/BTCUSDT_1h_train.parquet**. Внутри Train используется expanding-window TimeSeriesSplit: обучающая часть каждого следующего фолда растёт, а проверочная часть всегда находится позже неё во времени. Перемешивания нет.

Validation не передаётся в RandomizedSearchCV и загружается только после завершения поиска для всех трёх моделей. Затем модель с лучшими CV-параметрами заново обучается на полном Train и ровно один раз оценивается на Validation. Test не читается, не оценивается и не участвует ни в подборе параметров, ни в выборе победителя.

## Gap и prediction timestamp

Строка с timestamp t становится доступной для прогноза только после полного закрытия свечи t. Все признаки этой строки используют закрытую свечу t и более ранние данные. Цель описывает изменение Close от close[t] до Close через target.horizon_hours.

Между концом обучающей части и началом проверочной части каждого CV-фолда оставляется gap, равный горизонту цели в строках:

~~~text
gap = target.horizon_hours / длительность market_data.timeframe
~~~

Для target.horizon_hours = 3 и timeframe 1h это три строки. Такая очистка не позволяет целям последних обучающих строк использовать цены из проверочного временного интервала. Не кратное timeframe значение горизонта считается ошибкой конфигурации выполнения.

## Конфигурация и запуск

Все диапазоны находятся в **config.yaml**, в секциях time_series_validation, hyperparameter_search и search_spaces. Проектное имя метрики macro_f1 явно преобразуется в sklearn scorer f1_macro. Случайность выборки комбинаций фиксируется через random_state = 42.

Для LightGBM в tuned-пути фиксируется subsample_freq = 1, чтобы перебираемый subsample действительно включал выборку строк для каждого дерева. Эта настройка не применяется к моделям и baseline этапа 3.

Перед запуском должны существовать ML-сплиты этапа 2 и **artifacts/model_evaluation/model_comparison.json** этапа 3.

~~~powershell
.\.venv\Scripts\python.exe scripts\tune_models.py
~~~

## Артефакты

Лучшие модели сохраняются отдельно от моделей этапа 3:

- artifacts/tuned_models/catboost_tuned.joblib;
- artifacts/tuned_models/xgboost_tuned.joblib;
- artifacts/tuned_models/lightgbm_tuned.joblib.

В **artifacts/hyperparameter_search/** сохраняются CSV со всеми проверенными кандидатами, JSON с лучшими параметрами и CV-summary для каждой модели, сравнение исходной и tuned Validation Macro F1, метрики по классам, confusion matrix, метаданные и best_tuned_model.json.

CV-summary содержит оценки каждого фолда, среднее и стандартное отклонение Macro F1, длительность поиска, границы фолдов и наблюдаемый gap. Артефакты этапа 3 не перезаписываются.
