import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def asymmetric_msle_eval(preds, train_data):
    """
    Эта функция считает ассиметричный MSLE.

    :param preds: np.array, предсказания модели
    :param train_data: LightGBM Dataset, содержит реальные значения
    :return: tuple, название метрики, значение метрики и флаг (False означает "меньше лучше")
    """
    actuals = train_data.get_label()
    res = np.log1p(preds) - np.log1p(actuals)
    res_squared = np.square(res)
    
    # Увеличение ошибки, если предсказание меньше фактического
    cond_less = res < 0
    res_squared[cond_less] *= 1.2

    msle = np.mean(res_squared)

    return 'asymmetric_msle', msle, False

def asymmetric_msle_objective(preds, train_data):
    """
    Асимметричная версия функции потерь Mean Squared Logarithmic Error (MSLE) для использования
    в градиентном бустинге с LightGBM. Функция возвращает градиент и гессиан относительно предсказаний.

    Параметры:
        preds (np.array): Массив предсказанных значений, генерируемых моделью.
        train_data (lightgbm.Dataset): Данные для обучения, из которых можно получить фактические метки.

    Возвращает:
        tuple: Возвращает кортеж, содержащий массивы градиентов и гессианов, необходимые для обучения модели.

    Пример:
        Предположим, `train_data` — это объект `lightgbm.Dataset` с правильными метками, а `preds` — это
        предсказания модели. Тогда `grad` и `hess` могут быть использованы для кастомной оптимизации
        в процессе обучения модели LightGBM.
    """
    actual = train_data.get_label()
    res = np.log1p(preds) - np.log1p(actual)
    factor = np.where(res < 0, 1.2, 1.0)

    # Градиент для асимметричной MSLE
    grad = 2 * res * factor / (1 + preds)
    
    # Гессиан для асимметричной MSLE
    hess = -2 * (res - 1) * factor / (1 + preds)**2
    
    return grad, hess

def asymmetric_msle_score(preds, actual):
    """
    Вычисляет асимметричную версию среднеквадратичной логарифмической ошибки (MSLE).

    Параметры:
        preds (np.array): Массив предсказанных значений моделью.
        actual (np.array): Массив фактических значений, соответствующих предсказаниям.

    Возвращает:
        float: Значение асимметричного MSLE для предоставленных предсказаний и фактических значений.

    Пример использования:
        # Допустим, у нас есть массивы preds и actual, содержащие предсказания модели и фактические значения
        error = asymmetric_msle_score(preds, actual)
        print(f"Асимметричный MSLE: {error}")
    """
    res = np.log1p(preds) - np.log1p(actual)
    cond_less = res < 0
    res = res ** 2
    res[cond_less] = res[cond_less] * 1.2
    return res.mean()


def train_and_visualize(train_df, test_df, features, target, cat_features, params, n_splits=5):
    """
    Обучает LightGBM модель с визуализацией процесса обучения для асимметричной MSLE.

    Параметры:
        train_df (pandas.DataFrame): DataFrame с тренировочными данными.
        test_df (pandas.DataFrame): DataFrame с тестовыми данными.
        features (list): Список колонок, используемых в качестве признаков.
        target (pandas.Series): Целевая переменная.
        cat_features (list): Список категориальных признаков.
        params (dict): Параметры для обучения модели LightGBM.
        n_splits (int): Количество фолдов для кросс-валидации.
    
    Возвращает:
        np.array: Предсказания для валидационного и тестового наборов.
        pandas.DataFrame: Таблица фича импортанс, для дальнейшей визуализации.
    """
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=15)
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    feature_importance_df = pd.DataFrame()
    start = time.time()
    
    fig, axs = plt.subplots(n_splits, 1, figsize=(10, n_splits * 5))  # Создаем фигуру с subplots для каждого фолда

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=cat_features)
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=cat_features)
        
        evals_result = {}  # Словарь для хранения результатов обучения
        
        clf = lgb.train(
            params,
            trn_data,
            num_boost_round=10000,
            valid_sets=[trn_data, val_data],
            feval=asymmetric_msle_eval,
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(100),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
        
        # Визуализация результатов обучения для текущего фолда
        axs[fold_].plot(evals_result['training']['asymmetric_msle'], label='Train Asymmetric MSLE')
        axs[fold_].plot(evals_result['valid_1']['asymmetric_msle'], label='Validation Asymmetric MSLE')
        axs[fold_].set_title(f'Fold {fold_ + 1}')
        axs[fold_].set_xlabel('Boosting Rounds')
        axs[fold_].set_ylabel('Asymmetric MSLE')
        axs[fold_].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Training completed in {time.time() - start} seconds.")
    
    return oof, predictions, feature_importance_df

