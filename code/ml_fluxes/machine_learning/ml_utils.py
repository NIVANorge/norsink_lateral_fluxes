import time
import nivapy3 as nivapy
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_models(models, X_train, y_train, groups=None, cv_folds=5):
    """Fit ML models using cross-validation (with optional grouping) and
    print goodness-of-fit statistics.

    Args
        models: List of tuples ('Label', model_obj). Models to evaluate.
        X_train: Dataframe of feature data used for training.
        y_train: Dataseries of target values for training.
        groups: Dataseries or None. Group (i.e. catchment) IDs for each row
            in X_train. If None, no grouping is performed.
        cv_folds: Int. Number of folds to use for cross-validation.

    Returns
        Dict of goodness-of-fit statistics for each model in 'models'.
    """
    results = {}
    for name, model in models:
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", model)])
        if groups is not None:
            cv = list(
                GroupKFold(n_splits=cv_folds).split(X_train, y_train, groups=groups)
            )
        else:
            cv = cv_folds
        results[name] = {
            "RMSE": np.sqrt(
                -cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                ).mean()
            ),
            "R2": cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="r2",
                n_jobs=-1,
            ).mean(),
            "MAE": -cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            ).mean(),
        }
    return results


def evaluate_models_grid_search_cv(models, X_train, y_train, groups=None, cv_folds=5):
    """Fit ML models using cross-validation (with optional grouping) and
    print goodness-of-fit statistics.

    Args
        models: List of tuples ('Label', model_obj, param_grid). Models to evaluate.
        X_train: Dataframe of feature data used for training.
        y_train: Dataseries of target values for training.
        groups: Dataseries or None. Group (i.e. catchment) IDs for each row
            in X_train. If None, no grouping is performed.
        cv_folds: Int. Number of folds to use for cross-validation.

    Returns
        Dict of goodness-of-fit statistics for each model in 'models'.
    """

    results = {}

    for name, model, param_grid in models:

        print(f"\nStarting {name}...")
        start = time.time()

        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", model)])
        if groups is not None:
            cv = list(
                GroupKFold(n_splits=cv_folds).split(X_train, y_train, groups=groups)
            )
        else:
            cv = cv_folds

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=2,  # or -1
            verbose=2,
        )
        grid_search.fit(X_train, y_train)

        elapsed = time.time() - start
        print(f"{name} completed in {elapsed / 60:.1f} minutes")

        best_model = grid_search.best_estimator_

        print("Repeating CV to calculate R2...")

        scores = cross_validate(
            best_model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
            },
            n_jobs=1,
        )

        r2 = scores["test_r2"].mean()
        rmse = -scores["test_rmse"].mean()
        mae = -scores["test_mae"].mean()

        results[name] = {
            "Best Params": grid_search.best_params_,
            "Best Estimator": best_model,
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae
        }
    return results


def get_random_site(df, id_col):
    """Choose a site at random from the input dataframe."""
    catch_id = df[id_col].drop_duplicates().sample().iloc[0]
    catch_df = df.query(f"{id_col} == @catch_id").copy()
    # st_yr, end_yr = catch_df["year"].min(), catch_df["year"].max()
    return catch_id, catch_df #, st_yr, end_yr


def predict_target(pred_df, pipeline, target_col, X):
    """Predict values using the fitted model/pipeline."""
    pred_X = pred_df.drop(
        columns=[target_col],
        errors="ignore"
    )[X.columns]
    pred_df[f"pred_{target_col}"] = pipeline.predict(pred_X)
    pred_df[f"pred_{target_col}"] = pred_df[f"pred_{target_col}"].clip(lower=0)
    return pred_df


def predict_with_interval(pipeline, pred_df, target_col, X, alpha=0.95, n_boot=100):
    pred_X = pred_df.drop(
        columns=[
            "date",
            "catch_id",
            "station_name",
            "lon",
            "lat",
            target_col,
        ],
        errors="ignore"  # if cols not present
    )[X.columns]

    preds = []
    for i in range(n_boot):
        sample_indices = np.random.choice(
            range(len(pred_X)), size=len(pred_X), replace=True
        )
        sample_X = pred_X.iloc[sample_indices]
        preds.append(pipeline.predict(sample_X))
    preds = np.array(preds)
    lower_bound = np.percentile(preds, (1 - alpha) / 2 * 100, axis=0)
    upper_bound = np.percentile(preds, (1 + alpha) / 2 * 100, axis=0)

    pred_df[f"pred_{target_col}"] = pipeline.predict(pred_X)
    pred_df[f"pred_{target_col}"] = pred_df[f"pred_{target_col}"].clip(lower=0)

    pred_df[f"lower_{target_col}"] = lower_bound
    pred_df[f"lower_{target_col}"] = pred_df[f"lower_{target_col}"].clip(lower=0)

    pred_df[f"upper_{target_col}"] = upper_bound
    pred_df[f"upper_{target_col}"] = pred_df[f"upper_{target_col}"].clip(lower=0)

    return pred_df