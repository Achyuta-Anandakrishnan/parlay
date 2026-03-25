import math
import os
import pickle

import pandas as pd
from lightgbm import LGBMRegressor

INPUT_PATH = "ml_pipeline/data/features_player_points_context.csv"
TARGET_SOURCE_PATH = "ml_pipeline/data/player_games_rich.csv"
OUTPUT_FEATURES_PATH = "ml_pipeline/data/features_player_points_with_oof_minutes.csv"
OUTPUT_MODEL_PATH = "ml_pipeline/models/minutes_model_context_final.pkl"
DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "points_actual",
    "minutes_actual",
]
INITIAL_TRAIN_FRACTION = 0.10
OOF_FOLDS = 8


def build_model():
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
    )


def load_dataset():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    if "minutes_actual" not in df.columns:
        target_source = pd.read_csv(
            TARGET_SOURCE_PATH,
            usecols=["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MIN"],
        )
        target_source["GAME_DATE"] = pd.to_datetime(target_source["GAME_DATE"], errors="coerce")
        target_source = target_source.rename(columns={"MIN": "minutes_actual"})
        target_source["minutes_actual"] = pd.to_numeric(target_source["minutes_actual"], errors="coerce")

        df = df.merge(
            target_source,
            on=["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION"],
            how="left",
        )

    df["minutes_actual"] = pd.to_numeric(df["minutes_actual"], errors="coerce")
    df = df.dropna(subset=["minutes_actual"]).copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def get_model_drop_columns(df):
    additional_drop_columns = [
        column
        for column in df.columns
        if column not in DROP_COLUMNS and df[column].dtype == "object"
    ]
    return DROP_COLUMNS + additional_drop_columns


def main():
    df = load_dataset()
    model_drop_columns = get_model_drop_columns(df)
    X_full = df.drop(columns=model_drop_columns)
    y_full = df["minutes_actual"]

    total_rows = len(df)
    initial_train_rows = max(1, int(total_rows * INITIAL_TRAIN_FRACTION))
    remaining_rows = max(0, total_rows - initial_train_rows)
    validation_block_size = max(1, math.ceil(remaining_rows / OOF_FOLDS)) if remaining_rows else 1

    df["predicted_minutes_oof"] = pd.NA

    fold_start = initial_train_rows
    while fold_start < total_rows:
        fold_end = min(total_rows, fold_start + validation_block_size)

        X_train = X_full.iloc[:fold_start]
        y_train = y_full.iloc[:fold_start]
        X_valid = X_full.iloc[fold_start:fold_end]

        model = build_model()
        model.fit(X_train, y_train)
        df.loc[fold_start:fold_end - 1, "predicted_minutes_oof"] = model.predict(X_valid)

        fold_start = fold_end

    df["predicted_minutes_oof"] = pd.to_numeric(df["predicted_minutes_oof"], errors="coerce")
    df["predicted_minus_recent_minutes_oof"] = df["predicted_minutes_oof"] - df["min_last_5"]

    final_model = build_model()
    final_model.fit(X_full, y_full)

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    with open(OUTPUT_MODEL_PATH, "wb") as file:
        pickle.dump(final_model, file)

    output_df = df.copy()
    output_df["GAME_DATE"] = output_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_FEATURES_PATH, index=False)

    predicted_mask = output_df["predicted_minutes_oof"].notna()
    print(f"total rows: {total_rows}")
    print(f"rows with OOF predictions: {int(predicted_mask.sum())}")
    print(f"rows missing OOF predictions: {int((~predicted_mask).sum())}")
    print("predicted_minutes_oof summary:")
    print(output_df.loc[predicted_mask, "predicted_minutes_oof"].describe().to_string())
    correlation = df.loc[predicted_mask, "predicted_minutes_oof"].corr(df.loc[predicted_mask, "minutes_actual"])
    print(f"correlation(predicted_minutes_oof, minutes_actual): {correlation:.4f}")


if __name__ == "__main__":
    main()
