import os
import pickle

import pandas as pd

INPUT_PATH = "ml_pipeline/data/features_player_points_context.csv"
MINUTES_MODEL_PATH = "ml_pipeline/models/minutes_model_context.pkl"
TARGET_SOURCE_PATH = "ml_pipeline/data/player_games_rich.csv"
OUTPUT_PATH = "ml_pipeline/data/features_player_points_with_minutes.csv"


def add_minutes_actual_if_available(df):
    if "minutes_actual" in df.columns:
        df["minutes_actual"] = pd.to_numeric(df["minutes_actual"], errors="coerce")
        return df

    target_source = pd.read_csv(
        TARGET_SOURCE_PATH,
        usecols=["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MIN"],
    )
    target_source["GAME_DATE"] = pd.to_datetime(target_source["GAME_DATE"], errors="coerce")
    target_source = target_source.rename(columns={"MIN": "minutes_actual"})
    target_source["minutes_actual"] = pd.to_numeric(target_source["minutes_actual"], errors="coerce")

    return df.merge(
        target_source,
        on=["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION"],
        how="left",
    )


def main():
    with open(MINUTES_MODEL_PATH, "rb") as file:
        minutes_model = pickle.load(file)

    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = add_minutes_actual_if_available(df)

    minutes_feature_columns = list(minutes_model.feature_name_)
    X_minutes = df[minutes_feature_columns].copy()

    df["predicted_minutes"] = minutes_model.predict(X_minutes)
    df["predicted_minus_recent_minutes"] = df["predicted_minutes"] - df["min_last_5"]

    output_df = df.copy()
    output_df["GAME_DATE"] = output_df["GAME_DATE"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"total rows: {len(output_df)}")
    print("predicted_minutes summary:")
    print(df["predicted_minutes"].describe().to_string())

    if "minutes_actual" in df.columns and df["minutes_actual"].notna().any():
        correlation = df["predicted_minutes"].corr(df["minutes_actual"])
        print(f"correlation(predicted_minutes, minutes_actual): {correlation:.4f}")
    else:
        print("correlation(predicted_minutes, minutes_actual): unavailable")


if __name__ == "__main__":
    main()
