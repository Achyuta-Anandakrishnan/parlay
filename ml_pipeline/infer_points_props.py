import math
import pickle

import pandas as pd

MODEL_PATH = "ml_pipeline/models/points_model.pkl"
DATA_PATH = "ml_pipeline/data/features_player_points.csv"
CURRENT_SEASON = "2025-26"
DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "season",
    "points_actual",
]


def round_to_half(value):
    return round(value * 2) / 2


def normal_cdf(x, mean, std):
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    df = pd.read_csv(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["season"] = df["GAME_DATE"].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 10 else f"{d.year - 1}-{str(d.year)[-2:]}"
    )

    current_season_df = df[df["season"] == CURRENT_SEASON].copy()
    if current_season_df.empty:
        raise SystemExit(f"No feature rows found for season {CURRENT_SEASON}")

    today = current_season_df["GAME_DATE"].max()
    today_df = (
        current_season_df.sort_values(["PLAYER_ID", "GAME_DATE"])
        .groupby("PLAYER_ID", as_index=False)
        .tail(1)
        .copy()
    )

    X_today = today_df.drop(columns=DROP_COLUMNS)
    today_df["predicted_points"] = model.predict(X_today)

    today_df["std"] = pd.to_numeric(today_df["pts_std_last_10"], errors="coerce")
    today_df["std"] = today_df["std"].where(today_df["std"] > 0, 8.0).fillna(8.0)
    today_df["line"] = today_df["predicted_points"].apply(round_to_half)
    today_df["P_over"] = today_df.apply(
        lambda row: 1 - normal_cdf(row["line"], row["predicted_points"], row["std"]),
        axis=1,
    )

    output = today_df[["PLAYER_NAME", "predicted_points", "line", "std", "P_over"]].copy()
    output = output.sort_values("P_over", ascending=False).reset_index(drop=True)

    print(f"today: {today.date()}")
    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
