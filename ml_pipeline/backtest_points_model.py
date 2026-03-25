import math
import pickle

import pandas as pd

MODEL_PATH = "ml_pipeline/models/points_model.pkl"
DATA_PATH = "ml_pipeline/data/features_player_points.csv"
OUTPUT_PATH = "ml_pipeline/data/backtest_results.csv"
DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "points_actual",
]
BET_SIZE = 100.0
WIN_PROFIT = 90.91
LOSS_PROFIT = -100.0
SPORTSBOOK_IMPLIED_PROB = 0.524


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
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    X = df.drop(columns=DROP_COLUMNS)
    df["predicted_points"] = model.predict(X)

    df["std"] = pd.to_numeric(df["pts_std_last_10"], errors="coerce")
    df["std"] = df["std"].where(df["std"] > 0, 8.0).fillna(8.0).clip(lower=4.0, upper=15.0)
    df["line"] = df["season_avg_points_prior"].apply(round_to_half)

    df["p_over_model"] = df.apply(
        lambda row: 1 - normal_cdf(row["line"], row["predicted_points"], row["std"]),
        axis=1,
    )
    df["p_under_model"] = 1 - df["p_over_model"]
    df["sportsbook_implied_prob"] = SPORTSBOOK_IMPLIED_PROB

    df["edge_over"] = df["p_over_model"] - 0.5
    df["edge_under"] = df["p_under_model"] - 0.5
    df["bet"] = df.apply(
        lambda row: "over" if row["edge_over"] > row["edge_under"] else "under",
        axis=1,
    )
    df["chosen_edge"] = df.apply(
        lambda row: row["edge_over"] if row["bet"] == "over" else row["edge_under"],
        axis=1,
    )

    df["win"] = df.apply(
        lambda row: 1
        if (row["bet"] == "over" and row["points_actual"] > row["line"])
        or (row["bet"] == "under" and row["points_actual"] < row["line"])
        else 0,
        axis=1,
    )
    df["profit"] = df["win"].apply(lambda win: WIN_PROFIT if win == 1 else LOSS_PROFIT)

    output = df[
        [
            "PLAYER_NAME",
            "GAME_DATE",
            "predicted_points",
            "line",
            "bet",
            "points_actual",
            "win",
            "profit",
        ]
    ].copy()
    output = output.rename(columns={"points_actual": "actual_points"})
    output["GAME_DATE"] = output["GAME_DATE"].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_PATH, index=False)

    total_bets = len(df)
    win_rate = df["win"].mean()
    total_profit = df["profit"].sum()
    roi = total_profit / (total_bets * BET_SIZE) if total_bets else 0.0

    print(f"total bets: {total_bets}")
    print(f"win rate: {win_rate:.4f}")
    print(f"ROI: {roi:.4f}")
    print("top 20 highest edge bets:")
    print(
        df[
            [
                "PLAYER_NAME",
                "GAME_DATE",
                "predicted_points",
                "line",
                "bet",
                "points_actual",
                "chosen_edge",
            ]
        ]
        .sort_values("chosen_edge", ascending=False)
        .head(20)
        .assign(GAME_DATE=lambda frame: frame["GAME_DATE"].dt.strftime("%Y-%m-%d"))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
