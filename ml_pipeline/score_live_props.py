import math
import pickle
import re
import unicodedata

import pandas as pd

MODEL_PATH = "ml_pipeline/models/points_model.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points.csv"
PROPS_PATH = "ml_pipeline/data/upcoming_player_points_props.csv"
OUTPUT_PATH = "ml_pipeline/data/live_scored_props.csv"
DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "points_actual",
]
PLAYER_ALIASES = {
    "nicolas claxton": "nic claxton",
    "carlton carrington": "bub carrington",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = text.split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    text = " ".join(parts)
    text = PLAYER_ALIASES.get(text, text)
    return text


def normal_cdf(x, mean, std):
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def american_to_prob(odds):
    if pd.isna(odds):
        return math.nan

    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return math.nan


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    features = pd.read_csv(FEATURES_PATH)
    props = pd.read_csv(PROPS_PATH)

    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="coerce")
    features = features.sort_values("GAME_DATE").reset_index(drop=True)

    features["normalized_player_name"] = features["PLAYER_NAME"].apply(normalize_name)
    props["normalized_player_name"] = props["player_name"].apply(normalize_name)

    latest_features = features.drop_duplicates(subset=["PLAYER_ID"], keep="last").copy()
    latest_features = latest_features.sort_values("GAME_DATE").drop_duplicates(
        subset=["normalized_player_name"], keep="last"
    )

    merged = props.merge(
        latest_features,
        how="left",
        on="normalized_player_name",
        suffixes=("_prop", ""),
    )

    total_props_rows = len(merged)
    matched = merged[merged["PLAYER_ID"].notna()].copy()
    unmatched_rows = total_props_rows - len(matched)

    print(f"total props rows: {total_props_rows}")
    print(f"matched rows: {len(matched)}")
    print(f"unmatched rows: {unmatched_rows}")

    feature_columns = [
        column
        for column in features.columns
        if column not in DROP_COLUMNS and column != "normalized_player_name"
    ]

    matched["predicted_points"] = model.predict(matched[feature_columns])

    matched["std"] = pd.to_numeric(matched["pts_std_last_10"], errors="coerce")
    matched["std"] = matched["std"].where(matched["std"] > 0, 8.0).fillna(8.0).clip(lower=4.0, upper=15.0)

    matched["P_over_model"] = matched.apply(
        lambda row: 1 - normal_cdf(row["line"], row["predicted_points"], row["std"]),
        axis=1,
    )
    matched["P_under_model"] = 1 - matched["P_over_model"]

    matched["P_over_raw"] = matched["over_price"].apply(american_to_prob)
    matched["P_under_raw"] = matched["under_price"].apply(american_to_prob)
    matched["market_prob_sum"] = matched["P_over_raw"] + matched["P_under_raw"]
    matched["P_over_market"] = matched["P_over_raw"] / matched["market_prob_sum"]
    matched["P_under_market"] = matched["P_under_raw"] / matched["market_prob_sum"]

    matched["edge_over"] = matched["P_over_model"] - matched["P_over_market"]
    matched["edge_under"] = matched["P_under_model"] - matched["P_under_market"]
    matched["best_side"] = matched.apply(
        lambda row: "over" if row["edge_over"] > row["edge_under"] else "under",
        axis=1,
    )
    matched["best_edge"] = matched.apply(
        lambda row: row["edge_over"] if row["best_side"] == "over" else row["edge_under"],
        axis=1,
    )

    output = matched[
        [
            "player_name",
            "bookmaker",
            "line",
            "predicted_points",
            "P_over_model",
            "P_over_market",
            "edge_over",
            "edge_under",
            "best_side",
            "best_edge",
        ]
    ].copy()
    output = output.sort_values("best_edge", ascending=False).reset_index(drop=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("top 20 bets:")
    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
