import math
import pickle
import re
import unicodedata

import pandas as pd
from nba_api.stats.static import teams

MODEL_PATH = "ml_pipeline/models/points_model.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points.csv"
PROPS_PATH = "ml_pipeline/data/upcoming_player_points_props.csv"
OUTPUT_PATH = "ml_pipeline/data/corrected_live_scored_props.csv"
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
TEAM_NAME_ALIASES = {
    "la clippers": "los angeles clippers",
}


def normalize_text(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_player_name(value):
    text = normalize_text(value)
    parts = text.split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    text = " ".join(parts)
    return PLAYER_ALIASES.get(text, text)


def normalize_team_name(value):
    text = normalize_text(value)
    return TEAM_NAME_ALIASES.get(text, text)


def season_from_date(value):
    value = pd.Timestamp(value)
    if value.month >= 10:
        return f"{value.year}-{str(value.year + 1)[-2:]}"
    return f"{value.year - 1}-{str(value.year)[-2:]}"


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


def build_team_lookup():
    lookup = {}
    for team in teams.get_teams():
        lookup[team["abbreviation"]] = normalize_team_name(team["full_name"])
    return lookup


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    features = pd.read_csv(FEATURES_PATH)
    props = pd.read_csv(PROPS_PATH)

    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="coerce")
    props["commence_time"] = pd.to_datetime(props["commence_time"], errors="coerce", utc=True)

    current_season = season_from_date(props["commence_time"].max())
    features["season"] = features["GAME_DATE"].apply(season_from_date)
    features = features[features["season"] == current_season].copy()
    features = features.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)

    features["normalized_player_name"] = features["PLAYER_NAME"].apply(normalize_player_name)
    props["normalized_player_name"] = props["player_name"].apply(normalize_player_name)
    props["home_team_norm"] = props["home_team"].apply(normalize_team_name)
    props["away_team_norm"] = props["away_team"].apply(normalize_team_name)

    latest_features = features.drop_duplicates(subset=["PLAYER_ID"], keep="last").copy()
    latest_features = latest_features.sort_values("GAME_DATE").drop_duplicates(
        subset=["normalized_player_name"], keep="last"
    )

    merged = props.merge(
        latest_features,
        how="left",
        on="normalized_player_name",
        suffixes=("_prop", "_state"),
    )

    team_lookup = build_team_lookup()
    model_feature_columns = [
        column
        for column in features.columns
        if column not in DROP_COLUMNS and column not in {"season", "normalized_player_name"}
    ]

    reconstructed_rows = []
    unmatched_rows = []

    for row in merged.itertuples(index=False):
        if pd.isna(row.PLAYER_ID):
            unmatched_rows.append(row)
            continue

        player_team_norm = team_lookup.get(row.TEAM_ABBREVIATION, "")
        if player_team_norm == row.home_team_norm:
            is_home = 1
            opponent_team = row.away_team
        elif player_team_norm == row.away_team_norm:
            is_home = 0
            opponent_team = row.home_team
        else:
            unmatched_rows.append(row)
            continue

        commence_date = pd.Timestamp(row.commence_time).tz_convert(None).normalize()
        last_game_date = pd.Timestamp(row.GAME_DATE).normalize()
        days_since_last_game = (commence_date - last_game_date).days
        is_back_to_back = 1 if days_since_last_game == 1 else 0

        reconstructed = {
            "PLAYER_ID": row.PLAYER_ID,
            "PLAYER_NAME": row.PLAYER_NAME,
            "GAME_DATE": commence_date,
            "TEAM_ABBREVIATION": row.TEAM_ABBREVIATION,
            "opponent_team": opponent_team,
            "is_home": is_home,
            "points_actual": math.nan,
            "event_id": row.event_id,
            "commence_time": row.commence_time,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "bookmaker": row.bookmaker,
            "player_name": row.player_name,
            "line": row.line,
            "over_price": row.over_price,
            "under_price": row.under_price,
        }

        for column in model_feature_columns:
            if column in {"is_home", "days_since_last_game", "is_back_to_back"}:
                continue
            reconstructed[column] = getattr(row, column)

        reconstructed["days_since_last_game"] = days_since_last_game
        reconstructed["is_back_to_back"] = is_back_to_back
        reconstructed_rows.append(reconstructed)

    reconstructed_df = pd.DataFrame(reconstructed_rows)

    total_props_rows = len(props)
    matched_rows = len(reconstructed_df)
    unmatched_count = total_props_rows - matched_rows

    print(f"total props rows: {total_props_rows}")
    print(f"matched rows: {matched_rows}")
    print(f"unmatched rows: {unmatched_count}")

    X = reconstructed_df[model_feature_columns].copy()
    reconstructed_df["predicted_points"] = model.predict(X)

    reconstructed_df["std"] = pd.to_numeric(reconstructed_df["pts_std_last_10"], errors="coerce")
    reconstructed_df["std"] = (
        reconstructed_df["std"].where(reconstructed_df["std"] > 0, 8.0).fillna(8.0).clip(lower=4.0, upper=15.0)
    )

    reconstructed_df["P_over_model"] = reconstructed_df.apply(
        lambda row: 1 - normal_cdf(row["line"], row["predicted_points"], row["std"]),
        axis=1,
    )
    reconstructed_df["P_under_model"] = 1 - reconstructed_df["P_over_model"]

    reconstructed_df["P_over_raw"] = reconstructed_df["over_price"].apply(american_to_prob)
    reconstructed_df["P_under_raw"] = reconstructed_df["under_price"].apply(american_to_prob)
    reconstructed_df["market_prob_sum"] = reconstructed_df["P_over_raw"] + reconstructed_df["P_under_raw"]
    reconstructed_df["P_over_market"] = reconstructed_df["P_over_raw"] / reconstructed_df["market_prob_sum"]
    reconstructed_df["P_under_market"] = reconstructed_df["P_under_raw"] / reconstructed_df["market_prob_sum"]

    reconstructed_df["edge_over"] = reconstructed_df["P_over_model"] - reconstructed_df["P_over_market"]
    reconstructed_df["edge_under"] = reconstructed_df["P_under_model"] - reconstructed_df["P_under_market"]
    reconstructed_df["best_side"] = reconstructed_df.apply(
        lambda row: "over" if row["edge_over"] > row["edge_under"] else "under",
        axis=1,
    )
    reconstructed_df["best_edge"] = reconstructed_df.apply(
        lambda row: row["edge_over"] if row["best_side"] == "over" else row["edge_under"],
        axis=1,
    )

    output = reconstructed_df[
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

    print("top 20 edges:")
    print(output.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
