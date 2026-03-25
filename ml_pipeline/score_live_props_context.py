import math
import pickle
import re
import unicodedata

import pandas as pd
from nba_api.stats.static import teams

MODEL_PATH = "ml_pipeline/models/points_model_context.pkl"
CALIBRATION_PATH = "ml_pipeline/models/points_model_calibration.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
PROPS_PATH = "ml_pipeline/data/upcoming_player_points_props.csv"
OUTPUT_PATH = "ml_pipeline/data/live_scored_props_context.csv"
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
TEAM_CONTEXT_COLUMNS = [column for column in [
    "team_points_last_5",
    "team_points_last_10",
    "team_fga_last_5",
    "team_fg3a_last_5",
    "team_fta_last_5",
    "team_reb_last_5",
    "team_ast_last_5",
    "team_tov_last_5",
    "team_pace_proxy_last_5",
] if column]
OPP_CONTEXT_COLUMNS = [column for column in [
    "opp_points_allowed_last_5",
    "opp_points_allowed_last_10",
    "opp_fga_allowed_last_5",
    "opp_fg3a_allowed_last_5",
    "opp_fta_allowed_last_5",
    "opp_reb_allowed_last_5",
    "opp_ast_allowed_last_5",
    "opp_pace_proxy_last_5",
] if column]


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


def american_to_prob(odds):
    if pd.isna(odds):
        return math.nan

    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return math.nan


def build_team_maps():
    abbr_to_name = {}
    name_to_abbr = {}
    for team in teams.get_teams():
        normalized_name = normalize_team_name(team["full_name"])
        abbr_to_name[team["abbreviation"]] = normalized_name
        name_to_abbr[normalized_name] = team["abbreviation"]
    return abbr_to_name, name_to_abbr


def latest_by_key_prefer_current(df, key_column, current_season):
    sorted_df = df.sort_values(["GAME_DATE", key_column]).reset_index(drop=True)
    current_df = sorted_df[sorted_df["season"] == current_season].copy()

    preferred = current_df.drop_duplicates(subset=[key_column], keep="last")
    if preferred.empty:
        return sorted_df.drop_duplicates(subset=[key_column], keep="last")

    missing_keys = set(sorted_df[key_column]) - set(preferred[key_column])
    if not missing_keys:
        return preferred

    fallback = sorted_df[sorted_df[key_column].isin(missing_keys)].drop_duplicates(subset=[key_column], keep="last")
    return pd.concat([preferred, fallback], ignore_index=True)


def load_calibration_entries():
    with open(CALIBRATION_PATH, "rb") as file:
        payload = pickle.load(file)

    bins = payload["bins"]
    entries = []
    for index, record in enumerate(payload["records"]):
        entries.append(
            {
                "bin_label": record["bin_label"],
                "count": int(record["count"]),
                "empirical_p_over": record["empirical_p_over"],
                "left": bins[index],
                "right": bins[index + 1],
            }
        )
    return entries


def calibration_distance(diff, left, right):
    left_open = not math.isinf(left)

    if math.isinf(left) and diff <= right:
        return 0.0
    if math.isinf(right) and diff > left:
        return 0.0
    if (diff > left or (not left_open and diff >= left)) and diff <= right:
        return 0.0
    if diff <= left:
        return left - diff
    return diff - right


def calibrated_p_over(diff, calibration_entries):
    best_entry = None
    best_distance = None

    for entry in calibration_entries:
        if entry["count"] <= 0 or pd.isna(entry["empirical_p_over"]):
            continue

        distance = calibration_distance(diff, entry["left"], entry["right"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entry = entry

    if best_entry is None:
        return 0.5, ""

    return float(best_entry["empirical_p_over"]), best_entry["bin_label"]


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    calibration_entries = load_calibration_entries()

    features = pd.read_csv(FEATURES_PATH)
    props = pd.read_csv(PROPS_PATH)

    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="coerce")
    props["commence_time"] = pd.to_datetime(props["commence_time"], errors="coerce", utc=True)

    current_season = season_from_date(props["commence_time"].max())
    features["season"] = features["GAME_DATE"].apply(season_from_date)
    features["normalized_player_name"] = features["PLAYER_NAME"].apply(normalize_player_name)

    props["normalized_player_name"] = props["player_name"].apply(normalize_player_name)
    props["home_team_norm"] = props["home_team"].apply(normalize_team_name)
    props["away_team_norm"] = props["away_team"].apply(normalize_team_name)

    player_state = latest_by_key_prefer_current(features, "normalized_player_name", current_season)
    team_context = latest_by_key_prefer_current(features, "TEAM_ABBREVIATION", current_season)
    opponent_context = latest_by_key_prefer_current(features, "opponent_team", current_season)

    player_state_lookup = player_state.set_index("normalized_player_name")
    team_context_lookup = team_context.set_index("TEAM_ABBREVIATION")
    opponent_context_lookup = opponent_context.set_index("opponent_team")

    abbr_to_name, name_to_abbr = build_team_maps()
    model_feature_columns = list(model.feature_name_)

    reconstructed_rows = []
    unmatched_rows = []

    for prop in props.itertuples(index=False):
        if prop.normalized_player_name not in player_state_lookup.index:
            unmatched_rows.append((prop.player_name, "missing_player_state"))
            continue

        state = player_state_lookup.loc[prop.normalized_player_name]
        player_team_abbr = state["TEAM_ABBREVIATION"]
        player_team_norm = abbr_to_name.get(player_team_abbr, "")

        if player_team_norm == prop.home_team_norm:
            is_home = 1
            opponent_team_name = prop.away_team
            opponent_team_abbr = name_to_abbr.get(prop.away_team_norm)
        elif player_team_norm == prop.away_team_norm:
            is_home = 0
            opponent_team_name = prop.home_team
            opponent_team_abbr = name_to_abbr.get(prop.home_team_norm)
        else:
            unmatched_rows.append((prop.player_name, "team_mismatch"))
            continue

        if player_team_abbr not in team_context_lookup.index:
            unmatched_rows.append((prop.player_name, "missing_team_context"))
            continue
        if opponent_team_abbr not in opponent_context_lookup.index:
            unmatched_rows.append((prop.player_name, "missing_opponent_context"))
            continue

        commence_date = pd.Timestamp(prop.commence_time).tz_convert(None).normalize()
        last_game_date = pd.Timestamp(state["GAME_DATE"]).normalize()
        days_since_last_game = (commence_date - last_game_date).days
        is_back_to_back = 1 if days_since_last_game == 1 else 0

        reconstructed = {column: state[column] for column in model_feature_columns if column in state.index}
        reconstructed["is_home"] = is_home
        reconstructed["days_since_last_game"] = days_since_last_game
        reconstructed["is_back_to_back"] = is_back_to_back

        team_row = team_context_lookup.loc[player_team_abbr]
        for column in TEAM_CONTEXT_COLUMNS:
            reconstructed[column] = team_row[column]

        opp_row = opponent_context_lookup.loc[opponent_team_abbr]
        for column in OPP_CONTEXT_COLUMNS:
            reconstructed[column] = opp_row[column]

        reconstructed["player_name"] = prop.player_name
        reconstructed["bookmaker"] = prop.bookmaker
        reconstructed["line"] = prop.line
        reconstructed["over_price"] = prop.over_price
        reconstructed["under_price"] = prop.under_price
        reconstructed["TEAM_ABBREVIATION"] = player_team_abbr
        reconstructed["opponent_team"] = opponent_team_name
        reconstructed["GAME_DATE"] = commence_date
        reconstructed["PLAYER_NAME"] = state["PLAYER_NAME"]
        reconstructed["PLAYER_ID"] = state["PLAYER_ID"]
        reconstructed_rows.append(reconstructed)

    reconstructed_df = pd.DataFrame(reconstructed_rows)

    total_props = len(props)
    matched_rows = len(reconstructed_df)
    unmatched_rows_count = total_props - matched_rows

    print(f"total props: {total_props}")
    print(f"matched rows: {matched_rows}")
    print(f"unmatched rows: {unmatched_rows_count}")

    if reconstructed_df.empty:
        reconstructed_df.to_csv(OUTPUT_PATH, index=False)
        print("top 20 props by best_edge:")
        print("No matched rows.")
        print("edge distribution summary:")
        print("No edges computed.")
        return

    X = reconstructed_df[model_feature_columns].copy()
    reconstructed_df["predicted_points"] = model.predict(X)

    reconstructed_df["std_used"] = pd.to_numeric(reconstructed_df["pts_std_last_10"], errors="coerce")
    reconstructed_df["std_used"] = (
        reconstructed_df["std_used"].where(reconstructed_df["std_used"] > 0, 8.0).fillna(8.0).clip(lower=4.0, upper=15.0)
    )

    reconstructed_df["calibration_diff"] = reconstructed_df["predicted_points"] - reconstructed_df["line"]
    calibration_pairs = reconstructed_df["calibration_diff"].apply(
        lambda diff: calibrated_p_over(diff, calibration_entries)
    )
    reconstructed_df["p_over_model"] = calibration_pairs.apply(lambda pair: pair[0])
    reconstructed_df["calibration_bin"] = calibration_pairs.apply(lambda pair: pair[1])
    reconstructed_df["p_under_model"] = 1 - reconstructed_df["p_over_model"]

    reconstructed_df["p_over_raw"] = reconstructed_df["over_price"].apply(american_to_prob)
    reconstructed_df["p_under_raw"] = reconstructed_df["under_price"].apply(american_to_prob)
    reconstructed_df["market_prob_sum"] = reconstructed_df["p_over_raw"] + reconstructed_df["p_under_raw"]
    reconstructed_df["p_over_market_novig"] = reconstructed_df["p_over_raw"] / reconstructed_df["market_prob_sum"]
    reconstructed_df["p_under_market_novig"] = reconstructed_df["p_under_raw"] / reconstructed_df["market_prob_sum"]

    reconstructed_df["edge_over"] = reconstructed_df["p_over_model"] - reconstructed_df["p_over_market_novig"]
    reconstructed_df["edge_under"] = reconstructed_df["p_under_model"] - reconstructed_df["p_under_market_novig"]
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
            "std_used",
            "calibration_bin",
            "p_over_model",
            "p_under_model",
            "p_over_market_novig",
            "p_under_market_novig",
            "edge_over",
            "edge_under",
            "best_side",
            "best_edge",
        ]
    ].copy()
    output = output.sort_values("best_edge", ascending=False).reset_index(drop=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("top 20 props by best_edge:")
    print(output.head(20).to_string(index=False))
    print("edge distribution summary:")
    print(output["best_edge"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())


if __name__ == "__main__":
    main()
