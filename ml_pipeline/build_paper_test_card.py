import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from ml_pipeline.calibration_robust_utils import load_calibration_entries, lookup_calibrated_probability
    from ml_pipeline.selection_evaluation_utils import (
        DEFAULT_MAX_PLAYS,
        DEFAULT_MIN_EDGE,
        attach_selection_flags,
        dedupe_best_price,
        edge_bucket,
        latest_by_key_prefer_current,
        line_bucket,
        normalize_player_name,
        role_bucket,
        season_from_date,
        summarize_counts,
    )
except ModuleNotFoundError:
    from calibration_robust_utils import load_calibration_entries, lookup_calibrated_probability
    from selection_evaluation_utils import (
        DEFAULT_MAX_PLAYS,
        DEFAULT_MIN_EDGE,
        attach_selection_flags,
        dedupe_best_price,
        edge_bucket,
        latest_by_key_prefer_current,
        line_bucket,
        normalize_player_name,
        role_bucket,
        season_from_date,
        summarize_counts,
    )

FEATURES_PATH = Path("ml_pipeline/data/features_player_points_context.csv")
POINTS_RAW_PATH = Path("ml_pipeline/data/upcoming_player_points_props.csv")
POINTS_SCORED_PATH = Path("ml_pipeline/data/live_scored_props_context.csv")
REBOUNDS_RAW_PATH = Path("ml_pipeline/data/upcoming_player_rebounds_props.csv")
REBOUNDS_SCORED_PATH = Path("ml_pipeline/data/live_scored_rebounds_props.csv")
POINTS_CALIBRATION_PATH = Path("ml_pipeline/models/points_model_calibration_robust.pkl")
REBOUNDS_CALIBRATION_PATH = Path("ml_pipeline/models/rebounds_model_calibration_robust.pkl")

RAW_OUTPUT_PATH = Path("ml_pipeline/data/candidate_pool_robust_raw.csv")
DEDUPED_OUTPUT_PATH = Path("ml_pipeline/data/candidate_pool_robust_deduped.csv")
CARD_OUTPUT_PATH = Path("ml_pipeline/data/paper_test_card_robust.csv")
SUMMARY_OUTPUT_PATH = Path("ml_pipeline/data/paper_test_summary_robust.json")


def load_latest_features():
    features = pd.read_csv(FEATURES_PATH)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="coerce")
    features["season"] = features["GAME_DATE"].apply(season_from_date)
    features["normalized_player_name"] = features["PLAYER_NAME"].apply(normalize_player_name)
    current_season = season_from_date(features["GAME_DATE"].max())
    latest = latest_by_key_prefer_current(features, "normalized_player_name", current_season)
    keep_columns = [
        "normalized_player_name",
        "PLAYER_NAME",
        "games_played_prior",
        "min_last_5",
        "min_last_10",
        "season_avg_minutes_prior",
    ]
    return latest[keep_columns].copy()


def load_prop_type_frame(prop_type, raw_path, scored_path, calibration_path, latest_features, now_utc):
    raw = pd.read_csv(raw_path)
    scored = pd.read_csv(scored_path)
    calibration_entries = load_calibration_entries(calibration_path)

    if raw.empty or scored.empty:
        return pd.DataFrame()

    raw["commence_time"] = pd.to_datetime(raw["commence_time"], utc=True, errors="coerce")
    raw = raw[raw["commence_time"] >= now_utc].copy()
    if raw.empty:
        return pd.DataFrame()

    merge_columns = ["player_name", "bookmaker", "line"]
    frame = raw.merge(scored, on=merge_columns, how="inner")
    if frame.empty:
        return frame

    prediction_column = "predicted_points" if prop_type == "points" else "predicted_rebounds"
    frame["predicted_value"] = pd.to_numeric(frame[prediction_column], errors="coerce")
    frame["normalized_player_name"] = frame["player_name"].apply(normalize_player_name)
    frame = frame.merge(latest_features, on="normalized_player_name", how="left", suffixes=("", "_state"))
    frame["prop_type"] = prop_type
    frame["diff"] = frame["predicted_value"] - pd.to_numeric(frame["line"], errors="coerce")

    lookup_results = frame["diff"].apply(lambda diff: lookup_calibrated_probability(diff, calibration_entries))
    frame["p_over_model"] = lookup_results.apply(lambda item: item[0])
    frame["calibration_bin_robust"] = lookup_results.apply(lambda item: item[1])
    frame["calibration_count_robust"] = lookup_results.apply(lambda item: item[2])
    frame["tail_flag"] = lookup_results.apply(lambda item: item[3])
    frame["p_under_model"] = 1 - frame["p_over_model"]
    frame["edge_over"] = frame["p_over_model"] - pd.to_numeric(frame["p_over_market_novig"], errors="coerce")
    frame["edge_under"] = frame["p_under_model"] - pd.to_numeric(frame["p_under_market_novig"], errors="coerce")
    frame["best_side"] = frame.apply(
        lambda row: "over" if row["edge_over"] > row["edge_under"] else "under",
        axis=1,
    )
    frame["best_edge"] = frame.apply(
        lambda row: row["edge_over"] if row["best_side"] == "over" else row["edge_under"],
        axis=1,
    )
    frame["line_bucket"] = frame["line"].apply(lambda line: line_bucket(prop_type, line))
    frame["role_bucket"] = frame.apply(
        lambda row: role_bucket(row.get("min_last_5"), row.get("min_last_10")),
        axis=1,
    )
    frame["edge_bucket"] = frame["best_edge"].apply(edge_bucket)
    return frame


def build_breakdowns(frame):
    return {
        "by_prop_type": summarize_counts(frame, "prop_type"),
        "by_side": summarize_counts(frame, "best_side"),
        "by_edge_bucket": summarize_counts(frame, "edge_bucket"),
        "by_line_bucket": summarize_counts(frame, "line_bucket"),
        "by_role_bucket": summarize_counts(frame, "role_bucket"),
    }


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    parser.add_argument("--max-plays", type=int, default=DEFAULT_MAX_PLAYS)
    parser.add_argument("--exclude-extreme-edge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    now_utc = pd.Timestamp.utcnow()
    latest_features = load_latest_features()

    points = load_prop_type_frame(
        "points",
        POINTS_RAW_PATH,
        POINTS_SCORED_PATH,
        POINTS_CALIBRATION_PATH,
        latest_features,
        now_utc,
    )
    rebounds = load_prop_type_frame(
        "rebounds",
        REBOUNDS_RAW_PATH,
        REBOUNDS_SCORED_PATH,
        REBOUNDS_CALIBRATION_PATH,
        latest_features,
        now_utc,
    )

    raw = pd.concat([points, rebounds], ignore_index=True)
    raw = attach_selection_flags(raw)
    raw = raw.sort_values(["best_edge", "commence_time"], ascending=[False, True]).reset_index(drop=True)

    deduped, best_price_summary = dedupe_best_price(raw)
    deduped = attach_selection_flags(deduped)

    card = deduped.copy()
    card = card[card["best_edge"] >= float(args.min_edge)].copy()
    card = card[~card["flags"].apply(lambda flags: "TAIL_CALIBRATION" in flags)].copy()
    card = card[~card["flags"].apply(lambda flags: "LOW_SAMPLE" in flags)].copy()
    if args.exclude_extreme_edge:
        card = card[~card["flags"].apply(lambda flags: "EXTREME_EDGE" in flags)].copy()
    card = card.sort_values(["best_edge", "commence_time"], ascending=[False, True]).head(int(args.max_plays))

    output_columns = [
        "prop_type",
        "player_name",
        "home_team",
        "away_team",
        "bookmaker",
        "line",
        "best_side",
        "best_edge",
        "predicted_value",
        "over_price",
        "under_price",
        "p_over_model",
        "p_under_model",
        "p_over_market_novig",
        "p_under_market_novig",
        "calibration_bin_robust",
        "calibration_count_robust",
        "games_played_prior",
        "min_last_5",
        "min_last_10",
        "role_bucket",
        "edge_bucket",
        "line_bucket",
        "confidence_flag",
        "flags",
        "side_price",
        "side_payout_per_100",
        "row_count",
        "book_count",
    ]
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT_PATH, index=False)
    deduped.to_csv(DEDUPED_OUTPUT_PATH, index=False)
    card.to_csv(CARD_OUTPUT_PATH, index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config": {
            "min_edge": float(args.min_edge),
            "max_plays": int(args.max_plays),
            "exclude_tail_calibration": True,
            "exclude_low_sample": True,
            "exclude_extreme_edge": bool(args.exclude_extreme_edge),
        },
        "counts": {
            "raw_rows": int(len(raw)),
            "deduped_rows": int(len(deduped)),
            "paper_card_rows": int(len(card)),
        },
        "best_price_summary": best_price_summary,
        "raw_breakdowns": build_breakdowns(raw),
        "deduped_breakdowns": build_breakdowns(deduped),
        "paper_card_breakdowns": build_breakdowns(card),
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(summary), file, indent=2)

    print(f"raw_rows: {len(raw)}")
    print(f"deduped_rows: {len(deduped)}")
    print(f"paper_card_rows: {len(card)}")
    print(f"multi_book_opinions: {best_price_summary['multi_book_opinions']}")
    print(f"avg_books_per_opinion: {best_price_summary['avg_books_per_opinion']}")
    print("paper card top plays:")
    if card.empty:
        print("none")
    else:
        print(card[output_columns[:10]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
