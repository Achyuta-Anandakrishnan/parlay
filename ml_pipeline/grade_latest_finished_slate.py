import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

SNAPSHOT_CONFIGS = [
    {
        "prop_type": "points",
        "path": Path("ml_pipeline/data/tomorrow_ui_snapshot.json"),
        "actual_column": "PTS",
    },
    {
        "prop_type": "rebounds",
        "path": Path("ml_pipeline/data/tomorrow_ui_snapshot_rebounds.json"),
        "actual_column": "REB",
    },
]
ACTUALS_PATH = Path("ml_pipeline/data/player_games_rich.csv")
GRADED_OUTPUT_PATH = Path("ml_pipeline/data/latest_finished_slate_graded.csv")
SUMMARY_OUTPUT_PATH = Path("ml_pipeline/data/latest_finished_slate_summary.json")
LOCAL_TIMEZONE = "America/New_York"
RISK_PER_PROP = 100.0
DEFAULT_ODDS = -110.0
EDGE_BUCKET_BINS = [-math.inf, 0.02, 0.05, 0.10, 0.15, 0.20, math.inf]
EDGE_BUCKET_LABELS = ["<0.02", "0.02-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20+"]
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PLAYER_ALIASES = {
    "nicolas claxton": "nic claxton",
    "carlton carrington": "bub carrington",
}
OUTPUT_COLUMNS = [
    "player_name",
    "prop_type",
    "bookmaker",
    "line",
    "side",
    "predicted_value",
    "edge",
    "actual_value",
    "result",
    "game_date",
    "flags",
    "event_id",
    "home_team",
    "away_team",
    "commence_time",
    "side_price",
    "profit",
]
SNAPSHOT_BASE_COLUMNS = [
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "player_name",
    "line",
    "over_price",
    "under_price",
    "predicted_value_prod",
    "best_side_prod",
    "best_edge_prod",
    "confidence_flag",
]


def normalize_player_name(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = text.split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    text = " ".join(parts)
    return PLAYER_ALIASES.get(text, text)


def american_profit(odds, stake=RISK_PER_PROP):
    if pd.isna(odds):
        odds = DEFAULT_ODDS

    odds = float(odds)
    if odds > 0:
        return stake * (odds / 100.0)
    if odds < 0:
        return stake * (100.0 / abs(odds))
    return 0.0


def save_outputs(graded_df, summary):
    GRADED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graded_df.to_csv(GRADED_OUTPUT_PATH, index=False)
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def load_snapshot_props():
    frames = []

    for config in SNAPSHOT_CONFIGS:
        if not config["path"].exists():
            continue

        with open(config["path"], "r", encoding="utf-8") as file:
            payload = json.load(file)

        props = pd.DataFrame(payload.get("props", []))
        if props.empty:
            continue

        for column in SNAPSHOT_BASE_COLUMNS:
            if column not in props.columns:
                props[column] = pd.NA
        props = props[SNAPSHOT_BASE_COLUMNS].copy()

        props["prop_type"] = config["prop_type"]
        props["actual_stat_column"] = config["actual_column"]
        props["source_snapshot"] = str(config["path"])
        props["snapshot_generated_at"] = payload.get("generated_at", "")
        props["commence_time"] = pd.to_datetime(props["commence_time"], utc=True, errors="coerce")
        props["local_game_date"] = props["commence_time"].dt.tz_convert(LOCAL_TIMEZONE).dt.date
        props["normalized_player_name"] = props["player_name"].apply(normalize_player_name)
        props["predicted_value"] = pd.to_numeric(props["predicted_value_prod"], errors="coerce")
        props["edge"] = pd.to_numeric(props["best_edge_prod"], errors="coerce")
        props["side"] = props["best_side_prod"]
        props["flags"] = props["confidence_flag"].fillna("OK")
        props["side_price"] = props.apply(
            lambda row: row["over_price"] if row["side"] == "over" else row["under_price"],
            axis=1,
        )
        frames.append(props)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_actual_results():
    actuals = pd.read_csv(
        ACTUALS_PATH,
        usecols=["GAME_DATE", "PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "MIN"],
    )
    actuals["GAME_DATE"] = pd.to_datetime(actuals["GAME_DATE"], errors="coerce")
    actuals["game_date"] = actuals["GAME_DATE"].dt.date
    actuals["normalized_player_name"] = actuals["PLAYER_NAME"].apply(normalize_player_name)
    actuals["MIN"] = pd.to_numeric(actuals["MIN"], errors="coerce")
    actuals = (
        actuals.sort_values(["game_date", "normalized_player_name", "MIN"], ascending=[True, True, False])
        .drop_duplicates(subset=["game_date", "normalized_player_name"], keep="first")
        .reset_index(drop=True)
    )
    return actuals


def identify_latest_gradeable_date(props_df, actuals_df):
    if props_df.empty or actuals_df.empty:
        return None

    prop_dates = set(props_df["local_game_date"].dropna())
    actual_dates = set(actuals_df["game_date"].dropna())
    overlapping_dates = sorted(prop_dates & actual_dates)
    if not overlapping_dates:
        return None
    return overlapping_dates[-1]


def grade_result(side, actual_value, line):
    if pd.isna(actual_value) or pd.isna(line):
        return "unmatched"
    if actual_value == line:
        return "push"
    if side == "over":
        return "win" if actual_value > line else "loss"
    return "win" if actual_value < line else "loss"


def summarize_frame(frame, group_column=None):
    if frame.empty:
        if group_column is None:
            return {
                "count": 0,
                "win_rate": None,
                "push_rate": None,
                "roi": None,
            }
        return []

    def build_summary(group_df):
        graded = group_df[group_df["result"].isin(["win", "loss", "push"])].copy()
        total = len(graded)
        wins = int((graded["result"] == "win").sum())
        pushes = int((graded["result"] == "push").sum())
        losses = int((graded["result"] == "loss").sum())
        profit = float(graded["profit"].sum())
        return {
            "count": total,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / total, 4) if total else None,
            "push_rate": round(pushes / total, 4) if total else None,
            "roi": round(profit / (total * RISK_PER_PROP), 4) if total else None,
            "profit": round(profit, 2),
        }

    if group_column is None:
        return build_summary(frame)

    rows = []
    for group_value, group_df in frame.groupby(group_column, dropna=False):
        row = {group_column: group_value}
        row.update(build_summary(group_df))
        rows.append(row)
    rows.sort(key=lambda item: (item[group_column] is None, str(item[group_column])))
    return rows


def build_flag_summary(frame):
    if frame.empty:
        return []

    exploded = frame[["flags", "result", "profit"]].copy()
    exploded["flags"] = exploded["flags"].fillna("OK")
    exploded["flag"] = exploded["flags"].astype(str).str.split("|")
    exploded = exploded.explode("flag")
    exploded["flag"] = exploded["flag"].fillna("OK")
    return summarize_frame(exploded, "flag")


def build_edge_bucket_summary(frame):
    if frame.empty:
        return []

    bucketed = frame.copy()
    bucketed["edge_bucket"] = pd.cut(
        pd.to_numeric(bucketed["edge"], errors="coerce"),
        bins=EDGE_BUCKET_BINS,
        labels=EDGE_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )
    bucketed["edge_bucket"] = bucketed["edge_bucket"].astype(str).replace("nan", "unknown")
    return summarize_frame(bucketed, "edge_bucket")


def build_deduplicated_frame(frame):
    if frame.empty:
        return frame.copy()

    dedupe_keys = ["game_date", "normalized_player_name", "prop_type", "line", "side"]
    return (
        frame.sort_values(["edge", "bookmaker"], ascending=[False, True])
        .drop_duplicates(subset=dedupe_keys, keep="first")
        .reset_index(drop=True)
    )


def print_report(summary):
    print(f"latest_gradeable_game_date: {summary['latest_gradeable_game_date']}")
    print(f"total_graded_props: {summary['grading_summary']['count']}")
    print(f"win_rate: {summary['grading_summary']['win_rate']}")
    print(f"push_rate: {summary['grading_summary']['push_rate']}")
    print(f"roi: {summary['grading_summary']['roi']}")
    print("by_prop_type:")
    for row in summary["by_prop_type"]:
        print(f"  {row['prop_type']}: count={row['count']} win_rate={row['win_rate']} roi={row['roi']}")
    print("by_side:")
    for row in summary["by_side"]:
        print(f"  {row['side']}: count={row['count']} win_rate={row['win_rate']} roi={row['roi']}")
    print("by_bookmaker:")
    for row in summary["by_bookmaker"]:
        print(f"  {row['bookmaker']}: count={row['count']} win_rate={row['win_rate']} roi={row['roi']}")
    print("deduplicated:")
    print(
        f"  count={summary['deduplicated_summary']['count']} "
        f"win_rate={summary['deduplicated_summary']['win_rate']} "
        f"roi={summary['deduplicated_summary']['roi']}"
    )


def main():
    props_df = load_snapshot_props()
    actuals_df = load_actual_results()
    latest_gradeable_date = identify_latest_gradeable_date(props_df, actuals_df)

    if latest_gradeable_date is None:
        empty_frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
        summary = {
            "status": "no_gradeable_slate",
            "latest_gradeable_game_date": None,
            "available_prop_dates": sorted({str(date) for date in props_df["local_game_date"].dropna()}) if not props_df.empty else [],
            "latest_actual_game_date": str(actuals_df["game_date"].max()) if not actuals_df.empty else None,
            "grading_summary": summarize_frame(empty_frame),
            "by_prop_type": [],
            "by_bookmaker": [],
            "by_side": [],
            "by_edge_bucket": [],
            "by_flag": [],
            "deduplicated_summary": summarize_frame(empty_frame),
        }
        save_outputs(empty_frame, summary)
        print("No gradeable saved slate found.")
        print(f"available_prop_dates: {summary['available_prop_dates']}")
        print(f"latest_actual_game_date: {summary['latest_actual_game_date']}")
        return

    slate_props = props_df[props_df["local_game_date"] == latest_gradeable_date].copy()
    slate_actuals = actuals_df[actuals_df["game_date"] == latest_gradeable_date].copy()
    merged = slate_props.merge(
        slate_actuals[["game_date", "normalized_player_name", "PTS", "REB"]],
        left_on=["local_game_date", "normalized_player_name"],
        right_on=["game_date", "normalized_player_name"],
        how="left",
    )
    merged["actual_value"] = merged.apply(
        lambda row: row["PTS"] if row["prop_type"] == "points" else row["REB"],
        axis=1,
    )
    merged["result"] = merged.apply(
        lambda row: grade_result(row["side"], row["actual_value"], row["line"]),
        axis=1,
    )
    merged["side_price"] = pd.to_numeric(merged["side_price"], errors="coerce").fillna(DEFAULT_ODDS)
    merged["profit"] = merged.apply(
        lambda row: american_profit(row["side_price"]) if row["result"] == "win" else 0.0 if row["result"] in {"push", "unmatched"} else -RISK_PER_PROP,
        axis=1,
    )
    merged["game_date"] = merged["local_game_date"].astype(str)

    graded_df = merged[
        [
            "player_name",
            "prop_type",
            "bookmaker",
            "line",
            "side",
            "predicted_value",
            "edge",
            "actual_value",
            "result",
            "game_date",
            "flags",
            "event_id",
            "home_team",
            "away_team",
            "commence_time",
            "side_price",
            "profit",
            "normalized_player_name",
        ]
    ].copy()
    graded_df = graded_df.sort_values(["edge", "bookmaker", "player_name"], ascending=[False, True, True]).reset_index(drop=True)
    output_df = graded_df[OUTPUT_COLUMNS].copy()

    deduped_df = build_deduplicated_frame(graded_df)
    summary = {
        "status": "ok",
        "latest_gradeable_game_date": str(latest_gradeable_date),
        "grading_summary": summarize_frame(graded_df),
        "by_prop_type": summarize_frame(graded_df, "prop_type"),
        "by_bookmaker": summarize_frame(graded_df, "bookmaker"),
        "by_side": summarize_frame(graded_df, "side"),
        "by_edge_bucket": build_edge_bucket_summary(graded_df),
        "by_flag": build_flag_summary(graded_df),
        "deduplicated_summary": summarize_frame(deduped_df),
        "deduplicated_count": int(len(deduped_df)),
    }

    save_outputs(output_df, summary)
    print_report(summary)


if __name__ == "__main__":
    main()
