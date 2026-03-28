import math
import re
import unicodedata

import pandas as pd

LOW_SAMPLE_THRESHOLD = 8
EXTREME_EDGE_THRESHOLD = 0.18
DEFAULT_MIN_EDGE = 0.06
DEFAULT_MAX_PLAYS = 20
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PLAYER_ALIASES = {
    "nicolas claxton": "nic claxton",
    "carlton carrington": "bub carrington",
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


def season_from_date(value):
    value = pd.Timestamp(value)
    if value.month >= 10:
        return f"{value.year}-{str(value.year + 1)[-2:]}"
    return f"{value.year - 1}-{str(value.year)[-2:]}"


def latest_by_key_prefer_current(df, key_column, current_season):
    sorted_df = df.sort_values(["GAME_DATE", key_column]).reset_index(drop=True)
    current_df = sorted_df[sorted_df["season"] == current_season].copy()
    preferred = current_df.drop_duplicates(subset=[key_column], keep="last")
    if preferred.empty:
        return sorted_df.drop_duplicates(subset=[key_column], keep="last")

    missing_keys = set(sorted_df[key_column]) - set(preferred[key_column])
    if not missing_keys:
        return preferred

    fallback = (
        sorted_df[sorted_df[key_column].isin(missing_keys)]
        .drop_duplicates(subset=[key_column], keep="last")
    )
    return pd.concat([preferred, fallback], ignore_index=True)


def american_profit(odds, stake=100.0):
    if pd.isna(odds):
        return math.nan
    odds = float(odds)
    if odds > 0:
        return stake * (odds / 100.0)
    if odds < 0:
        return stake * (100.0 / abs(odds))
    return 0.0


def role_bucket(min_last_5, min_last_10):
    minutes = min_last_5
    if pd.isna(minutes):
        minutes = min_last_10
    if pd.isna(minutes):
        return "unknown"
    minutes = float(minutes)
    if minutes < 12:
        return "<12m"
    if minutes < 24:
        return "12-24m"
    if minutes < 32:
        return "24-32m"
    return "32m+"


def line_bucket(prop_type, line):
    if pd.isna(line):
        return "unknown"
    line = float(line)
    if prop_type == "points":
        if line < 10:
            return "<10"
        if line < 15:
            return "10-14.5"
        if line < 20:
            return "15-19.5"
        if line < 25:
            return "20-24.5"
        return "25+"

    if line < 4.5:
        return "<4.5"
    if line < 7:
        return "4.5-6.5"
    if line < 10:
        return "7-9.5"
    return "10+"


def edge_bucket(edge):
    if pd.isna(edge):
        return "unknown"
    edge = float(edge)
    if edge < 0.06:
        return "<0.06"
    if edge < 0.10:
        return "0.06-0.10"
    if edge < 0.15:
        return "0.10-0.15"
    if edge < 0.20:
        return "0.15-0.20"
    return "0.20+"


def attach_selection_flags(frame):
    if frame.empty:
        frame["flags"] = [[] for _ in range(len(frame))]
        frame["confidence_flag"] = []
        return frame

    def build_flags(row):
        flags = []
        if pd.notna(row.get("games_played_prior")) and float(row["games_played_prior"]) < LOW_SAMPLE_THRESHOLD:
            flags.append("LOW_SAMPLE")
        if bool(row.get("tail_flag", False)):
            flags.append("TAIL_CALIBRATION")
        if pd.notna(row.get("best_edge")) and float(row["best_edge"]) > EXTREME_EDGE_THRESHOLD:
            flags.append("EXTREME_EDGE")
        return flags

    frame["flags"] = frame.apply(build_flags, axis=1)
    frame["confidence_flag"] = frame["flags"].apply(lambda flags: "|".join(flags) if flags else "OK")
    return frame


def dedupe_best_price(frame):
    if frame.empty:
        return frame.copy(), {
            "raw_rows": 0,
            "deduped_rows": 0,
            "multi_book_opinions": 0,
            "avg_books_per_opinion": 0.0,
        }

    working = frame.copy()
    working["side_price"] = working.apply(
        lambda row: row["over_price"] if row["best_side"] == "over" else row["under_price"],
        axis=1,
    )
    working["side_payout_per_100"] = working["side_price"].apply(american_profit)
    opinion_keys = ["normalized_player_name", "prop_type", "line", "best_side"]

    group_summary = (
        working.groupby(opinion_keys, dropna=False)
        .agg(
            row_count=("bookmaker", "size"),
            book_count=("bookmaker", "nunique"),
            best_available_payout_per_100=("side_payout_per_100", "max"),
            mean_available_payout_per_100=("side_payout_per_100", "mean"),
        )
        .reset_index()
    )

    selected = (
        working.sort_values(
            ["side_payout_per_100", "best_edge", "bookmaker"],
            ascending=[False, False, True],
        )
        .drop_duplicates(subset=opinion_keys, keep="first")
        .reset_index(drop=True)
    )
    selected = selected.merge(group_summary, on=opinion_keys, how="left")
    selected["price_advantage_vs_mean_per_100"] = (
        selected["side_payout_per_100"] - selected["mean_available_payout_per_100"]
    )

    summary = {
        "raw_rows": int(len(working)),
        "deduped_rows": int(len(selected)),
        "multi_book_opinions": int((group_summary["book_count"] > 1).sum()),
        "avg_books_per_opinion": round(float(group_summary["book_count"].mean()), 4),
        "selected_bookmakers": (
            selected["bookmaker"].value_counts().sort_index().to_dict()
        ),
    }
    return selected, summary


def summarize_counts(frame, group_column):
    if frame.empty:
        return []

    rows = []
    for value, group in frame.groupby(group_column, dropna=False):
        rows.append(
            {
                group_column: value,
                "count": int(len(group)),
                "avg_edge": round(float(group["best_edge"].mean()), 4),
                "median_edge": round(float(group["best_edge"].median()), 4),
                "max_edge": round(float(group["best_edge"].max()), 4),
            }
        )
    rows.sort(key=lambda item: (item["count"] * -1, str(item[group_column])))
    return rows
