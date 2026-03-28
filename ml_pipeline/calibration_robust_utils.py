import math
import pickle

import pandas as pd

DEFAULT_DIFF_BINS = [-float("inf"), -10, -5, -3, -1, 0, 1, 3, 5, 10, float("inf")]
DEFAULT_MIN_BIN_COUNT = 200
DEFAULT_SMOOTHING_STRENGTH = 50.0


def format_bound(value):
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{float(value):.1f}"


def interval_label(left, right):
    return f"({format_bound(left)}, {format_bound(right)}]"


def build_raw_calibration_table(validation_df, diff_column, target_column, bins=None):
    bins = list(bins or DEFAULT_DIFF_BINS)
    working = validation_df[[diff_column, target_column]].copy()
    working[diff_column] = pd.to_numeric(working[diff_column], errors="coerce")
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=[diff_column, target_column]).copy()

    bin_intervals = pd.IntervalIndex.from_breaks(bins, closed="right")
    working["diff_bin"] = pd.cut(
        working[diff_column],
        bins=bins,
        include_lowest=True,
        right=True,
    )

    grouped = (
        working.groupby("diff_bin", observed=False)
        .agg(
            count=(target_column, "size"),
            wins=(target_column, "sum"),
            diff_min=(diff_column, "min"),
            diff_max=(diff_column, "max"),
        )
        .reindex(bin_intervals, fill_value=0)
        .reset_index()
        .rename(columns={"index": "diff_bin"})
    )

    grouped["left"] = [float(interval.left) for interval in grouped["diff_bin"]]
    grouped["right"] = [float(interval.right) for interval in grouped["diff_bin"]]
    grouped["bin_label"] = grouped.apply(
        lambda row: interval_label(row["left"], row["right"]),
        axis=1,
    )
    grouped["empirical_p_over"] = grouped.apply(
        lambda row: float(row["wins"]) / float(row["count"]) if row["count"] else math.nan,
        axis=1,
    )
    grouped.loc[grouped["count"] == 0, ["diff_min", "diff_max"]] = math.nan
    return grouped[
        [
            "bin_label",
            "left",
            "right",
            "count",
            "wins",
            "empirical_p_over",
            "diff_min",
            "diff_max",
        ]
    ].copy()


def _merge_two_blocks(left_block, right_block):
    diff_min = left_block["diff_min"]
    if pd.isna(diff_min) or (pd.notna(right_block["diff_min"]) and right_block["diff_min"] < diff_min):
        diff_min = right_block["diff_min"]

    diff_max = left_block["diff_max"]
    if pd.isna(diff_max) or (pd.notna(right_block["diff_max"]) and right_block["diff_max"] > diff_max):
        diff_max = right_block["diff_max"]

    merged = {
        "left": min(left_block["left"], right_block["left"]),
        "right": max(left_block["right"], right_block["right"]),
        "count": int(left_block["count"]) + int(right_block["count"]),
        "wins": float(left_block["wins"]) + float(right_block["wins"]),
        "diff_min": diff_min,
        "diff_max": diff_max,
        "source_bins": list(left_block["source_bins"]) + list(right_block["source_bins"]),
    }
    merged["bin_label"] = interval_label(merged["left"], merged["right"])
    merged["empirical_p_over"] = (
        float(merged["wins"]) / float(merged["count"]) if merged["count"] else math.nan
    )
    return merged


def merge_sparse_bins(raw_table, min_bin_count=DEFAULT_MIN_BIN_COUNT):
    merged = []
    for row in raw_table.to_dict(orient="records"):
        if int(row["count"]) <= 0:
            continue
        row = dict(row)
        row["source_bins"] = [row["bin_label"]]
        merged.append(row)

    if not merged:
        return pd.DataFrame(
            columns=[
                "bin_label",
                "left",
                "right",
                "count",
                "wins",
                "empirical_p_over",
                "diff_min",
                "diff_max",
                "source_bins",
            ]
        )

    while len(merged) > 1:
        low_indices = [index for index, row in enumerate(merged) if int(row["count"]) < int(min_bin_count)]
        if not low_indices:
            break

        target_index = min(low_indices, key=lambda index: (merged[index]["count"], index))
        if target_index == 0:
            neighbor_index = 1
        elif target_index == len(merged) - 1:
            neighbor_index = target_index - 1
        else:
            left_neighbor = merged[target_index - 1]
            right_neighbor = merged[target_index + 1]
            neighbor_index = (
                target_index - 1
                if int(left_neighbor["count"]) <= int(right_neighbor["count"])
                else target_index + 1
            )

        start = min(target_index, neighbor_index)
        end = max(target_index, neighbor_index)
        replacement = _merge_two_blocks(merged[start], merged[end])
        merged = merged[:start] + [replacement] + merged[end + 1 :]

    merged_df = pd.DataFrame(merged)
    if merged_df.empty:
        return merged_df
    merged_df["source_bin_count"] = merged_df["source_bins"].apply(len)
    return merged_df.reset_index(drop=True)


def weighted_isotonic_increasing(values, weights):
    blocks = []
    for value, weight in zip(values, weights):
        blocks.append(
            {
                "value": float(value),
                "weight": float(weight),
                "length": 1,
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left["weight"] + right["weight"]
            merged_value = (
                (left["value"] * left["weight"]) + (right["value"] * right["weight"])
            ) / merged_weight
            blocks.append(
                {
                    "value": merged_value,
                    "weight": merged_weight,
                    "length": left["length"] + right["length"],
                }
            )

    output = []
    for block in blocks:
        output.extend([block["value"]] * block["length"])
    return output


def build_robust_calibration_payload(
    validation_df,
    diff_column,
    target_column,
    bins=None,
    min_bin_count=DEFAULT_MIN_BIN_COUNT,
    smoothing_strength=DEFAULT_SMOOTHING_STRENGTH,
):
    bins = list(bins or DEFAULT_DIFF_BINS)
    raw_table = build_raw_calibration_table(validation_df, diff_column, target_column, bins=bins)
    merged_table = merge_sparse_bins(raw_table, min_bin_count=min_bin_count)

    total_count = float(raw_table["count"].sum())
    total_wins = float(raw_table["wins"].sum())
    global_prior = total_wins / total_count if total_count else 0.5

    if not merged_table.empty:
        merged_table["smoothed_p_over_pre_isotonic"] = (
            merged_table["wins"] + (smoothing_strength * global_prior)
        ) / (merged_table["count"] + smoothing_strength)
        merged_table["smoothed_p_over"] = weighted_isotonic_increasing(
            merged_table["smoothed_p_over_pre_isotonic"].tolist(),
            merged_table["count"].tolist(),
        )
        merged_table["tail_flag"] = merged_table["count"] < int(min_bin_count)
    else:
        merged_table["smoothed_p_over_pre_isotonic"] = pd.Series(dtype=float)
        merged_table["smoothed_p_over"] = pd.Series(dtype=float)
        merged_table["tail_flag"] = pd.Series(dtype=bool)

    payload = {
        "bins": bins,
        "global_prior": global_prior,
        "min_bin_count": int(min_bin_count),
        "smoothing_strength": float(smoothing_strength),
        "raw_records": raw_table.to_dict(orient="records"),
        "records": merged_table[
            [
                "bin_label",
                "left",
                "right",
                "count",
                "wins",
                "empirical_p_over",
                "smoothed_p_over_pre_isotonic",
                "smoothed_p_over",
                "diff_min",
                "diff_max",
                "source_bins",
                "source_bin_count",
                "tail_flag",
            ]
        ].to_dict(orient="records"),
    }
    return payload, raw_table, merged_table


def save_calibration_payload(path, payload):
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def load_calibration_entries(path):
    with open(path, "rb") as file:
        payload = pickle.load(file)

    records = payload.get("records", [])
    if not records:
        return []

    if "left" in records[0] and "right" in records[0]:
        entries = []
        for record in records:
            probability = record.get("smoothed_p_over", record.get("empirical_p_over"))
            if pd.isna(probability):
                continue
            entries.append(
                {
                    "bin_label": record["bin_label"],
                    "left": float(record["left"]),
                    "right": float(record["right"]),
                    "count": int(record["count"]),
                    "p_over": float(probability),
                    "tail_flag": bool(record.get("tail_flag", False)),
                }
            )
        return entries

    bins = payload["bins"]
    entries = []
    for index, record in enumerate(records):
        probability = record.get("smoothed_p_over", record.get("empirical_p_over"))
        if pd.isna(probability):
            continue
        entries.append(
            {
                "bin_label": record["bin_label"],
                "left": float(bins[index]),
                "right": float(bins[index + 1]),
                "count": int(record["count"]),
                "p_over": float(probability),
                "tail_flag": False,
            }
        )
    return entries


def calibration_distance(diff, left, right):
    if math.isinf(left) and diff <= right:
        return 0.0
    if math.isinf(right) and diff > left:
        return 0.0
    if diff > left and diff <= right:
        return 0.0
    if diff <= left:
        return left - diff
    return diff - right


def lookup_calibrated_probability(diff, entries):
    best_entry = None
    best_distance = None
    for entry in entries:
        distance = calibration_distance(float(diff), entry["left"], entry["right"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entry = entry

    if best_entry is None:
        return 0.5, "", 0, True

    return (
        float(best_entry["p_over"]),
        best_entry["bin_label"],
        int(best_entry["count"]),
        bool(best_entry["tail_flag"]),
    )
