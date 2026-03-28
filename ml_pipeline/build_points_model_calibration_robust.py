import os
import pickle

import pandas as pd

try:
    from ml_pipeline.calibration_robust_utils import build_robust_calibration_payload, save_calibration_payload
except ModuleNotFoundError:
    from calibration_robust_utils import build_robust_calibration_payload, save_calibration_payload

MODEL_PATH = "ml_pipeline/models/points_model_context.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
OUTPUT_PATH = "ml_pipeline/models/points_model_calibration_robust.pkl"
DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "points_actual",
]


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    df = pd.read_csv(FEATURES_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    validation_df = df.iloc[split_index:].copy()

    additional_drop_columns = [
        column
        for column in validation_df.columns
        if column not in DROP_COLUMNS and validation_df[column].dtype == "object"
    ]
    X_validation = validation_df.drop(columns=DROP_COLUMNS + additional_drop_columns)
    validation_df["predicted_points"] = model.predict(X_validation)
    validation_df["diff"] = validation_df["predicted_points"] - validation_df["season_avg_points_prior"]
    validation_df["over_hit"] = (
        validation_df["points_actual"] > validation_df["season_avg_points_prior"]
    ).astype(int)

    payload, raw_table, merged_table = build_robust_calibration_payload(
        validation_df=validation_df,
        diff_column="diff",
        target_column="over_hit",
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_calibration_payload(OUTPUT_PATH, payload)

    print("raw bins:")
    print(raw_table[["bin_label", "count", "empirical_p_over"]].to_string(index=False))
    print("merged bins:")
    print(
        merged_table[
            [
                "bin_label",
                "count",
                "empirical_p_over",
                "smoothed_p_over",
                "source_bin_count",
                "tail_flag",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
