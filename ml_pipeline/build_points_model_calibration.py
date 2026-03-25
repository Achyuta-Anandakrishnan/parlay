import os
import pickle

import pandas as pd

MODEL_PATH = "ml_pipeline/models/points_model_context.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
OUTPUT_PATH = "ml_pipeline/models/points_model_calibration.pkl"
DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "points_actual",
]
DIFF_BINS = [-float("inf"), -10, -5, -3, -1, 0, 1, 3, 5, 10, float("inf")]


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
    model_drop_columns = DROP_COLUMNS + additional_drop_columns

    X_validation = validation_df.drop(columns=model_drop_columns)
    validation_df["predicted_points"] = model.predict(X_validation)
    validation_df["diff"] = validation_df["predicted_points"] - validation_df["season_avg_points_prior"]
    validation_df["over_hit"] = (
        validation_df["points_actual"] > validation_df["season_avg_points_prior"]
    ).astype(int)

    validation_df["diff_bin"] = pd.cut(
        validation_df["diff"],
        bins=DIFF_BINS,
        include_lowest=True,
        right=True,
    )

    calibration = (
        validation_df.groupby("diff_bin", observed=False)
        .agg(
            count=("over_hit", "size"),
            empirical_p_over=("over_hit", "mean"),
            diff_min=("diff", "min"),
            diff_max=("diff", "max"),
        )
        .reset_index()
    )

    calibration["bin_label"] = calibration["diff_bin"].astype(str)
    calibration_records = calibration[
        ["bin_label", "count", "empirical_p_over", "diff_min", "diff_max"]
    ].to_dict(orient="records")

    calibration_payload = {
        "bins": DIFF_BINS,
        "records": calibration_records,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as file:
        pickle.dump(calibration_payload, file)

    print(
        calibration[
            ["bin_label", "count", "empirical_p_over", "diff_min", "diff_max"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
