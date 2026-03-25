import os
import pickle

import pandas as pd

try:
    from ml_pipeline.rebounds_context_utils import get_model_drop_columns, load_context_with_rebounds
except ModuleNotFoundError:
    from rebounds_context_utils import get_model_drop_columns, load_context_with_rebounds

MODEL_PATH = "ml_pipeline/models/rebounds_model_context.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
PLAYER_GAMES_PATH = "ml_pipeline/data/player_games_rich.csv"
OUTPUT_PATH = "ml_pipeline/models/rebounds_model_calibration.pkl"
DIFF_BINS = [-float("inf"), -10, -5, -3, -1, 0, 1, 3, 5, 10, float("inf")]


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    df = load_context_with_rebounds(FEATURES_PATH, PLAYER_GAMES_PATH)
    df = df.dropna(subset=["rebounds_actual"]).copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    validation_df = df.iloc[split_index:].copy()

    model_drop_columns = get_model_drop_columns(validation_df)
    X_validation = validation_df.drop(columns=model_drop_columns)

    validation_df["predicted_rebounds"] = model.predict(X_validation)
    validation_df = validation_df.dropna(subset=["season_avg_rebounds_prior"]).copy()
    validation_df["diff"] = validation_df["predicted_rebounds"] - validation_df["season_avg_rebounds_prior"]
    validation_df["over_hit"] = (
        validation_df["rebounds_actual"] > validation_df["season_avg_rebounds_prior"]
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
    calibration_payload = {"bins": DIFF_BINS, "records": calibration_records}

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
