import os
import pickle

import pandas as pd

try:
    from ml_pipeline.calibration_robust_utils import build_robust_calibration_payload, save_calibration_payload
    from ml_pipeline.rebounds_context_utils import get_model_drop_columns, load_context_with_rebounds
except ModuleNotFoundError:
    from calibration_robust_utils import build_robust_calibration_payload, save_calibration_payload
    from rebounds_context_utils import get_model_drop_columns, load_context_with_rebounds

MODEL_PATH = "ml_pipeline/models/rebounds_model_context.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
PLAYER_GAMES_PATH = "ml_pipeline/data/player_games_rich.csv"
OUTPUT_PATH = "ml_pipeline/models/rebounds_model_calibration_robust.pkl"


def main():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    df = load_context_with_rebounds(FEATURES_PATH, PLAYER_GAMES_PATH)
    df = df.dropna(subset=["rebounds_actual"]).copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    validation_df = df.iloc[split_index:].copy()
    validation_df = validation_df.dropna(subset=["season_avg_rebounds_prior"]).copy()

    X_validation = validation_df.drop(columns=get_model_drop_columns(validation_df))
    validation_df["predicted_rebounds"] = model.predict(X_validation)
    validation_df["diff"] = (
        validation_df["predicted_rebounds"] - validation_df["season_avg_rebounds_prior"]
    )
    validation_df["over_hit"] = (
        validation_df["rebounds_actual"] > validation_df["season_avg_rebounds_prior"]
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
