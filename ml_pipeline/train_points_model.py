import math
import os
import pickle

import pandas as pd
from lightgbm import LGBMRegressor

INPUT_PATH = "ml_pipeline/data/features_player_points.csv"
OUTPUT_PATH = "ml_pipeline/models/points_model.pkl"


def main():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    target_column = "points_actual"
    drop_columns = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "GAME_DATE",
        "TEAM_ABBREVIATION",
        "opponent_team",
        "points_actual",
    ]

    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[target_column]

    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = (y_test - predictions).abs().mean()
    rmse = math.sqrt(((y_test - predictions) ** 2).mean())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"train size: {len(train_df)}")
    print(f"test size: {len(test_df)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
