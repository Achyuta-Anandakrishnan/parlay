import math
import os
import pickle

import pandas as pd
from lightgbm import LGBMRegressor

try:
    from ml_pipeline.rebounds_context_utils import get_model_drop_columns, load_context_with_rebounds
except ModuleNotFoundError:
    from rebounds_context_utils import get_model_drop_columns, load_context_with_rebounds

FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
PLAYER_GAMES_PATH = "ml_pipeline/data/player_games_rich.csv"
OUTPUT_PATH = "ml_pipeline/models/rebounds_model_context.pkl"


def main():
    df = load_context_with_rebounds(FEATURES_PATH, PLAYER_GAMES_PATH)
    df = df.dropna(subset=["rebounds_actual"]).copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    model_drop_columns = get_model_drop_columns(df)

    X_train = train_df.drop(columns=model_drop_columns)
    y_train = train_df["rebounds_actual"]
    X_test = test_df.drop(columns=model_drop_columns)
    y_test = test_df["rebounds_actual"]

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

    feature_importance = (
        pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"train size: {len(train_df)}")
    print(f"test size: {len(test_df)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("top 20 feature importances:")
    print(feature_importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
