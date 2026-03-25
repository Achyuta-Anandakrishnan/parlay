import os

import pandas as pd

INPUT_PATH = "ml_pipeline/data/player_games_rich.csv"
OUTPUT_PATH = "ml_pipeline/data/features_player_points_rich.csv"


def season_from_date(value):
    value = pd.Timestamp(value)
    if value.month >= 10:
        return f"{value.year}-{str(value.year + 1)[-2:]}"
    return f"{value.year - 1}-{str(value.year)[-2:]}"


def add_rolling_features(frame):
    player_group = frame.groupby("PLAYER_ID", sort=False)

    frame["pts_last_3"] = player_group["points_actual"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    frame["pts_last_5"] = player_group["points_actual"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["pts_last_10"] = player_group["points_actual"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )

    frame["min_last_3"] = player_group["minutes_actual"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    frame["min_last_5"] = player_group["minutes_actual"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["min_last_10"] = player_group["minutes_actual"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )

    frame["fga_last_5"] = player_group["FGA"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["fga_last_10"] = player_group["FGA"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    frame["fg3a_last_5"] = player_group["FG3A"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["fta_last_5"] = player_group["FTA"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    frame["pts_std_last_10"] = player_group["points_actual"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=2).std()
    )

    frame["reb_last_5"] = player_group["REB"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["reb_last_10"] = player_group["REB"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    frame["ast_last_5"] = player_group["AST"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["ast_last_10"] = player_group["AST"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    frame["fg3m_last_5"] = player_group["FG3M"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["fg3m_last_10"] = player_group["FG3M"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    frame["tov_last_5"] = player_group["TOV"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    frame["pf_last_5"] = player_group["PF"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    frame["fg_pct_last_10"] = player_group["FG_PCT"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    frame["fg3_pct_last_10"] = player_group["FG3_PCT"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    frame["ft_pct_last_10"] = player_group["FT_PCT"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )

    frame["usage_proxy_last_5"] = frame["fga_last_5"] + (0.44 * frame["fta_last_5"])
    frame["usage_proxy_last_10"] = frame["fga_last_10"] + (
        0.44
        * player_group["FTA"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=1).mean())
    )
    frame["shot_volume_trend"] = frame["fga_last_5"] - frame["fga_last_10"]
    frame["min_trend"] = frame["min_last_5"] - frame["min_last_10"]

    return frame


def add_season_features(frame):
    season_group = frame.groupby(["PLAYER_ID", "season"], sort=False)

    frame["games_played_prior"] = season_group.cumcount()
    frame["season_avg_points_prior"] = season_group["points_actual"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    frame["season_avg_minutes_prior"] = season_group["minutes_actual"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return frame


def add_rest_features(frame):
    player_group = frame.groupby("PLAYER_ID", sort=False)

    frame["days_since_last_game"] = player_group["GAME_DATE"].diff().dt.days
    frame["is_back_to_back"] = (frame["days_since_last_game"] == 1).astype(int)

    return frame


def main():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    df = df.rename(columns={"PTS": "points_actual", "MIN": "minutes_actual"})
    numeric_columns = [
        "minutes_actual",
        "points_actual",
        "FGA",
        "FG3A",
        "FTA",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "FGM",
        "FG_PCT",
        "FG3M",
        "FG3_PCT",
        "FTM",
        "FT_PCT",
        "PLUS_MINUS",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["season"] = df["GAME_DATE"].apply(season_from_date)

    df = add_rolling_features(df)
    df = add_season_features(df)
    df = add_rest_features(df)

    total_rows_before_filtering = len(df)

    keep_columns = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "GAME_DATE",
        "TEAM_ABBREVIATION",
        "opponent_team",
        "is_home",
        "pts_last_3",
        "pts_last_5",
        "pts_last_10",
        "min_last_3",
        "min_last_5",
        "min_last_10",
        "fga_last_5",
        "fg3a_last_5",
        "fta_last_5",
        "pts_std_last_10",
        "reb_last_5",
        "reb_last_10",
        "ast_last_5",
        "ast_last_10",
        "fg3m_last_5",
        "fg3m_last_10",
        "tov_last_5",
        "pf_last_5",
        "fg_pct_last_10",
        "fg3_pct_last_10",
        "ft_pct_last_10",
        "usage_proxy_last_5",
        "usage_proxy_last_10",
        "shot_volume_trend",
        "min_trend",
        "games_played_prior",
        "season_avg_points_prior",
        "season_avg_minutes_prior",
        "days_since_last_game",
        "is_back_to_back",
        "points_actual",
    ]

    features = df.loc[:, keep_columns].copy()
    features = features[features["games_played_prior"] >= 5].copy()
    features = features.dropna().reset_index(drop=True)
    features["GAME_DATE"] = features["GAME_DATE"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"rows before filtering: {total_rows_before_filtering}")
    print(f"rows after filtering: {len(features)}")
    print(f"number of players: {features['PLAYER_ID'].nunique()}")


if __name__ == "__main__":
    main()
