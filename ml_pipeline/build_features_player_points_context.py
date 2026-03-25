import os

import pandas as pd

PLAYER_GAMES_PATH = "ml_pipeline/data/player_games_rich.csv"
PLAYER_FEATURES_PATH = "ml_pipeline/data/features_player_points_rich.csv"
OUTPUT_PATH = "ml_pipeline/data/features_player_points_context.csv"

TEAM_AGG_COLUMNS = ["PTS", "FGA", "FG3A", "FTA", "REB", "AST", "TOV"]
NEW_CONTEXT_COLUMNS = [
    "team_points_last_5",
    "team_points_last_10",
    "team_fga_last_5",
    "team_fg3a_last_5",
    "team_fta_last_5",
    "team_reb_last_5",
    "team_ast_last_5",
    "team_tov_last_5",
    "team_pace_proxy_last_5",
    "opp_points_allowed_last_5",
    "opp_points_allowed_last_10",
    "opp_fga_allowed_last_5",
    "opp_fg3a_allowed_last_5",
    "opp_fta_allowed_last_5",
    "opp_reb_allowed_last_5",
    "opp_ast_allowed_last_5",
    "opp_pace_proxy_last_5",
]


def add_team_rolling_context(team_games):
    team_games = team_games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    team_group = team_games.groupby("TEAM_ABBREVIATION", sort=False)

    team_games["team_points_last_5"] = team_group["PTS"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_games["team_points_last_10"] = team_group["PTS"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    team_games["team_fga_last_5"] = team_group["FGA"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_games["team_fg3a_last_5"] = team_group["FG3A"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_games["team_fta_last_5"] = team_group["FTA"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_games["team_reb_last_5"] = team_group["REB"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_games["team_ast_last_5"] = team_group["AST"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    team_games["team_tov_last_5"] = team_group["TOV"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    team_games["team_pace_proxy"] = team_games["FGA"] + (0.44 * team_games["FTA"]) + team_games["TOV"]
    team_games["team_pace_proxy_last_5"] = team_group["team_pace_proxy"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    return team_games[
        [
            "GAME_DATE",
            "TEAM_ABBREVIATION",
            "team_points_last_5",
            "team_points_last_10",
            "team_fga_last_5",
            "team_fg3a_last_5",
            "team_fta_last_5",
            "team_reb_last_5",
            "team_ast_last_5",
            "team_tov_last_5",
            "team_pace_proxy_last_5",
        ]
    ].copy()


def add_opponent_allowed_context(team_games):
    opponent_games = team_games[
        ["GAME_DATE", "opponent_team", "PTS", "FGA", "FG3A", "FTA", "REB", "AST", "team_pace_proxy"]
    ].copy()
    opponent_games = opponent_games.rename(
        columns={
            "opponent_team": "TEAM_ABBREVIATION",
            "PTS": "points_allowed",
            "FGA": "fga_allowed",
            "FG3A": "fg3a_allowed",
            "FTA": "fta_allowed",
            "REB": "reb_allowed",
            "AST": "ast_allowed",
            "team_pace_proxy": "opp_pace_proxy_source",
        }
    )
    opponent_games = opponent_games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    opponent_group = opponent_games.groupby("TEAM_ABBREVIATION", sort=False)

    opponent_games["opp_points_allowed_last_5"] = opponent_group["points_allowed"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    opponent_games["opp_points_allowed_last_10"] = opponent_group["points_allowed"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    opponent_games["opp_fga_allowed_last_5"] = opponent_group["fga_allowed"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    opponent_games["opp_fg3a_allowed_last_5"] = opponent_group["fg3a_allowed"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    opponent_games["opp_fta_allowed_last_5"] = opponent_group["fta_allowed"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    opponent_games["opp_reb_allowed_last_5"] = opponent_group["reb_allowed"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    opponent_games["opp_ast_allowed_last_5"] = opponent_group["ast_allowed"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    opponent_games["opp_pace_proxy_last_5"] = opponent_group["opp_pace_proxy_source"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    return opponent_games[
        [
            "GAME_DATE",
            "TEAM_ABBREVIATION",
            "opp_points_allowed_last_5",
            "opp_points_allowed_last_10",
            "opp_fga_allowed_last_5",
            "opp_fg3a_allowed_last_5",
            "opp_fta_allowed_last_5",
            "opp_reb_allowed_last_5",
            "opp_ast_allowed_last_5",
            "opp_pace_proxy_last_5",
        ]
    ].copy()


def main():
    player_games = pd.read_csv(PLAYER_GAMES_PATH)
    player_features = pd.read_csv(PLAYER_FEATURES_PATH)

    player_games["GAME_DATE"] = pd.to_datetime(player_games["GAME_DATE"], errors="coerce")
    player_features["GAME_DATE"] = pd.to_datetime(player_features["GAME_DATE"], errors="coerce")

    team_games = (
        player_games.groupby(["GAME_DATE", "TEAM_ABBREVIATION", "opponent_team"], as_index=False)[TEAM_AGG_COLUMNS]
        .sum()
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        .reset_index(drop=True)
    )

    team_games["team_pace_proxy"] = team_games["FGA"] + (0.44 * team_games["FTA"]) + team_games["TOV"]

    team_context = add_team_rolling_context(team_games)
    opponent_context = add_opponent_allowed_context(team_games)

    row_count_before_merge = len(player_features)

    merged = player_features.merge(
        team_context,
        how="left",
        on=["GAME_DATE", "TEAM_ABBREVIATION"],
        validate="m:1",
    )
    merged = merged.merge(
        opponent_context,
        how="left",
        left_on=["GAME_DATE", "opponent_team"],
        right_on=["GAME_DATE", "TEAM_ABBREVIATION"],
        suffixes=("", "_opp_ctx"),
        validate="m:1",
    )
    merged = merged.drop(columns=["TEAM_ABBREVIATION_opp_ctx"])

    merged["GAME_DATE"] = merged["GAME_DATE"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    missing_counts = {column: int(merged[column].isna().sum()) for column in NEW_CONTEXT_COLUMNS}

    print(f"row count before merge: {row_count_before_merge}")
    print(f"row count after merge: {len(merged)}")
    print(f"number of players: {merged['PLAYER_ID'].nunique()}")
    print(f"new context columns added: {NEW_CONTEXT_COLUMNS}")
    print(f"missing values per new context column: {missing_counts}")


if __name__ == "__main__":
    main()
