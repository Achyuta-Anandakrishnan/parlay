import pandas as pd

DROP_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "opponent_team",
    "points_actual",
    "rebounds_actual",
]


def season_from_date(value):
    value = pd.Timestamp(value)
    if value.month >= 10:
        return f"{value.year}-{str(value.year + 1)[-2:]}"
    return f"{value.year - 1}-{str(value.year)[-2:]}"


def load_context_with_rebounds(features_path, player_games_path):
    features = pd.read_csv(features_path)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="coerce")

    rebounds_source = pd.read_csv(
        player_games_path,
        usecols=["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION", "REB"],
    )
    rebounds_source["GAME_DATE"] = pd.to_datetime(rebounds_source["GAME_DATE"], errors="coerce")
    rebounds_source = rebounds_source.rename(columns={"REB": "rebounds_actual"})
    rebounds_source["rebounds_actual"] = pd.to_numeric(rebounds_source["rebounds_actual"], errors="coerce")

    frame = features.merge(
        rebounds_source,
        on=["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION"],
        how="left",
    )

    frame["season"] = frame["GAME_DATE"].apply(season_from_date)
    frame = frame.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    season_group = frame.groupby(["PLAYER_ID", "season"], group_keys=False)
    shifted_rebounds = season_group["rebounds_actual"].shift(1)
    frame["season_avg_rebounds_prior"] = (
        shifted_rebounds.groupby([frame["PLAYER_ID"], frame["season"]]).transform(
            lambda values: values.expanding().mean()
        )
    )
    return frame


def get_model_drop_columns(df, extra_drop_columns=None):
    drop_columns = list(DROP_COLUMNS)
    if extra_drop_columns:
        drop_columns.extend(extra_drop_columns)

    additional_drop_columns = [
        column
        for column in df.columns
        if column not in drop_columns and df[column].dtype == "object"
    ]
    return drop_columns + additional_drop_columns
