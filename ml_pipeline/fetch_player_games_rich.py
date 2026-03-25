import os
import time

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs

SEASONS = ["2023-24", "2024-25", "2025-26"]
KEEP_COLUMNS = [
    "GAME_DATE",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "MIN",
    "PTS",
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
NUMERIC_COLUMNS = [
    "PLAYER_ID",
    "MIN",
    "PTS",
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
    "is_home",
]
OUTPUT_PATH = "ml_pipeline/data/player_games_rich.csv"
SLEEP_SECONDS = 0.7


def parse_opponent_team(matchup):
    if pd.isna(matchup):
        return None

    matchup = str(matchup)
    if " vs. " in matchup:
        return matchup.split(" vs. ", 1)[1].strip()
    if " @ " in matchup:
        return matchup.split(" @ ", 1)[1].strip()
    return None


def parse_is_home(matchup):
    if pd.isna(matchup):
        return None

    matchup = str(matchup)
    if " vs. " in matchup:
        return 1
    if " @ " in matchup:
        return 0
    return None


def fetch_season_logs(season):
    response = PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    frames = response.get_data_frames()
    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS + ["opponent_team", "is_home"])

    season_logs = frames[0][KEEP_COLUMNS].copy()
    season_logs["opponent_team"] = season_logs["MATCHUP"].apply(parse_opponent_team)
    season_logs["is_home"] = season_logs["MATCHUP"].apply(parse_is_home)
    season_logs["season"] = season
    return season_logs


def main():
    all_frames = []

    for season in SEASONS:
        try:
            season_logs = fetch_season_logs(season)
            if not season_logs.empty:
                all_frames.append(season_logs)
        except Exception as exc:
            print(f"warning: skipped season {season}: {exc}")
        finally:
            time.sleep(SLEEP_SECONDS)

    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        final_df["GAME_DATE"] = pd.to_datetime(final_df["GAME_DATE"], errors="coerce")
        for column in NUMERIC_COLUMNS:
            final_df[column] = pd.to_numeric(final_df[column], errors="coerce")
        final_df = final_df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
        final_df["GAME_DATE"] = final_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
    else:
        final_df = pd.DataFrame(columns=KEEP_COLUMNS + ["opponent_team", "is_home", "season"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    seasons_covered = sorted(final_df["season"].dropna().unique().tolist()) if not final_df.empty else []
    print(f"total rows: {len(final_df)}")
    print(f"unique players: {final_df['PLAYER_ID'].nunique() if not final_df.empty else 0}")
    print(f"seasons covered: {seasons_covered}")
    print(f"columns: {final_df.columns.tolist()}")


if __name__ == "__main__":
    main()
