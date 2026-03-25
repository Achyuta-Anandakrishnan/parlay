import os
import time

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs
from nba_api.stats.static import players

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
]
OUTPUT_PATH = "data/player_games.csv"
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
        return pd.DataFrame(columns=KEEP_COLUMNS)

    season_logs = frames[0][KEEP_COLUMNS].copy()
    season_logs["opponent_team"] = season_logs["MATCHUP"].apply(parse_opponent_team)
    season_logs["is_home"] = season_logs["MATCHUP"].apply(parse_is_home)
    return season_logs


def main():
    active_players = players.get_active_players()
    active_player_ids = {player["id"] for player in active_players}
    all_frames = []
    total_players = len(active_players)
    season_logs_by_year = {}

    for season in SEASONS:
        try:
            season_logs = fetch_season_logs(season)
            if not season_logs.empty:
                season_logs = season_logs[season_logs["PLAYER_ID"].isin(active_player_ids)].copy()
            season_logs_by_year[season] = season_logs
        except Exception as exc:
            print(f"Skipping season {season}: {exc}")
            season_logs_by_year[season] = pd.DataFrame(columns=KEEP_COLUMNS + ["opponent_team", "is_home"])
        finally:
            time.sleep(SLEEP_SECONDS)

    for index, player in enumerate(active_players, start=1):
        player_id = player["id"]
        player_name = player["full_name"]
        print(f"Processing player {index}/{total_players}: {player_name}")

        for season in SEASONS:
            game_logs = season_logs_by_year.get(season)
            if game_logs is None or game_logs.empty:
                continue

            player_logs = game_logs[game_logs["PLAYER_ID"] == player_id].copy()
            if not player_logs.empty:
                all_frames.append(player_logs)

    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        final_df["GAME_DATE"] = pd.to_datetime(final_df["GAME_DATE"], errors="coerce")
        final_df = final_df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
        final_df["GAME_DATE"] = final_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
    else:
        final_df = pd.DataFrame(columns=KEEP_COLUMNS + ["opponent_team", "is_home"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(final_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
