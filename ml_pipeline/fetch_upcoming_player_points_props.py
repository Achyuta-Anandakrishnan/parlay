import os
import time
from collections import defaultdict

import pandas as pd
import requests

SPORT_KEY = "basketball_nba"
MARKET_KEY = "player_points"
REGIONS = "us"
ODDS_FORMAT = "american"
OUTPUT_PATH = "ml_pipeline/data/upcoming_player_points_props.csv"
BASE_URL = "https://api.the-odds-api.com/v4"
TIMEOUT = 30
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 1.5


def get_api_key():
    api_key = (
        os.getenv("THE_ODDS_API_KEY")
        or os.getenv("ODDS_API_KEY")
        or os.getenv("THE_ODDS_API_KEY_VALUE")
    )
    if not api_key:
        raise RuntimeError("Missing API key. Set THE_ODDS_API_KEY in the environment.")
    return api_key


def get_session():
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


def fetch_events(session, api_key):
    response = session.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/events",
        params={"apiKey": api_key},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def fetch_event_odds(session, api_key, event_id):
    url = f"{BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKET_KEY,
        "oddsFormat": ODDS_FORMAT,
    }

    for attempt in range(MAX_RETRIES + 1):
        response = session.get(url, params=params, timeout=TIMEOUT)
        if response.status_code != 429:
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}

        if attempt == MAX_RETRIES:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        sleep_seconds = float(retry_after) if retry_after else RETRY_SLEEP_SECONDS
        time.sleep(sleep_seconds)

    return {}


def normalize_team_name(value):
    return str(value or "").strip()


def derive_opponent_team(outcome, home_team, away_team):
    player_team = (
        outcome.get("team")
        or outcome.get("participant")
        or outcome.get("participant_name")
        or outcome.get("player_team")
    )
    player_team = normalize_team_name(player_team)
    if not player_team:
        return ""

    if player_team == normalize_team_name(home_team):
        return away_team
    if player_team == normalize_team_name(away_team):
        return home_team
    return ""


def build_rows(event_payload):
    rows = []
    event_id = event_payload.get("id", "")
    commence_time = event_payload.get("commence_time", "")
    home_team = event_payload.get("home_team", "")
    away_team = event_payload.get("away_team", "")

    for bookmaker in event_payload.get("bookmakers", []):
        bookmaker_name = bookmaker.get("title") or bookmaker.get("key") or ""
        grouped_props = defaultdict(
            lambda: {
                "event_id": event_id,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookmaker_name,
                "player_name": "",
                "market_key": MARKET_KEY,
                "line": None,
                "over_price": None,
                "under_price": None,
                "opponent_team": "",
            }
        )

        for market in bookmaker.get("markets", []):
            if market.get("key") != MARKET_KEY:
                continue

            for outcome in market.get("outcomes", []):
                player_name = str(outcome.get("description", "")).strip()
                line = outcome.get("point")
                if not player_name or line is None:
                    continue

                group_key = (player_name, line)
                prop_row = grouped_props[group_key]
                prop_row["player_name"] = player_name
                prop_row["line"] = line
                prop_row["opponent_team"] = derive_opponent_team(outcome, home_team, away_team)

                outcome_name = str(outcome.get("name", "")).strip().lower()
                price = outcome.get("price")
                if outcome_name == "over":
                    prop_row["over_price"] = price
                elif outcome_name == "under":
                    prop_row["under_price"] = price

        rows.extend(grouped_props.values())

    return rows


def main():
    api_key = get_api_key()
    session = get_session()

    try:
        events = fetch_events(session, api_key)
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to fetch events: {exc}") from exc

    print(f"events found: {len(events)}")

    all_rows = []
    for event in events:
        event_id = event.get("id", "")
        try:
            event_payload = fetch_event_odds(session, api_key, event_id)
        except requests.RequestException as exc:
            print(f"warning: skipped event {event_id}: {exc}")
            continue

        all_rows.extend(build_rows(event_payload))
        time.sleep(0.2)

    df = pd.DataFrame(
        all_rows,
        columns=[
            "event_id",
            "commence_time",
            "home_team",
            "away_team",
            "bookmaker",
            "player_name",
            "market_key",
            "line",
            "over_price",
            "under_price",
            "opponent_team",
        ],
    )

    if not df.empty:
        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
        df["over_price"] = pd.to_numeric(df["over_price"], errors="coerce")
        df["under_price"] = pd.to_numeric(df["under_price"], errors="coerce")
        df = df.sort_values(
            ["commence_time", "home_team", "away_team", "bookmaker", "player_name", "line"]
        ).reset_index(drop=True)
        df["commence_time"] = df["commence_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    bookmakers = sorted(df["bookmaker"].dropna().unique().tolist()) if not df.empty else []
    print(f"props extracted: {len(df)}")
    print(f"bookmakers represented: {bookmakers}")


if __name__ == "__main__":
    main()
