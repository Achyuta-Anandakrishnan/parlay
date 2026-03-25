import os
import time
from collections import defaultdict

import pandas as pd
import requests

SPORT_KEY = "basketball_nba"
MARKET_KEY = "player_rebounds"
REGIONS = "us"
ODDS_FORMAT = "american"
OUTPUT_PATH = "ml_pipeline/data/upcoming_player_rebounds_props.csv"
BASE_URL = "https://api.the-odds-api.com/v4"
TIMEOUT = 30
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 1.5
OUTPUT_COLUMNS = [
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
]


def get_api_key():
    return (
        os.getenv("THE_ODDS_API_KEY")
        or os.getenv("ODDS_API_KEY")
        or os.getenv("THE_ODDS_API_KEY_VALUE")
        or ""
    ).strip()


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

                row = grouped_props[(player_name, line)]
                row["player_name"] = player_name
                row["line"] = line

                outcome_name = str(outcome.get("name", "")).strip().lower()
                if outcome_name == "over":
                    row["over_price"] = outcome.get("price")
                elif outcome_name == "under":
                    row["under_price"] = outcome.get("price")

        rows.extend(grouped_props.values())

    return rows


def load_cached():
    if not os.path.exists(OUTPUT_PATH):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_csv(OUTPUT_PATH)
    for column in ["line", "over_price", "under_price"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def main():
    api_key = get_api_key()
    cached_df = load_cached()

    if not api_key:
        cached_df.to_csv(OUTPUT_PATH, index=False)
        print("events found: 0")
        print(f"props extracted: {len(cached_df)}")
        bookmakers = sorted(cached_df["bookmaker"].dropna().unique().tolist()) if not cached_df.empty else []
        print(f"bookmakers represented: {bookmakers}")
        return

    session = get_session()
    events = fetch_events(session, api_key)
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

    df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
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
