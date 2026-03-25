import datetime as dt
import json
import math
import os
import pickle
import re
import unicodedata
from collections import defaultdict

import pandas as pd
import requests
from nba_api.stats.static import teams
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from ml_pipeline.rebounds_context_utils import load_context_with_rebounds
except ModuleNotFoundError:
    from rebounds_context_utils import load_context_with_rebounds

PRODUCTION_POINTS_MODEL_PATH = "ml_pipeline/models/points_model_context.pkl"
PRODUCTION_CALIBRATION_PATH = "ml_pipeline/models/points_model_calibration.pkl"
PRODUCTION_REBOUNDS_MODEL_PATH = "ml_pipeline/models/rebounds_model_context.pkl"
PRODUCTION_REBOUNDS_CALIBRATION_PATH = "ml_pipeline/models/rebounds_model_calibration.pkl"
EXPERIMENTAL_MINUTES_MODEL_PATH = "ml_pipeline/models/minutes_model_context_final.pkl"
EXPERIMENTAL_POINTS_MODEL_PATH = "ml_pipeline/models/points_model_with_oof_minutes.pkl"
EXPERIMENTAL_CALIBRATION_PATH = "ml_pipeline/models/points_model_calibration_with_oof_minutes.pkl"
FEATURES_PATH = "ml_pipeline/data/features_player_points_context.csv"
CACHED_POINTS_PROPS_PATH = "ml_pipeline/data/upcoming_player_points_props.csv"
CACHED_REBOUNDS_PROPS_PATH = "ml_pipeline/data/upcoming_player_rebounds_props.csv"
SNAPSHOT_JSON_PATH = "ml_pipeline/data/tomorrow_ui_snapshot.json"
SNAPSHOT_CSV_PATH = "ml_pipeline/data/tomorrow_ui_snapshot.csv"
SNAPSHOT_REBOUNDS_JSON_PATH = "ml_pipeline/data/tomorrow_ui_snapshot_rebounds.json"
SNAPSHOT_REBOUNDS_CSV_PATH = "ml_pipeline/data/tomorrow_ui_snapshot_rebounds.csv"
SNAPSHOT_ASSISTS_JSON_PATH = "ml_pipeline/data/tomorrow_ui_snapshot_assists.json"
SNAPSHOT_ASSISTS_CSV_PATH = "ml_pipeline/data/tomorrow_ui_snapshot_assists.csv"

ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
POINTS_MARKET_KEY = "player_points"
REBOUNDS_MARKET_KEY = "player_rebounds"
REGIONS = "us"
ODDS_FORMAT = "american"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
TIMEOUT = 30
MAX_RETRIES = 2
DEFAULT_WINDOW_DAYS = 2
LOW_SAMPLE_THRESHOLD = 8
EXTREME_EDGE_THRESHOLD = 0.18
TAIL_CALIBRATION_THRESHOLD = 75
DISAGREEMENT_PREDICTION_THRESHOLD = 3.5
DISAGREEMENT_EDGE_THRESHOLD = 0.08
EDGE_HIGHLIGHT_THRESHOLD = 0.07

PLAYER_ALIASES = {
    "nicolas claxton": "nic claxton",
    "carlton carrington": "bub carrington",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
TEAM_NAME_ALIASES = {
    "la clippers": "los angeles clippers",
}
TEAM_CONTEXT_COLUMNS = [
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
OPP_CONTEXT_COLUMNS = [
    "opp_points_allowed_last_5",
    "opp_points_allowed_last_10",
    "opp_fga_allowed_last_5",
    "opp_fg3a_allowed_last_5",
    "opp_fta_allowed_last_5",
    "opp_reb_allowed_last_5",
    "opp_ast_allowed_last_5",
    "opp_pace_proxy_last_5",
]


def get_snapshot_paths(prop_type):
    if prop_type == "rebounds":
        return SNAPSHOT_REBOUNDS_JSON_PATH, SNAPSHOT_REBOUNDS_CSV_PATH
    if prop_type == "assists":
        return SNAPSHOT_ASSISTS_JSON_PATH, SNAPSHOT_ASSISTS_CSV_PATH
    return SNAPSHOT_JSON_PATH, SNAPSHOT_CSV_PATH


def get_prop_config(prop_type):
    if prop_type == "rebounds":
        return {
            "display_name": "rebounds",
            "market_key": REBOUNDS_MARKET_KEY,
            "cached_props_path": CACHED_REBOUNDS_PROPS_PATH,
            "production_model_path": PRODUCTION_REBOUNDS_MODEL_PATH,
            "production_calibration_path": PRODUCTION_REBOUNDS_CALIBRATION_PATH,
            "prediction_column": "predicted_rebounds_prod",
            "prediction_label": "rebounds",
            "primary_model_label": "Production: context-only calibrated rebounds",
            "experimental_model_label": "Experimental comparison not available for rebounds yet.",
            "experimental_supported": False,
            "coming_soon_note": "Assists remain intentionally marked coming soon because no trained assists model exists yet.",
        }
    if prop_type == "assists":
        return {
            "display_name": "assists",
            "market_key": "",
            "cached_props_path": "",
            "production_model_path": "",
            "production_calibration_path": "",
            "prediction_column": "predicted_assists_prod",
            "prediction_label": "assists",
            "primary_model_label": "Assists model coming soon",
            "experimental_model_label": "No assists model is available yet.",
            "experimental_supported": False,
            "coming_soon_note": "Assists are intentionally marked coming soon because no trained assists model exists yet.",
        }
    return {
        "display_name": "points",
        "market_key": POINTS_MARKET_KEY,
        "cached_props_path": CACHED_POINTS_PROPS_PATH,
        "production_model_path": PRODUCTION_POINTS_MODEL_PATH,
        "production_calibration_path": PRODUCTION_CALIBRATION_PATH,
        "prediction_column": "predicted_points_prod",
        "prediction_label": "points",
        "primary_model_label": "Production: context-only calibrated points",
        "experimental_model_label": "Experimental: OOF-minutes stacked points",
        "experimental_supported": True,
        "coming_soon_note": "Assists remain intentionally marked coming soon because no trained assists model exists yet.",
    }


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        if value.tzinfo is not None:
            return value.isoformat().replace("+00:00", "Z")
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def normalize_text(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_player_name(value):
    text = normalize_text(value)
    parts = text.split()
    if parts and parts[-1] in SUFFIXES:
        parts = parts[:-1]
    text = " ".join(parts)
    return PLAYER_ALIASES.get(text, text)


def normalize_team_name(value):
    text = normalize_text(value)
    return TEAM_NAME_ALIASES.get(text, text)


def season_from_date(value):
    value = pd.Timestamp(value)
    if value.month >= 10:
        return f"{value.year}-{str(value.year + 1)[-2:]}"
    return f"{value.year - 1}-{str(value.year)[-2:]}"


def get_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def get_api_key():
    return (
        os.getenv("THE_ODDS_API_KEY")
        or os.getenv("ODDS_API_KEY")
        or os.getenv("THE_ODDS_API_KEY_VALUE")
        or ""
    ).strip()


def american_to_prob(odds):
    if pd.isna(odds):
        return math.nan

    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return math.nan


def build_team_maps():
    abbr_to_name = {}
    name_to_abbr = {}
    for team in teams.get_teams():
        normalized_name = normalize_team_name(team["full_name"])
        abbr_to_name[team["abbreviation"]] = normalized_name
        name_to_abbr[normalized_name] = team["abbreviation"]
    return abbr_to_name, name_to_abbr


def latest_by_key_prefer_current(df, key_column, current_season):
    sorted_df = df.sort_values(["GAME_DATE", key_column]).reset_index(drop=True)
    current_df = sorted_df[sorted_df["season"] == current_season].copy()

    preferred = current_df.drop_duplicates(subset=[key_column], keep="last")
    if preferred.empty:
        return sorted_df.drop_duplicates(subset=[key_column], keep="last")

    missing_keys = set(sorted_df[key_column]) - set(preferred[key_column])
    if not missing_keys:
        return preferred

    fallback = sorted_df[sorted_df[key_column].isin(missing_keys)].drop_duplicates(subset=[key_column], keep="last")
    return pd.concat([preferred, fallback], ignore_index=True)


def load_calibration_entries(path):
    with open(path, "rb") as file:
        payload = pickle.load(file)

    bins = payload["bins"]
    entries = []
    for index, record in enumerate(payload["records"]):
        entries.append(
            {
                "bin_label": record["bin_label"],
                "count": int(record["count"]),
                "empirical_p_over": record["empirical_p_over"],
                "left": bins[index],
                "right": bins[index + 1],
            }
        )
    return entries


def calibration_distance(diff, left, right):
    left_open = not math.isinf(left)

    if math.isinf(left) and diff <= right:
        return 0.0
    if math.isinf(right) and diff > left:
        return 0.0
    if (diff > left or (not left_open and diff >= left)) and diff <= right:
        return 0.0
    if diff <= left:
        return left - diff
    return diff - right


def calibrated_probability(diff, calibration_entries):
    best_entry = None
    best_distance = None

    for entry in calibration_entries:
        if entry["count"] <= 0 or pd.isna(entry["empirical_p_over"]):
            continue

        distance = calibration_distance(diff, entry["left"], entry["right"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entry = entry

    if best_entry is None:
        return 0.5, "", 0

    return float(best_entry["empirical_p_over"]), best_entry["bin_label"], int(best_entry["count"])


def fetch_schedule(days=DEFAULT_WINDOW_DAYS, start_offset=1):
    today = utcnow().date()
    games = []

    with get_session() as session:
        for day_index in range(start_offset, start_offset + days):
            target_date = today + dt.timedelta(days=day_index)
            try:
                response = session.get(
                    ESPN_SCOREBOARD_URL,
                    params={"dates": target_date.strftime("%Y%m%d")},
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                continue

            for event in payload.get("events", []):
                competitions = event.get("competitions") or []
                if not competitions:
                    continue

                competition = competitions[0]
                competitors = competition.get("competitors") or []
                try:
                    home_comp = next(item for item in competitors if item.get("homeAway") == "home")
                    away_comp = next(item for item in competitors if item.get("homeAway") == "away")
                except StopIteration:
                    continue

                home_team = home_comp["team"]["displayName"]
                away_team = away_comp["team"]["displayName"]
                broadcasts = competition.get("broadcasts") or []
                national = []
                local = []
                for broadcast in broadcasts:
                    names = broadcast.get("names") or []
                    if broadcast.get("market") == "national":
                        national.extend(names)
                    else:
                        local.extend(names)
                broadcast_names = national or local

                commence_time = competition.get("date") or event.get("date")
                commence_dt = pd.to_datetime(commence_time, utc=True, errors="coerce")
                games.append(
                    {
                        "schedule_key": build_schedule_key(commence_dt, home_team, away_team),
                        "event_id": event.get("id", ""),
                        "commence_time": commence_dt.isoformat().replace("+00:00", "Z") if pd.notna(commence_dt) else "",
                        "home_team": home_team,
                        "away_team": away_team,
                        "broadcast": ", ".join(broadcast_names),
                    }
                )

    games.sort(key=lambda row: row["commence_time"])
    return games


def build_schedule_key(commence_time, home_team, away_team):
    commence_dt = pd.to_datetime(commence_time, utc=True, errors="coerce")
    date_key = commence_dt.strftime("%Y-%m-%d") if pd.notna(commence_dt) else ""
    return "|".join([date_key, normalize_team_name(home_team), normalize_team_name(away_team)])


def fetch_events(session, api_key):
    response = session.get(
        f"{ODDS_BASE_URL}/sports/{SPORT_KEY}/events",
        params={"apiKey": api_key},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def fetch_event_odds(session, api_key, event_id, market_key):
    response = session.get(
        f"{ODDS_BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "regions": REGIONS,
            "markets": market_key,
            "oddsFormat": ODDS_FORMAT,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def derive_opponent_team(outcome, home_team, away_team):
    player_team = (
        outcome.get("team")
        or outcome.get("participant")
        or outcome.get("participant_name")
        or outcome.get("player_team")
    )
    player_team = str(player_team or "").strip()
    if not player_team:
        return ""

    if player_team == str(home_team).strip():
        return away_team
    if player_team == str(away_team).strip():
        return home_team
    return ""


def build_prop_rows(event_payload, market_key):
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
                "market_key": market_key,
                "line": None,
                "over_price": None,
                "under_price": None,
                "opponent_team": "",
            }
        )

        for market in bookmaker.get("markets", []):
            if market.get("key") != market_key:
                continue

            for outcome in market.get("outcomes", []):
                player_name = str(outcome.get("description", "")).strip()
                line = outcome.get("point")
                if not player_name or line is None:
                    continue

                prop_key = (player_name, line)
                row = grouped_props[prop_key]
                row["player_name"] = player_name
                row["line"] = line
                row["opponent_team"] = derive_opponent_team(outcome, home_team, away_team)

                side_name = str(outcome.get("name", "")).strip().lower()
                if side_name == "over":
                    row["over_price"] = outcome.get("price")
                elif side_name == "under":
                    row["under_price"] = outcome.get("price")

        rows.extend(grouped_props.values())

    return rows


def fetch_props(prop_type="points", days=DEFAULT_WINDOW_DAYS, start_offset=1, refresh_props=False):
    config = get_prop_config(prop_type)
    cached_df = load_cached_props(config["cached_props_path"])
    api_key = get_api_key()

    if refresh_props and api_key:
        try:
            props_df = fetch_props_from_api(
                api_key,
                market_key=config["market_key"],
                days=days,
                start_offset=start_offset,
            )
            props_df.to_csv(config["cached_props_path"], index=False)
        except requests.RequestException:
            props_df = cached_df
    else:
        if not cached_df.empty:
            props_df = cached_df
        elif api_key and config["market_key"]:
            props_df = fetch_props_from_api(
                api_key,
                market_key=config["market_key"],
                days=days,
                start_offset=start_offset,
            )
        else:
            props_df = cached_df

    if props_df.empty:
        return props_df

    window_start = pd.Timestamp(utcnow().date() + dt.timedelta(days=start_offset), tz="UTC")
    window_end = window_start + pd.Timedelta(days=days)
    props_df["commence_time"] = pd.to_datetime(props_df["commence_time"], utc=True, errors="coerce")
    props_df = props_df[(props_df["commence_time"] >= window_start) & (props_df["commence_time"] < window_end)].copy()
    props_df["schedule_key"] = props_df.apply(
        lambda row: build_schedule_key(row["commence_time"], row["home_team"], row["away_team"]),
        axis=1,
    )
    props_df = props_df.sort_values(
        ["commence_time", "home_team", "away_team", "bookmaker", "player_name", "line"]
    ).reset_index(drop=True)
    return props_df


def fetch_props_from_api(api_key, market_key, days=DEFAULT_WINDOW_DAYS, start_offset=1):
    if not api_key or not market_key:
        return pd.DataFrame()

    today = utcnow().date()
    window_start = pd.Timestamp(today + dt.timedelta(days=start_offset), tz="UTC")
    window_end = window_start + pd.Timedelta(days=days)
    all_rows = []

    with get_session() as session:
        events = fetch_events(session, api_key)
        for event in events:
            commence_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
            if pd.isna(commence_time) or commence_time < window_start or commence_time >= window_end:
                continue

            event_payload = fetch_event_odds(session, api_key, event.get("id", ""), market_key)
            all_rows.extend(build_prop_rows(event_payload, market_key))

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

    if df.empty:
        return df

    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_price"] = pd.to_numeric(df["over_price"], errors="coerce")
    df["under_price"] = pd.to_numeric(df["under_price"], errors="coerce")
    return df


def load_cached_props(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(
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
            ]
        )

    df = pd.read_csv(path)
    for column in ["line", "over_price", "under_price"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_features(prop_type="points"):
    if prop_type == "rebounds":
        features = load_context_with_rebounds(FEATURES_PATH, "ml_pipeline/data/player_games_rich.csv")
    else:
        features = pd.read_csv(FEATURES_PATH)
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="coerce")
    features["season"] = features["GAME_DATE"].apply(season_from_date)
    features["normalized_player_name"] = features["PLAYER_NAME"].apply(normalize_player_name)
    return features


def load_production_artifacts(prop_type="points"):
    config = get_prop_config(prop_type)
    with open(config["production_model_path"], "rb") as file:
        model = pickle.load(file)
    calibration_entries = load_calibration_entries(config["production_calibration_path"])
    return {
        "model": model,
        "calibration_entries": calibration_entries,
        "prop_type": prop_type,
        "prediction_column": config["prediction_column"],
        "prediction_label": config["prediction_label"],
    }


def load_experimental_artifacts(prop_type="points"):
    if prop_type != "points":
        return None

    required_paths = [
        EXPERIMENTAL_MINUTES_MODEL_PATH,
        EXPERIMENTAL_POINTS_MODEL_PATH,
        EXPERIMENTAL_CALIBRATION_PATH,
    ]
    if not all(os.path.exists(path) for path in required_paths):
        return None

    with open(EXPERIMENTAL_MINUTES_MODEL_PATH, "rb") as file:
        minutes_model = pickle.load(file)
    with open(EXPERIMENTAL_POINTS_MODEL_PATH, "rb") as file:
        points_model = pickle.load(file)
    calibration_entries = load_calibration_entries(EXPERIMENTAL_CALIBRATION_PATH)
    return {
        "minutes_model": minutes_model,
        "points_model": points_model,
        "calibration_entries": calibration_entries,
    }


def reconstruct_base_rows(props_df, features):
    if props_df.empty:
        return pd.DataFrame(), {}

    current_season = season_from_date(props_df["commence_time"].max())
    player_state = latest_by_key_prefer_current(features, "normalized_player_name", current_season)
    team_context = latest_by_key_prefer_current(features, "TEAM_ABBREVIATION", current_season)
    opponent_context = latest_by_key_prefer_current(features, "opponent_team", current_season)

    player_state_lookup = player_state.set_index("normalized_player_name")
    team_context_lookup = team_context.set_index("TEAM_ABBREVIATION")
    opponent_context_lookup = opponent_context.set_index("opponent_team")
    abbr_to_name, name_to_abbr = build_team_maps()

    rows = []
    unmatched = {}

    for prop in props_df.itertuples(index=False):
        normalized_player = normalize_player_name(prop.player_name)
        if normalized_player not in player_state_lookup.index:
            unmatched[(prop.event_id, prop.bookmaker, prop.player_name, prop.line)] = "missing_player_state"
            continue

        state = player_state_lookup.loc[normalized_player]
        player_team_abbr = state["TEAM_ABBREVIATION"]
        player_team_norm = abbr_to_name.get(player_team_abbr, "")
        home_team_norm = normalize_team_name(prop.home_team)
        away_team_norm = normalize_team_name(prop.away_team)

        if player_team_norm == home_team_norm:
            is_home = 1
            opponent_team_name = prop.away_team
            opponent_team_abbr = name_to_abbr.get(away_team_norm)
        elif player_team_norm == away_team_norm:
            is_home = 0
            opponent_team_name = prop.home_team
            opponent_team_abbr = name_to_abbr.get(home_team_norm)
        else:
            unmatched[(prop.event_id, prop.bookmaker, prop.player_name, prop.line)] = "team_mismatch"
            continue

        if player_team_abbr not in team_context_lookup.index:
            unmatched[(prop.event_id, prop.bookmaker, prop.player_name, prop.line)] = "missing_team_context"
            continue
        if opponent_team_abbr not in opponent_context_lookup.index:
            unmatched[(prop.event_id, prop.bookmaker, prop.player_name, prop.line)] = "missing_opponent_context"
            continue

        commence_date = pd.Timestamp(prop.commence_time).tz_convert(None).normalize()
        last_game_date = pd.Timestamp(state["GAME_DATE"]).normalize()
        days_since_last_game = (commence_date - last_game_date).days

        reconstructed = state.to_dict()
        reconstructed["is_home"] = is_home
        reconstructed["days_since_last_game"] = days_since_last_game
        reconstructed["is_back_to_back"] = 1 if days_since_last_game == 1 else 0

        team_row = team_context_lookup.loc[player_team_abbr]
        for column in TEAM_CONTEXT_COLUMNS:
            reconstructed[column] = team_row[column]

        opp_row = opponent_context_lookup.loc[opponent_team_abbr]
        for column in OPP_CONTEXT_COLUMNS:
            reconstructed[column] = opp_row[column]

        reconstructed["event_id"] = prop.event_id
        reconstructed["schedule_key"] = prop.schedule_key
        reconstructed["commence_time"] = prop.commence_time
        reconstructed["home_team"] = prop.home_team
        reconstructed["away_team"] = prop.away_team
        reconstructed["player_name"] = prop.player_name
        reconstructed["bookmaker"] = prop.bookmaker
        reconstructed["prop_type"] = prop.market_key
        reconstructed["line"] = prop.line
        reconstructed["over_price"] = prop.over_price
        reconstructed["under_price"] = prop.under_price
        reconstructed["opponent_team_live"] = opponent_team_name
        rows.append(reconstructed)

    return pd.DataFrame(rows), unmatched


def apply_market_probabilities(frame):
    frame["p_over_raw"] = frame["over_price"].apply(american_to_prob)
    frame["p_under_raw"] = frame["under_price"].apply(american_to_prob)
    frame["market_prob_sum"] = frame["p_over_raw"] + frame["p_under_raw"]
    frame["p_over_market_novig"] = frame["p_over_raw"] / frame["market_prob_sum"]
    frame["p_under_market_novig"] = frame["p_under_raw"] / frame["market_prob_sum"]
    return frame


def score_production(frame, artifacts):
    if frame.empty:
        return frame

    model = artifacts["model"]
    calibration_entries = artifacts["calibration_entries"]
    feature_columns = list(model.feature_name_)
    prediction_column = artifacts["prediction_column"]
    prediction_label = artifacts["prediction_label"]

    X_prod = frame[feature_columns].copy()
    frame[prediction_column] = model.predict(X_prod)
    frame["predicted_value_prod"] = frame[prediction_column]
    frame["predicted_value_label"] = prediction_label
    frame["prod_diff"] = frame["predicted_value_prod"] - frame["line"]

    calibration_results = frame["prod_diff"].apply(lambda diff: calibrated_probability(diff, calibration_entries))
    frame["p_over_prod"] = calibration_results.apply(lambda item: item[0])
    frame["prod_calibration_bin"] = calibration_results.apply(lambda item: item[1])
    frame["prod_calibration_count"] = calibration_results.apply(lambda item: item[2])
    frame["p_under_prod"] = 1 - frame["p_over_prod"]
    frame["edge_over_prod"] = frame["p_over_prod"] - frame["p_over_market_novig"]
    frame["edge_under_prod"] = frame["p_under_prod"] - frame["p_under_market_novig"]
    frame["best_side_prod"] = frame.apply(
        lambda row: "over" if row["edge_over_prod"] > row["edge_under_prod"] else "under",
        axis=1,
    )
    frame["best_edge_prod"] = frame.apply(
        lambda row: row["edge_over_prod"] if row["best_side_prod"] == "over" else row["edge_under_prod"],
        axis=1,
    )
    return frame


def score_experimental(frame, artifacts):
    if frame.empty or artifacts is None:
        frame["predicted_points_exp"] = pd.NA
        frame["predicted_value_exp"] = pd.NA
        frame["best_edge_exp"] = pd.NA
        frame["best_side_exp"] = pd.NA
        frame["predicted_minutes_exp"] = pd.NA
        return frame

    minutes_model = artifacts["minutes_model"]
    points_model = artifacts["points_model"]
    calibration_entries = artifacts["calibration_entries"]

    minutes_features = list(minutes_model.feature_name_)
    points_features = list(points_model.feature_name_)

    X_minutes = frame[minutes_features].copy()
    frame["predicted_minutes_exp"] = minutes_model.predict(X_minutes)
    frame["predicted_minutes_oof"] = frame["predicted_minutes_exp"]
    frame["predicted_minus_recent_minutes_oof"] = frame["predicted_minutes_exp"] - frame["min_last_5"]

    X_points = frame[points_features].copy()
    frame["predicted_points_exp"] = points_model.predict(X_points)
    frame["predicted_value_exp"] = frame["predicted_points_exp"]
    frame["exp_diff"] = frame["predicted_points_exp"] - frame["line"]

    calibration_results = frame["exp_diff"].apply(lambda diff: calibrated_probability(diff, calibration_entries))
    frame["p_over_exp"] = calibration_results.apply(lambda item: item[0])
    frame["p_under_exp"] = 1 - frame["p_over_exp"]
    frame["edge_over_exp"] = frame["p_over_exp"] - frame["p_over_market_novig"]
    frame["edge_under_exp"] = frame["p_under_exp"] - frame["p_under_market_novig"]
    frame["best_side_exp"] = frame.apply(
        lambda row: "over" if row["edge_over_exp"] > row["edge_under_exp"] else "under",
        axis=1,
    )
    frame["best_edge_exp"] = frame.apply(
        lambda row: row["edge_over_exp"] if row["best_side_exp"] == "over" else row["edge_under_exp"],
        axis=1,
    )
    return frame


def add_flags(frame):
    if frame.empty:
        return frame

    def build_row_flags(row):
        flags = []
        notes = []
        predicted_value_prod = row.get("predicted_value_prod")
        predicted_value_exp = row.get("predicted_value_exp")

        if pd.isna(predicted_value_prod):
            flags.append("UNMATCHED")
            notes.append("No production player-state match for this prop.")

        if pd.notna(row.get("games_played_prior")) and float(row["games_played_prior"]) < LOW_SAMPLE_THRESHOLD:
            flags.append("LOW_SAMPLE")
            notes.append(f"Only {int(row['games_played_prior'])} prior games in player state.")

        if pd.notna(row.get("best_edge_prod")) and float(row["best_edge_prod"]) > EXTREME_EDGE_THRESHOLD:
            flags.append("EXTREME_EDGE")
            notes.append("Production edge is in the extreme tail. Treat as fragile.")

        if pd.notna(row.get("prod_calibration_count")) and int(row["prod_calibration_count"]) < TAIL_CALIBRATION_THRESHOLD:
            flags.append("TAIL_CALIBRATION")
            notes.append(f"Calibration bin has only {int(row['prod_calibration_count'])} historical samples.")

        best_side_exp = row.get("best_side_exp")
        best_edge_exp = row.get("best_edge_exp")
        if (
            pd.notna(predicted_value_exp)
            and pd.notna(predicted_value_prod)
            and (
                abs(float(predicted_value_exp) - float(predicted_value_prod)) >= DISAGREEMENT_PREDICTION_THRESHOLD
                or (
                    pd.notna(best_edge_exp)
                    and abs(float(best_edge_exp) - float(row["best_edge_prod"])) >= DISAGREEMENT_EDGE_THRESHOLD
                )
                or (
                    isinstance(best_side_exp, str)
                    and isinstance(row.get("best_side_prod"), str)
                    and best_side_exp != row["best_side_prod"]
                )
            )
        ):
            flags.append("EXPERIMENTAL_DIFFERENCE")
            notes.append("Production and experimental pipelines disagree materially.")

        return flags, notes

    flag_results = frame.apply(build_row_flags, axis=1)
    frame["flags"] = flag_results.apply(lambda item: item[0])
    frame["notes"] = flag_results.apply(lambda item: " ".join(item[1]))
    frame["confidence_flag"] = frame["flags"].apply(lambda flags: "|".join(flags) if flags else "OK")
    frame["excluded_from_test"] = frame["flags"].apply(
        lambda flags: any(flag in {"LOW_SAMPLE", "EXTREME_EDGE", "TAIL_CALIBRATION", "UNMATCHED"} for flag in flags)
    )
    return frame


def merge_scores_into_props(props_df, scored_df, unmatched_map):
    merged = props_df.copy()
    if scored_df.empty:
        merged["predicted_value_prod"] = pd.NA
        merged["best_edge_prod"] = pd.NA
        merged["best_side_prod"] = pd.NA
        merged["confidence_flag"] = "UNMATCHED"
        merged["notes"] = "No matched player-state row available."
        merged["flags"] = [["UNMATCHED"]] * len(merged)
        merged["excluded_from_test"] = True
        return merged

    score_columns = [
        "event_id",
        "bookmaker",
        "player_name",
        "line",
        "predicted_value_prod",
        "predicted_value_label",
        "p_over_prod",
        "p_under_prod",
        "edge_over_prod",
        "edge_under_prod",
        "best_side_prod",
        "best_edge_prod",
        "prod_calibration_bin",
        "prod_calibration_count",
        "predicted_points_exp",
        "best_side_exp",
        "best_edge_exp",
        "predicted_minutes_exp",
        "p_over_market_novig",
        "p_under_market_novig",
        "confidence_flag",
        "notes",
        "flags",
        "excluded_from_test",
        "games_played_prior",
    ]
    for optional_column in ["predicted_points_prod", "predicted_rebounds_prod", "predicted_value_exp"]:
        if optional_column in scored_df.columns and optional_column not in score_columns:
            score_columns.append(optional_column)
    scored_subset = scored_df[score_columns].copy()
    merged = merged.merge(
        scored_subset,
        on=["event_id", "bookmaker", "player_name", "line"],
        how="left",
    )

    def fallback_flag(row):
        key = (row["event_id"], row["bookmaker"], row["player_name"], row["line"])
        if key not in unmatched_map:
            return row["confidence_flag"]
        return "UNMATCHED"

    def fallback_notes(row):
        key = (row["event_id"], row["bookmaker"], row["player_name"], row["line"])
        if key not in unmatched_map:
            return row["notes"]
        return f"Unmatched prop: {unmatched_map[key].replace('_', ' ')}."

    merged["confidence_flag"] = merged.apply(fallback_flag, axis=1)
    merged["notes"] = merged.apply(fallback_notes, axis=1)
    merged["flags"] = merged["flags"].apply(lambda value: value if isinstance(value, list) else ["UNMATCHED"] if pd.isna(value) else value)
    merged["excluded_from_test"] = merged["excluded_from_test"].fillna(True)
    return merged


def build_summary(games, props_df, prop_type):
    bookmakers = sorted(props_df["bookmaker"].dropna().unique().tolist()) if not props_df.empty else []
    matched_mask = props_df["predicted_value_prod"].notna() if "predicted_value_prod" in props_df else pd.Series(dtype=bool)
    risky_mask = props_df["confidence_flag"].fillna("UNMATCHED") != "OK" if "confidence_flag" in props_df else pd.Series(dtype=bool)
    config = get_prop_config(prop_type)

    return {
        "games_found": len(games),
        "props_found": int(len(props_df)),
        "matched_props": int(matched_mask.sum()) if not matched_mask.empty else 0,
        "bookmakers": bookmakers,
        "average_edge_prod": float(props_df.loc[matched_mask, "best_edge_prod"].mean()) if matched_mask.any() else 0.0,
        "above_threshold_count": int((props_df.get("best_edge_prod", pd.Series(dtype=float)).fillna(0) >= EDGE_HIGHLIGHT_THRESHOLD).sum()) if not props_df.empty else 0,
        "risky_props_count": int(risky_mask.sum()) if not risky_mask.empty else 0,
        "primary_model": config["primary_model_label"],
        "experimental_available": bool(props_df["predicted_value_exp"].notna().any()) if "predicted_value_exp" in props_df else False,
    }


def build_test_plan(props_df, prop_type):
    config = get_prop_config(prop_type)
    if props_df.empty:
        return {
            "props_in_scope": 0,
            "primary_model": config["primary_model_label"],
            "experimental_model": config["experimental_model_label"],
            "excluded_due_to_flags": 0,
            "excluded_examples": [],
            "notes": [
                f"{config['primary_model_label']} is the default ranking path for this prop type.",
                config["experimental_model_label"],
                config["coming_soon_note"],
            ],
        }

    in_scope = props_df[props_df["predicted_value_prod"].notna()].copy()
    excluded = props_df[props_df["excluded_from_test"]].copy()
    excluded_items = excluded.sort_values(["confidence_flag", "best_edge_prod"], ascending=[True, False]).head(12)

    return {
        "props_in_scope": int(len(in_scope)),
        "primary_model": config["primary_model_label"],
        "experimental_model": config["experimental_model_label"],
        "excluded_due_to_flags": int(len(excluded)),
        "excluded_examples": excluded_items[
            ["player_name", "home_team", "away_team", "bookmaker", "line", "confidence_flag"]
        ].to_dict(orient="records"),
        "notes": [
            f"{config['primary_model_label']} is the default ranking path for this prop type.",
            config["experimental_model_label"],
            config["coming_soon_note"],
        ],
    }


def build_snapshot(prop_type="points", days=DEFAULT_WINDOW_DAYS, refresh_props=False, include_experimental=True, start_offset=1):
    config = get_prop_config(prop_type)
    schedule_games = fetch_schedule(days=days, start_offset=start_offset)
    if prop_type == "assists":
        props_df = pd.DataFrame(
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
                "schedule_key",
            ]
        )
        merged_props = props_df.copy()
    else:
        props_df = fetch_props(prop_type=prop_type, days=days, start_offset=start_offset, refresh_props=refresh_props)
        features = load_features(prop_type=prop_type)
        production_artifacts = load_production_artifacts(prop_type=prop_type)
        experimental_artifacts = load_experimental_artifacts(prop_type=prop_type) if include_experimental else None

        if props_df.empty:
            merged_props = props_df.copy()
        else:
            base_rows, unmatched_map = reconstruct_base_rows(props_df, features)
            if not base_rows.empty:
                base_rows = apply_market_probabilities(base_rows)
                base_rows = score_production(base_rows, production_artifacts)
                base_rows = score_experimental(base_rows, experimental_artifacts)
                base_rows = add_flags(base_rows)
            merged_props = merge_scores_into_props(props_df, base_rows, unmatched_map)

    if "prop_type" not in merged_props.columns:
        merged_props["prop_type"] = config["market_key"] or config["display_name"]
    merged_props["display_prop_type"] = config["display_name"]
    merged_props["commence_time"] = pd.to_datetime(merged_props.get("commence_time"), utc=True, errors="coerce")
    schedule_by_key = {game["schedule_key"]: game for game in schedule_games}

    props_counts = merged_props.groupby("schedule_key").size().to_dict() if not merged_props.empty else {}
    matched_counts = (
        merged_props[merged_props["predicted_value_prod"].notna()].groupby("schedule_key").size().to_dict()
        if not merged_props.empty
        else {}
    )

    for game in schedule_games:
        game["props_count"] = int(props_counts.get(game["schedule_key"], 0))
        game["matched_props"] = int(matched_counts.get(game["schedule_key"], 0))

    known_schedule_keys = {game["schedule_key"] for game in schedule_games}
    if not merged_props.empty:
        for schedule_key, group in merged_props.groupby("schedule_key"):
            if schedule_key in known_schedule_keys:
                continue
            sample = group.iloc[0]
            schedule_games.append(
                {
                    "schedule_key": schedule_key,
                    "event_id": sample["event_id"],
                    "commence_time": sample["commence_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "home_team": sample["home_team"],
                    "away_team": sample["away_team"],
                    "broadcast": "",
                    "props_count": int(len(group)),
                    "matched_props": int(group["predicted_value_prod"].notna().sum()),
                }
            )
        schedule_games.sort(key=lambda row: row["commence_time"])

    if not merged_props.empty:
        merged_props["broadcast"] = merged_props["schedule_key"].map(lambda key: schedule_by_key.get(key, {}).get("broadcast", ""))
        merged_props["game_label"] = merged_props.apply(
            lambda row: f"{row['away_team']} @ {row['home_team']}",
            axis=1,
        )
        merged_props["best_edge_prod"] = pd.to_numeric(merged_props["best_edge_prod"], errors="coerce")
        merged_props = merged_props.sort_values(
            ["commence_time", "home_team", "away_team", "best_edge_prod"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
        merged_props["commence_time"] = merged_props["commence_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    summary = build_summary(schedule_games, merged_props, prop_type=prop_type)
    empty_test_frame = pd.DataFrame(columns=["predicted_value_prod", "excluded_from_test"])
    test_plan = build_test_plan(merged_props if not merged_props.empty else empty_test_frame, prop_type=prop_type)

    snapshot = {
        "generated_at": utcnow().isoformat().replace("+00:00", "Z"),
        "window_days": days,
        "window_label": f"Next {days} day{'s' if days != 1 else ''}",
        "prop_type": prop_type,
        "display_prop_type": config["display_name"],
        "games": schedule_games,
        "props": merged_props.to_dict(orient="records") if not merged_props.empty else [],
        "summary": summary,
        "test_plan": test_plan,
        "bookmakers": summary["bookmakers"],
        "experimental_available": summary["experimental_available"],
    }
    return sanitize_for_json(snapshot)


def save_snapshot(snapshot, prop_type="points"):
    json_path, csv_path = get_snapshot_paths(prop_type)
    sanitized_snapshot = sanitize_for_json(snapshot)
    props_df = pd.DataFrame(sanitized_snapshot["props"])
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(sanitized_snapshot, file, indent=2, allow_nan=False)
    props_df.to_csv(csv_path, index=False)


def load_saved_snapshot(prop_type="points"):
    json_path, _ = get_snapshot_paths(prop_type)
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as file:
        return sanitize_for_json(json.load(file))


def build_and_save_snapshot(prop_type="points", days=DEFAULT_WINDOW_DAYS, refresh_props=False, include_experimental=True, start_offset=1):
    snapshot = build_snapshot(
        prop_type=prop_type,
        days=days,
        refresh_props=refresh_props,
        include_experimental=include_experimental,
        start_offset=start_offset,
    )
    save_snapshot(snapshot, prop_type=prop_type)
    return snapshot
