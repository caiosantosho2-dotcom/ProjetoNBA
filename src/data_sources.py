"""
Data pulls for the project: Defensive Win Shares from basketball-reference
(used once, only to rank and pick the top 10 defenders) and player game logs
from nba_api (used for every train/test series). Both are cached to
data/raw/ so re-running the pipeline during development doesn't repeat
network calls.
"""
from __future__ import annotations

import io
import logging
import pathlib
import re
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Identified, low-volume (single request) User-Agent for a portfolio/research pull.
BREF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; nba-poisson-negbin-case-study/1.0; "
        "personal portfolio project, single request per run)"
    )
}


def _season_to_bref_year(season: str) -> int:
    """basketball-reference labels a season by its END year: '2024-25' -> 2025."""
    start_year = int(season.split("-")[0])
    return start_year + 1


def get_dws_table(season: str = "2024-25", min_games: int = 40, force_refresh: bool = False) -> pd.DataFrame:
    """
    Season 'Advanced' stats table from basketball-reference (includes
    Defensive Win Shares). One HTTP request, cached to data/raw/.

    Players traded mid-season appear as a 'TOT' row (combined stats) plus one
    row per team; we keep only the combined row to avoid double counting.
    `min_games` filters out small-sample noise (e.g. a player who appeared in
    only a handful of games).
    """
    year = _season_to_bref_year(season)
    cache_path = RAW_DIR / f"bref_advanced_{year}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    resp = requests.get(url, headers=BREF_HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    tables = pd.read_html(io.StringIO(html), attrs={"id": "advanced"})
    if not tables:
        # basketball-reference sometimes ships secondary tables inside HTML
        # comments to discourage naive scraping; unwrap and retry once.
        uncommented = re.sub(r"<!--|-->", "", html)
        tables = pd.read_html(io.StringIO(uncommented), attrs={"id": "advanced"})
    if not tables:
        raise RuntimeError(f"Could not find the 'advanced' stats table at {url}")

    df = tables[0]
    df = df[df["Player"] != "Player"].copy()  # drop repeated mid-table header rows

    team_col = "Team" if "Team" in df.columns else "Tm"
    df["G"] = pd.to_numeric(df["G"], errors="coerce")
    df["DWS"] = pd.to_numeric(df["DWS"], errors="coerce")
    df = df.dropna(subset=["G", "DWS"])

    has_tot = df.groupby("Player")[team_col].transform(lambda s: (s == "TOT").any())
    df = df[~has_tot | (df[team_col] == "TOT")]

    df = df[df["G"] >= min_games].reset_index(drop=True)
    df = df.sort_values("DWS", ascending=False).reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    return df


def _retry(fn, attempts: int = 3, base_delay: float = 1.5):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # nba_api / network calls can be flaky
            last_exc = exc
            logger.warning("attempt %d/%d failed: %s", i + 1, attempts, exc)
            time.sleep(base_delay * (i + 1))
    raise last_exc


def get_player_game_log(
    player_id: int,
    season: str,
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Game-by-game log for one player/season via nba_api, sorted chronologically
    (ascending GAME_DATE). Cached to data/raw/.
    """
    cache_path = RAW_DIR / f"gamelog_{player_id}_{season}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, parse_dates=["GAME_DATE"])

    from nba_api.stats.endpoints import playergamelog

    def _pull():
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60,
        )
        return log.get_data_frames()[0]

    df = _retry(_pull)
    time.sleep(0.6)  # be polite to stats.nba.com between calls

    if df.empty:
        return df

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df
