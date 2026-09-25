"""
Data pulls for the project: season Defensive Rating from nba_api (used once,
only to rank and pick the top 10 defenders) and player game logs from
nba_api (used for every train/test series). Both are cached to data/raw/ so
re-running the pipeline during development doesn't repeat network calls.
"""
from __future__ import annotations

import logging
import pathlib
import time

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


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


def get_defense_ranking_table(
    season: str = "2024-25", min_games: int = 40, force_refresh: bool = False
) -> pd.DataFrame:
    """
    Season-average Defensive Rating (points allowed per 100 possessions
    while the player is on court; lower is better) for every player, via
    stats.nba.com (`leaguedashplayerstats`, MeasureType=Advanced). One
    request, cached to data/raw/.

    We originally ranked by basketball-reference's Defensive Win Shares, but
    b-ref now blocks automated requests behind a Cloudflare JS challenge that
    can't be solved with plain HTTP headers. Defensive Rating is nba_api's
    closest built-in defensive metric and needs no scraping. It's a
    team-context stat (not purely individual), which is a known limitation
    of this ranking step -- see README.

    `min_games` filters out small-sample noise (e.g. a two-way/garbage-time
    player with a handful of appearances).
    """
    cache_path = RAW_DIR / f"def_rating_{season}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    from nba_api.stats.endpoints import leaguedashplayerstats

    def _pull():
        resp = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=60,
        )
        return resp.get_data_frames()[0]

    df = _retry(_pull)
    df = df[["PLAYER_ID", "PLAYER_NAME", "GP", "DEF_RATING"]].rename(
        columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "Player", "GP": "G"}
    )
    df = df[df["G"] >= min_games].sort_values("DEF_RATING").reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    return df


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
