"""
Picks the top 10 defenders of a season by Defensive Rating, straight from
nba_api -- no cross-source name resolution needed, since the ranking already
carries the nba_api player_id.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from .data_sources import get_defense_ranking_table

PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def select_top10(season: str = "2024-25", min_games: int = 40, save: bool = True) -> pd.DataFrame:
    """
    Returns the top 10 defenders of `season` by Defensive Rating (lower is
    better), each with its nba_api player_id.
    """
    ranking = get_defense_ranking_table(season=season, min_games=min_games)
    top10 = ranking.head(10).reset_index(drop=True)[["Player", "player_id", "DEF_RATING", "G"]]

    if save:
        top10.to_csv(PROCESSED_DIR / "top10.csv", index=False)

    return top10
