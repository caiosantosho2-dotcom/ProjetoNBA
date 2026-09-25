"""
Runs the full comparison across the 20 series: 10 players (selected by
Defensive Rating) x 2 stats (STL, BLK). One reusable loop calling
fit_and_compare() per series, so no logic is duplicated across players/stats.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

from .data_sources import get_player_game_log
from .models import fit_and_compare
from .select_top10 import select_top10

logger = logging.getLogger(__name__)

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATS = ["STL", "BLK"]


def run_pipeline(
    season_train: str = "2024-25",
    season_test: str = "2025-26",
    test_n_games: int = 10,
    min_games: int = 40,
    save: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns (summary_df, records). `records` keeps the raw fitted params and
    count arrays (underscore-prefixed keys) needed for plotting; `summary_df`
    is the clean, CSV-ready table (those keys dropped).
    """
    top10 = select_top10(season=season_train, min_games=min_games, save=save)

    records: list[dict] = []
    for row in top10.itertuples(index=False):
        train_log = get_player_game_log(row.player_id, season_train)
        if train_log.empty:
            logger.warning("%s: sem game log em %s, serie(s) descartada(s)", row.Player, season_train)
            continue

        test_log = get_player_game_log(row.player_id, season_test)
        if test_log.empty:
            logger.warning(
                "%s: sem game log em %s (trade/lesao/saida da liga?) - "
                "series de teste ficam vazias para este jogador",
                row.Player,
                season_test,
            )

        for stat in STATS:
            train_counts = train_log[stat].to_numpy()
            test_counts = (
                test_log[stat].to_numpy()[:test_n_games] if not test_log.empty else np.array([])
            )
            record = fit_and_compare(train_counts, test_counts, row.Player, stat)
            records.append(record)

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in records])

    if save:
        summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    return summary_df, records


def summarize_pattern(summary_df: pd.DataFrame) -> dict:
    """High-level counts used for the README / LinkedIn write-up."""
    n_series = len(summary_df)
    train_counts = summary_df["winner_train_aic"].value_counts().to_dict()
    test_counts = (
        summary_df.dropna(subset=["winner_test"])["winner_test"].value_counts().to_dict()
    )
    vmr_by_stat = summary_df.groupby("stat")["vmr"].mean().to_dict()
    significant_overdispersion = int((summary_df["dispersion_chi2_pvalue"] < 0.05).sum())

    return {
        "n_series": n_series,
        "winner_train_counts": train_counts,
        "winner_test_counts": test_counts,
        "mean_vmr_by_stat": vmr_by_stat,
        "n_series_significant_overdispersion_p05": significant_overdispersion,
    }
