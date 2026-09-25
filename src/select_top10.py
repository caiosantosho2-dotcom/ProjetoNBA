"""
Ranks players by Defensive Win Shares (DWS) and resolves each one to an
nba_api player_id, so the rest of the pipeline can pull game logs.
"""
from __future__ import annotations

import pathlib
import unicodedata

import pandas as pd

from .data_sources import get_dws_table

PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# basketball-reference spelling -> nba_api spelling, for the rare case the
# automatic normalized match below doesn't find a hit (e.g. suffix or accent
# handled differently between sources). Extend this if select_top10() warns
# about unresolved names.
MANUAL_NAME_OVERRIDES: dict[str, str] = {}


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.replace(".", "").replace("*", "").strip().lower()
    return " ".join(name.split())


def _resolve_player_id(bref_name: str, nba_players: list[dict]) -> int | None:
    target = MANUAL_NAME_OVERRIDES.get(bref_name, bref_name)
    target_norm = _normalize(target)
    for p in nba_players:
        if _normalize(p["full_name"]) == target_norm:
            return p["id"]
    return None


def select_top10(season: str = "2024-25", min_games: int = 40, save: bool = True) -> pd.DataFrame:
    """
    Returns the top 10 defenders of `season` by DWS, each with a resolved
    nba_api player_id. Players whose bref name can't be matched to nba_api
    are skipped (with a warning) in favor of the next-highest DWS player, so
    the result always has 10 rows (given enough candidates).
    """
    from nba_api.stats.static import players as nba_players_module

    dws_table = get_dws_table(season=season, min_games=min_games)
    nba_players = nba_players_module.get_players()

    dws_table = dws_table.copy()
    dws_table["player_id"] = dws_table["Player"].apply(lambda n: _resolve_player_id(n, nba_players))

    unresolved = dws_table[dws_table["player_id"].isna()]
    if not unresolved.empty:
        print(
            "Aviso: nomes nao resolvidos para nba_api "
            "(adicione um mapeamento em MANUAL_NAME_OVERRIDES se algum "
            "estiver entre os top 10):"
        )
        print(unresolved["Player"].head(20).to_list())

    resolved = dws_table.dropna(subset=["player_id"]).copy()
    resolved["player_id"] = resolved["player_id"].astype(int)

    top10 = resolved.head(10).reset_index(drop=True)[["Player", "player_id", "DWS", "G"]]

    if save:
        top10.to_csv(PROCESSED_DIR / "top10.csv", index=False)

    return top10
