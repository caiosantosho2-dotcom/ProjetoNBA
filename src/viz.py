"""
Plots for the summary README/LinkedIn write-up: static PNGs via matplotlib.

Palette (from the project's dataviz guidelines): categorical slot 1 (blue) /
slot 2 (orange) is a validated adjacent pair (CVD-safe in both light and dark
mode), reused consistently as the identity color across every chart --
Poisson vs. Negative Binomial in the per-series fit plots, and steals vs.
blocks in the dispersion overview.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from .models import negbin_logpmf

FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
PRIMARY_INK = "#0b0b0b"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

STAT_LABELS = {"STL": "Roubos de bola (STL)", "BLK": "Tocos (BLK)"}


def _apply_base_style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_fit_comparison(record: dict, out_path: pathlib.Path, season_train: str = "2024-25") -> None:
    """
    Observed per-game counts (bars) vs. fitted Poisson and Negative Binomial
    PMFs (stems) for one player/stat series -- the shape of the mismatch is
    the whole point, so both fitted curves stay visible against the data.
    """
    train_counts = record["_train_counts"]
    max_count = int(max(train_counts.max(), record["poisson_lambda"] * 2.5, 3))
    x = np.arange(0, max_count + 1)

    poisson_pmf = sp_stats.poisson.pmf(x, record["poisson_lambda"])
    negbin_pmf = np.exp(negbin_logpmf(x, record["negbin_mu"], record["negbin_alpha"]))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    _apply_base_style(ax)

    bins = np.arange(-0.5, max_count + 1.5, 1)
    ax.hist(
        train_counts, bins=bins, density=True, color=GRIDLINE, edgecolor=SURFACE,
        linewidth=1.5, zorder=2, label="Observado",
    )
    ax.plot(x, poisson_pmf, "o-", color=BLUE, linewidth=2, markersize=5, zorder=3,
            label=f"Poisson (AIC={record['poisson_aic']:.1f})")
    ax.plot(x, negbin_pmf, "o-", color=ORANGE, linewidth=2, markersize=5, zorder=3,
            label=f"Bin. Negativa (AIC={record['negbin_aic']:.1f})")

    ax.set_xlabel(f"{STAT_LABELS.get(record['stat'], record['stat'])} por jogo", color=PRIMARY_INK)
    ax.set_ylabel("Densidade", color=PRIMARY_INK)
    ax.set_title(
        f"{record['player']} — {STAT_LABELS.get(record['stat'], record['stat'])}\n"
        f"Temporada {season_train} (treino), n={record['n_train']} jogos",
        color=PRIMARY_INK, fontsize=11, loc="left",
    )
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_dispersion_overview(summary_df, out_path: pathlib.Path) -> None:
    """
    One dot per series: variance/mean ratio (VMR), grouped by player and
    colored by stat. VMR=1 is the Poisson expectation; points well above the
    reference line are the overdispersed series where Negative Binomial
    should fit better.
    """
    df = summary_df.sort_values(["player", "stat"]).reset_index(drop=True)
    players = list(dict.fromkeys(df["player"]))
    y_pos = {player: i for i, player in enumerate(players)}

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=150)
    _apply_base_style(ax)
    ax.axvline(1.0, color=MUTED_INK, linewidth=1, linestyle="--", zorder=1)
    ax.text(1.0, len(players) - 0.3, " VMR = 1 (Poisson)", color=MUTED_INK, fontsize=8, va="bottom")

    for stat, color, offset in (("STL", BLUE, 0.12), ("BLK", ORANGE, -0.12)):
        sub = df[df["stat"] == stat]
        ys = [y_pos[p] + offset for p in sub["player"]]
        ax.scatter(sub["vmr"], ys, color=color, s=60, zorder=3,
                    label=STAT_LABELS.get(stat, stat), edgecolor=SURFACE, linewidth=1)

    ax.set_yticks(range(len(players)))
    ax.set_yticklabels(players, color=PRIMARY_INK, fontsize=9)
    ax.set_xlabel("Razao variancia / media (VMR) -- treino 2024-25", color=PRIMARY_INK)
    ax.set_title("Overdispersion por jogador e estatistica", color=PRIMARY_INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_top_interesting_cases(records: list[dict], out_dir: pathlib.Path = FIG_DIR, n: int = 4,
                                season_train: str = "2024-25") -> list[pathlib.Path]:
    """Plots the `n` series with the largest |AIC(Poisson) - AIC(NegBin)| gap."""
    ranked = sorted(records, key=lambda r: abs(r["aic_diff_poisson_minus_negbin"]), reverse=True)
    paths = []
    for rec in ranked[:n]:
        fname = f"fit_{rec['player'].replace(' ', '_')}_{rec['stat']}.png"
        out_path = out_dir / fname
        plot_fit_comparison(rec, out_path, season_train=season_train)
        paths.append(out_path)
    return paths
