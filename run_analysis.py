"""
Ponto de entrada do projeto: roda o pipeline completo (top 10 defensores por
Defensive Win Shares -> game logs via nba_api -> Poisson vs. Binomial
Negativa nas 20 series) e gera a tabela-resumo, os graficos e um rascunho do
resumo em linguagem natural para o LinkedIn.

Uso:
    python run_analysis.py
"""
from __future__ import annotations

import logging
import pathlib

from src.pipeline import run_pipeline, summarize_pattern
from src.viz import plot_dispersion_overview, plot_top_interesting_cases

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

BASE_DIR = pathlib.Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "results" / "figures"


def _format_vmr_by_stat(vmr_by_stat: dict) -> str:
    return "; ".join(f"{stat}: {vmr:.2f}" for stat, vmr in sorted(vmr_by_stat.items()))


def _format_win_counts(counts: dict) -> str:
    return "; ".join(f"{model}: {n}" for model, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def _format_series_list(series: list[tuple]) -> str:
    """series: (player, stat, vmr[, pvalue]) tuples -> 'Player (STAT, VMR=x.xx)' list."""
    parts = []
    for row in series:
        player, stat, vmr = row[0], row[1], row[2]
        parts.append(f"{player} ({stat}, VMR={vmr:.2f})")
    return "; ".join(parts)


def write_linkedin_summary(pattern: dict, season_train: str, season_test: str) -> None:
    winner_train = pattern["winner_train_counts"]
    winner_test = pattern["winner_test_counts"]
    vmr_by_stat = pattern["mean_vmr_by_stat"]
    n_series = pattern["n_series"]
    n_significant = pattern["n_series_significant_overdispersion_p05"]
    n_negbin_train = winner_train.get("NegBin", 0)
    n_poisson_train = winner_train.get("Poisson", 0)
    test_lo, test_hi = pattern["test_n_games_range"]

    significant_list = _format_series_list(pattern["significant_overdispersion_series"])
    negbin_list = _format_series_list(pattern["negbin_aic_winner_series"])

    text = f"""# Resumo para o LinkedIn (rascunho)

Comparei Poisson vs. Binomial Negativa para prever roubos de bola (STL) e
tocos (BLK) por jogo dos 10 melhores defensores da NBA na temporada
{season_train} (ranking por Defensive Rating), validando contra os
primeiros jogos da temporada {season_test} ({test_lo} a {test_hi} jogos por
jogador). Ao todo, {n_series} series (jogador x estatistica) foram ajustadas.

**Achado principal: na maioria dos casos, Poisson ja basta.** No treino,
Poisson venceu no AIC em {n_poisson_train} das {n_series} series, e so
{n_negbin_train} exigiram Binomial Negativa. Overdispersion estatisticamente
significativa (teste formal, p<0.05) apareceu em apenas {n_significant} das
{n_series} series -- nao e a regra, e a excecao.

Mas quando aparece, e um risco real de subestimar volatilidade: {significant_list}.

**Fora da amostra (log-likelihood nos jogos de teste):**
{_format_win_counts(winner_test)} -- mais parelho que no treino, mas os testes
sao curtos ({test_lo}-{test_hi} jogos por jogador), entao essa comparacao tem
bem menos poder estatistico do que o AIC no treino e deve ser lida com essa
ressalva.

Series onde a Binomial Negativa venceu no AIC: {negbin_list}.

Razao variancia/media (VMR) medio por estatistica: {_format_vmr_by_stat(vmr_by_stat)}
-- overdispersion leve na media, concentrada em poucos jogadores/estatisticas,
nao um padrao geral.

**Por que isso importa na pratica:** a licao nao e "troque Poisson por
Binomial Negativa sempre". E "nao assuma Poisson sem checar" -- para a maioria
dos jogadores/estatisticas aqui, Poisson descreve bem o padrao de jogo a
jogo, mas para um subconjunto especifico (normalmente estatisticas de volume
baixo, como tocos de nao-pivos) a variancia real e maior do que a media
prevista, e ai um modelo Poisson para uma linha de over/under, fantasy ou
scouting vai subestimar a chance de jogos fora da curva.

---
*Rascunho gerado automaticamente a partir de results/summary_table.csv --
revise os numeros e ajuste o tom antes de publicar.*
"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "linkedin_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    season_train = "2024-25"
    season_test = "2025-26"

    logging.info("Rodando pipeline: top10 por Defensive Rating -> game logs -> Poisson vs NegBin (20 series)")
    summary_df, records = run_pipeline(season_train=season_train, season_test=season_test)

    if summary_df.empty:
        logging.error(
            "Pipeline nao retornou nenhuma serie. Verifique a conectividade com nba_api."
        )
        return

    pattern = summarize_pattern(summary_df)
    logging.info("Vencedor no treino (AIC): %s", pattern["winner_train_counts"])
    logging.info("Vencedor fora da amostra: %s", pattern["winner_test_counts"])
    logging.info("VMR medio por stat: %s", pattern["mean_vmr_by_stat"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_dispersion_overview(summary_df, FIGURES_DIR / "dispersion_overview.png")
    plot_top_interesting_cases(records, out_dir=FIGURES_DIR, n=4, season_train=season_train)

    write_linkedin_summary(pattern, season_train, season_test)

    logging.info(
        "Concluido. Tabela: results/summary_table.csv | Figuras: results/figures/ | "
        "Resumo: outputs/linkedin_summary.md"
    )


if __name__ == "__main__":
    main()
