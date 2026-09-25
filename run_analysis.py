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


def write_linkedin_summary(pattern: dict, season_train: str, season_test: str) -> None:
    winner_train = pattern["winner_train_counts"]
    winner_test = pattern["winner_test_counts"]
    vmr_by_stat = pattern["mean_vmr_by_stat"]
    higher_stat = max(vmr_by_stat, key=vmr_by_stat.get) if vmr_by_stat else None
    lower_stat = min(vmr_by_stat, key=vmr_by_stat.get) if vmr_by_stat else None

    text = f"""# Resumo para o LinkedIn (rascunho)

Comparei Poisson vs. Binomial Negativa para prever roubos de bola (STL) e
tocos (BLK) por jogo dos 10 melhores defensores da NBA na temporada
{season_train} (ranking por Defensive Rating), validando contra os
primeiros jogos da temporada {season_test}. Ao todo, {pattern['n_series']}
series (jogador x estatistica) foram ajustadas.

**Resultado no treino (AIC):** {_format_win_counts(winner_train)}
**Resultado fora da amostra (log-likelihood nos jogos de teste):** {_format_win_counts(winner_test)}

Em {pattern['n_series_significant_overdispersion_p05']} das {pattern['n_series']}
series, o teste formal de overdispersion rejeitou Poisson a 5% de
significancia -- ou seja, a variancia observada foi maior do que a media
prevista pela Poisson com mais frequencia do que o esperado ao acaso.

Razao variancia/media (VMR) media por estatistica: {_format_vmr_by_stat(vmr_by_stat)}.
{f"{higher_stat} mostrou, em media, maior overdispersion do que {lower_stat} neste recorte de jogadores." if higher_stat else ""}

**Por que isso importa na pratica:** quem usa esse tipo de previsao para
apostas esportivas (linhas de over/under), fantasy ou scouting assume
implicitamente uma distribuicao ao trabalhar so com a media. Se a estatistica
real e overdispersed e o modelo assume Poisson, a probabilidade de "jogos
explosivos" (bem acima da media) fica subestimada -- o que tem custo direto
em decisao de risco.

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
