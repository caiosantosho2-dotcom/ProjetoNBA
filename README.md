# Poisson vs. Binomial Negativa: prevendo estatísticas defensivas na NBA

Case study de ciência de dados aplicada a estatística esportiva. Pergunta central:
**quando prevemos roubos de bola (steals) e tocos (blocks) por jogo de um jogador,
Poisson é suficiente ou a variância real exige uma Binomial Negativa?**

## Por que isso importa na prática (e não é só exercício acadêmico)

Qualquer previsão de "quantos X por jogo" — para uma linha de aposta esportiva
(over/under), uma projeção de fantasy, ou uma nota de scouting — assume,
implícita ou explicitamente, uma distribuição de probabilidade em torno da média.

- **Poisson** assume que a variância é igual à média. É a escolha "padrão" para
  contagens, simples e com um único parâmetro.
- **Binomial Negativa** permite variância maior que a média (*overdispersion*),
  o que é comum quando o processo gerador tem mais volatilidade do que um
  processo Poisson "puro" explicaria — por exemplo, um jogador com jogos de
  "explosão defensiva" ocasionais intercalados com jogos discretos.

Se a estatística real é overdispersed e o modelo assume Poisson, a
probabilidade de resultados extremos (acima ou abaixo da média) é
**subestimada**. Isso tem custo direto: uma casa de apostas ou um jogador de
fantasy que usa Poisson onde deveria usar Binomial Negativa está
sistematicamente mal calibrado para o risco de "jogos fora da curva".

## Escopo

- **Top 10 defensores da temporada 2024-25**, selecionados por *Defensive
  Rating* (pontos permitidos por 100 posses com o jogador em quadra, menor é
  melhor) — métrica de contexto de time exposta pelo próprio `nba_api`, em vez
  de olhar só o volume bruto de steals+blocks.
- Para cada um dos 10 jogadores: **20 séries no total** (steals e blocks
  separadamente).
- **Treino:** todos os jogos da temporada 2024-25 de cada jogador.
- **Teste/validação:** os primeiros jogos da temporada 2025-26 de cada jogador
  (fora da amostra de treino).

## Metodologia

1. **Seleção do top 10** — `leaguedashplayerstats` (MeasureType=Advanced) do
   `nba_api`, ranqueando por `DEF_RATING` e filtrando jogadores com poucos
   jogos (ruído de amostra pequena).
2. **Game logs jogo-a-jogo** via [`nba_api`](https://github.com/swar/nba_api)
   (`PlayerGameLog`), para 2024-25 (treino) e para os primeiros N jogos de
   2025-26 (teste). N é configurável (`run_pipeline(test_n_games=10)`).
3. **Ajuste por MLE** de Poisson (intercept-only: λ = média do treino) e
   Binomial Negativa NB2 (`statsmodels.NegativeBinomial`) em cada uma das 20
   séries de treino.
4. **Teste de overdispersion** por série:
   - razão variância/média (VMR) — diagnóstico simples, sempre reportado;
   - teste qui-quadrado formal de dispersão de Poisson
     (`sum((y - média)² ) / média ~ χ²(n-1)` sob H0: Poisson);
   - *likelihood-ratio test* Poisson vs. Binomial Negativa (α=0 é o caso
     Poisson, aninhado). Como α=0 está na fronteira do espaço de parâmetros,
     o p-valor usa a correção de Self & Liang (1987): mistura 50/50 de
     χ²(0) e χ²(1), em vez do χ²(1) ingênuo.
5. **Comparação de ajuste**: AIC no treino (menor vence) **e** validação fora
   da amostra — log-likelihood dos dois modelos treinados avaliado nos jogos
   de teste de 2025-26, para checar se o "vencedor" no treino generaliza.
6. **Consolidação**: quantas das 20 séries cada modelo venceu (treino vs.
   teste), e se existe um padrão consistente entre steals e blocks.

## Decisões técnicas relevantes

- **Defensive Rating via nba_api, não Defensive Win Shares via
  basketball-reference**: a versão original desta seleção usava DWS
  (basketball-reference), mas o site passou a bloquear requisições
  automatizadas atrás de um desafio JS do Cloudflare, que não dá pra resolver
  com headers HTTP simples. Trocamos para Defensive Rating, exposto direto
  pelo `nba_api` — elimina scraping por completo, ao custo de ser uma métrica
  de contexto de time (rating defensivo do time com o jogador em quadra), não
  puramente individual como DWS.
- **Jogadores sem dados em 2025-26** (aposentadoria, lesão prolongada, saída da
  liga) são logados e excluídos apenas das séries de teste correspondentes —
  não derrubam o pipeline nem são silenciosamente ignorados.
- **Correção de fronteira no LRT**: reportar um p-valor de χ²(1) sem essa
  correção infla artificialmente a significância da Binomial Negativa contra
  Poisson; a correção de Self & Liang é o tratamento padrão para esse teste
  aninhado nesse caso específico.
- **Cache local em `data/raw/`**: evita repetir chamadas de API durante o
  desenvolvimento/debug do pipeline. Não é versionado (ver `.gitignore`) por
  ser inteiramente regenerável.
- **Limitação conhecida**: cada série tem, no máximo, ~82 observações de
  treino (uma temporada) — suficiente para estimar média e VMR com razoável
  confiança, mas o parâmetro α da Binomial Negativa pode ter intervalo de
  confiança amplo em séries com poucos eventos (ex.: um jogador com poucos
  blocks por jogo). Isso é esperado e discutido nos resultados, não escondido.

## Estrutura do repositório

```
├── src/
│   ├── data_sources.py    # pulls nba_api: defensive rating + game logs
│   ├── select_top10.py    # rankeia por Defensive Rating (nba_api)
│   ├── models.py          # fit Poisson/NegBin, AIC, teste de overdispersion
│   ├── pipeline.py        # loop reutilizavel sobre as 20 series
│   └── viz.py             # graficos dos casos mais interessantes
├── run_analysis.py        # script principal: roda tudo, gera outputs/
├── data/
│   ├── raw/                # cache local (nao versionado)
│   └── processed/          # top10.csv
├── results/
│   ├── summary_table.csv   # tabela consolidada das 20 series
│   └── figures/             # PNGs
└── outputs/
    └── linkedin_summary.md # resumo em linguagem natural (gerado ao rodar)
```

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python run_analysis.py
```

Isso gera `results/summary_table.csv`, os gráficos em `results/figures/` e um
rascunho de resumo em `outputs/linkedin_summary.md`.

## Resultados

_A rodar — `results/summary_table.csv` e `outputs/linkedin_summary.md` são
gerados executando `python run_analysis.py`; este README será atualizado com
o resumo real dos resultados depois da primeira execução completa._
