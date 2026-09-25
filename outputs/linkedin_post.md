# Post para o LinkedIn

Poisson resolve qualquer estatística de contagem no esporte? Resolvi testar
com dados reais.

Peguei os 10 melhores defensores da NBA na temporada 2024-25 (ranking por
Defensive Rating) e ajustei dois modelos para prever roubos de bola e tocos
por jogo de cada um: Poisson, o modelo "padrão" para contar eventos, e
Binomial Negativa, que permite mais volatilidade do que a média sozinha
explicaria.

20 combinações jogador x estatística no total, usando a temporada inteira
como treino. O resultado, sinceramente, surpreendeu pela falta de drama: na
maioria dos casos — 17 de 20 — o modelo simples já dá conta do recado.

Mas em dois casos específicos, a variância real ficou claramente acima do
que Poisson previa, o suficiente para passar num teste estatístico formal:

- Kenrich Williams, tocos por jogo — variância ~48% maior que o esperado
- Nicolas Batum, tocos por jogo — variância ~39% maior que o esperado

Na prática: para esses dois, um modelo que só olha a média subestima a
chance de "jogos fora da curva" — 3 tocos numa noite, zero na outra. Para
quem usa esse tipo de projeção em apostas esportivas, fantasy ou scouting,
isso importa — a cauda da distribuição é onde o risco (e a oportunidade)
mora.

A lição que fica não é "troque sempre pro modelo mais complexo". É "teste
antes de assumir": na maioria das vezes o simples já resolve, e saber
identificar a minoria de casos em que não resolve é o que separa uma
projeção ingênua de uma calibrada.

Código, dados e gráficos completos no GitHub:
https://github.com/caiosantosho2-dotcom/ProjetoNBA

---
*Rascunho — ajuste o tom/comprimento antes de publicar. Os números batem
com results/summary_table.csv (regenerado por `python run_analysis.py`).*
