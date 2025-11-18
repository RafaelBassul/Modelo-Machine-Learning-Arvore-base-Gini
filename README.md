# 📊 Explicação Detalhada: Como Funciona o Modelo de Árvore de Decisão

Executar: arvoreGini.py

## 🎯 Objetivo do Projeto

Este projeto usa **Machine Learning** para prever se o preço de fechamento das ações da Petrobras (PETR3 e PETR4) vai **subir** (Alta) ou **descer** (Baixa) no próximo dia de negociação, baseado em indicadores técnicos calculados a partir dos dados históricos.

---

## 📈 Parte 1: Entendendo os Dados do CSV

### O que temos no arquivo Bovespa.csv?

O arquivo contém dados históricos de ações da B3 (Bolsa de Valores brasileira) de **28/09/2015 a 28/09/2016**, com aproximadamente **14.977 registros** de diferentes empresas.

**Estrutura de cada linha:**
- **Date**: Data da negociação
- **Ticker**: Símbolo da ação (ex: PETR3, PETR4, VALE3, etc.)
- **Open**: Preço de abertura do dia
- **High**: Maior preço negociado no dia
- **Low**: Menor preço negociado no dia
- **Close**: Preço de fechamento do dia
- **Volume**: Quantidade de ações negociadas

### Filtragem dos Dados

O código filtra apenas os registros das ações **PETR3** e **PETR4** (Petrobras), resultando em **262 registros** iniciais.

**Estatísticas básicas dos dados filtrados:**
- Preço médio de fechamento: **R$ 10,69**
- Preço mínimo: **R$ 5,91**
- Preço máximo: **R$ 16,39**
- Volume médio negociado: **13,2 milhões** de ações

---

## 🔧 Parte 2: Preparação dos Dados (Engenharia de Atributos)

### Por que criar novos atributos?

Os dados brutos (Open, High, Low, Close, Volume) são importantes, mas **indicadores técnicos** derivados deles capturam padrões mais complexos que ajudam na previsão.

### Atributos Criados:

1. **retorno_diario**: Variação percentual do preço de fechamento em relação ao dia anterior
   - Fórmula: `(Close hoje - Close ontem) / Close ontem`
   - Indica se houve ganho ou perda no dia

2. **mm_5_close**: Média móvel de 5 dias do preço de fechamento
   - Média dos últimos 5 preços de fechamento
   - Indica tendência de curto prazo

3. **mm_10_close**: Média móvel de 10 dias do preço de fechamento
   - Média dos últimos 10 preços de fechamento
   - Indica tendência de médio prazo

4. **volatilidade_5**: Desvio padrão dos retornos diários dos últimos 5 dias
   - Mede a instabilidade/risco do ativo
   - Valores altos = maior incerteza

5. **amplitude_pct**: Amplitude percentual do dia
   - Fórmula: `(High - Low) / Open`
   - Indica a variação máxima do preço durante o dia

### Variável Alvo (O que queremos prever)

**classe_alvo**: Indica se o preço vai subir ou descer no próximo dia
- **"Alta"**: Se o fechamento do dia seguinte > fechamento atual
- **"Baixa"**: Se o fechamento do dia seguinte ≤ fechamento atual

**Exemplo prático:**
- Se hoje a ação fecha a R$ 10,00 e amanhã fecha a R$ 10,50 → Classe = "Alta"
- Se hoje a ação fecha a R$ 10,00 e amanhã fecha a R$ 9,80 → Classe = "Baixa"

### Resultado da Preparação

Após criar os indicadores e remover valores faltantes, temos:
- **253 registros** válidos
- **10 atributos** para previsão
- Distribuição das classes: ~56% "Baixa" e ~44% "Alta" (dados levemente desbalanceados)

---

## 🌳 Parte 3: Como Funciona a Árvore de Decisão

### Conceito Básico

Imagine uma **árvore de decisão** como um **fluxograma** que faz perguntas sequenciais sobre os dados e, no final, chega a uma conclusão (predição).

**Exemplo do dia a dia:**
```
Pergunta 1: Está chovendo?
  ├─ SIM → Pergunta 2: Tenho guarda-chuva?
  │         ├─ SIM → Vou sair
  │         └─ NÃO → Fico em casa
  └─ NÃO → Vou sair
```

### O Algoritmo CART (Classification and Regression Trees)

Nosso código usa o algoritmo **CART** com o critério de **Índice de Gini** para construir a árvore.

#### Passo 1: Medir a "Impureza" (Índice de Gini)

O **Índice de Gini** mede o quão "misturadas" estão as classes em um conjunto de dados:
- **Gini = 0.0**: Nó "puro" (todas as amostras têm a mesma classe)
- **Gini = 0.5**: Máxima impureza (distribuição uniforme entre duas classes)
- **Gini próximo de 0**: Maior pureza (mais fácil de classificar)

**Exemplo:**
- Se temos 100 amostras e todas são "Alta" → Gini = 0.0 (puro)
- Se temos 50 "Alta" e 50 "Baixa" → Gini = 0.5 (máxima impureza)
- Se temos 80 "Alta" e 20 "Baixa" → Gini = 0.32 (relativamente puro)

#### Passo 2: Encontrar a Melhor Divisão

Para cada nó, o algoritmo:
1. Testa **todos os atributos** (Open, High, Low, Close, Volume, retorno_diario, etc.)
2. Para cada atributo, testa **todos os valores possíveis** como ponto de corte
3. Escolhe a divisão que **minimiza a impureza média ponderada**

**Exemplo de divisão:**
```
Nó inicial: 100 amostras (50 Alta, 50 Baixa) - Gini = 0.5

Teste: "retorno_diario <= -0.02"
  ├─ SIM (esquerda): 30 amostras (5 Alta, 25 Baixa) - Gini = 0.28
  └─ NÃO (direita): 70 amostras (45 Alta, 25 Baixa) - Gini = 0.46
  
Impureza média = (30/100) × 0.28 + (70/100) × 0.46 = 0.406
```

Se essa divisão reduzir a impureza, ela é escolhida!

#### Passo 3: Construir a Árvore Recursivamente

O processo se repete para cada subconjunto criado, até que:
- Todas as amostras no nó tenham a mesma classe (nó puro)
- Atinga a profundidade máxima (no nosso caso, 5 níveis)
- Tenha poucas amostras (menos de 2)

#### Passo 4: Fazer Predições

Para prever uma nova amostra:
1. Começa na raiz da árvore
2. Em cada nó, verifica a condição (ex: "retorno_diario <= -0.02?")
3. Se SIM, vai para a esquerda; se NÃO, vai para a direita
4. Repete até chegar em uma folha (nó terminal)
5. A classe da folha é a predição

---

## 📊 Parte 4: Resultados e Análise

### Importância dos Atributos

Baseado na análise da árvore construída, os atributos mais usados foram:

1. **retorno_diario** (3 vezes) - ⭐ **MAIS IMPORTANTE**
   - O retorno do dia atual é o melhor indicador do movimento futuro
   - Se o retorno foi negativo, há maior chance de queda no próximo dia

2. **volatilidade_5** (3 vezes) - ⭐ **MUITO IMPORTANTE**
   - A instabilidade recente influencia a direção do preço
   - Alta volatilidade pode indicar incerteza

3. **Open, High, amplitude_pct** (2 vezes cada)
   - Preços de abertura e máxima, além da amplitude do dia, também são relevantes

4. **Close, Volume, mm_5_close, mm_10_close** (1 vez cada)
   - Importantes, mas menos decisivos

5. **Low** (0 vezes)
   - Não foi usado na árvore (menos relevante para este problema)

### Desempenho do Modelo

**Acurácia: 49.0%** (49 acertos em 100 tentativas)

**Análise:**
- A acurácia está próxima de **50%**, que é o desempenho de um "chute aleatório" para um problema binário balanceado
- Isso indica que **prever movimentos de preço de ações é extremamente difícil**
- O mercado financeiro tem muitos fatores externos (notícias, eventos, sentimentos) que não estão nos dados técnicos

**Matriz de Confusão:**
```
                Predito
              Alta  Baixa
Verdadeiro Alta   10    16
          Baixa   10    15
```

**Interpretação:**
- O modelo acertou **25 predições** (10+15) e errou **26** (16+10)
- Há uma tendência de prever mais "Baixa" do que "Alta"
- Isso pode estar relacionado ao desbalanceamento dos dados (56% Baixa vs 44% Alta)

### Exemplos de Predição

**Exemplo 1 - Predição Correta:**
```
Caminho na árvore:
  Nível 0: mm_10_close <= 9.334? (valor=12.916) → NÃO
  Nível 1: retorno_diario <= -0.023? (valor=0.012) → NÃO
  Nível 2: High <= 9.710? (valor=13.950) → NÃO
  Nível 3: amplitude_pct <= 0.056? (valor=0.028) → SIM
  Nível 4: retorno_diario <= 0.018? (valor=0.012) → SIM
  → Predição final: Baixa

Classe real: Baixa ✓ CORRETO
```

**Interpretação:**
- A média móvel de 10 dias estava alta (12.916 > 9.334)
- O retorno do dia foi positivo mas pequeno (0.012)
- O preço máximo estava alto (13.950)
- A amplitude foi pequena (0.028 ≤ 0.056)
- O retorno foi positivo mas baixo (0.012 ≤ 0.018)
- **Conclusão**: Mesmo com alguns indicadores positivos, o modelo previu "Baixa" e acertou!

---

## 🎓 Parte 5: Por Que Esses Resultados?

### Fatores que Influenciaram o Modelo

1. **retorno_diario é o mais importante**
   - Faz sentido: se a ação teve um bom retorno hoje, pode continuar subindo amanhã (ou pode haver correção)
   - O mercado financeiro tem "momentum" (tendência de continuidade) e "reversão à média"

2. **volatilidade_5 é muito relevante**
   - Períodos de alta volatilidade indicam incerteza
   - Pode sinalizar mudanças de tendência

3. **Médias móveis (mm_5_close, mm_10_close)**
   - Usadas para identificar tendências
   - Quando o preço está acima da média, pode indicar força
   - Quando está abaixo, pode indicar fraqueza

4. **Amplitude percentual**
   - Dias com grande variação podem indicar indecisão do mercado
   - Pode preceder movimentos maiores

### Limitações do Modelo

1. **Mercado financeiro é complexo**
   - Muitos fatores externos (notícias, política, economia global)
   - Comportamento não-linear e caótico
   - Eficiência de mercado (preços já refletem informações disponíveis)

2. **Dados limitados**
   - Apenas 1 ano de dados (253 registros válidos)
   - Período específico (2015-2016) pode não representar outros períodos
   - Apenas indicadores técnicos (não considera fundamentos)

3. **Profundidade da árvore**
   - Limitada a 5 níveis (para evitar overfitting)
   - Pode não capturar padrões mais complexos

4. **Acurácia de 49%**
   - Próxima do acaso (50%)
   - Indica que prever movimentos de preço é extremamente difícil
   - Em finanças, até modelos sofisticados têm dificuldade para superar o acaso

---

## 💡 Conclusão

Este projeto demonstra:
- ✅ Como preparar dados financeiros para Machine Learning
- ✅ Como construir uma árvore de decisão do zero
- ✅ Como avaliar o desempenho de um modelo
- ✅ Quais indicadores técnicos são mais relevantes

**Principais aprendizados:**
1. **retorno_diario** e **volatilidade_5** são os atributos mais importantes
2. Prever movimentos de preço é muito difícil (acurácia próxima do acaso)
3. O modelo identifica padrões, mas o mercado financeiro tem muitos fatores não capturados pelos dados técnicos

**Possíveis melhorias:**
- Usar mais dados históricos
- Adicionar indicadores fundamentais (lucro, receita, etc.)
- Testar outros algoritmos (Random Forest, XGBoost, etc.)
- Considerar dados de outras fontes (notícias, sentimentos, etc.)

---

## 📚 Glossário

- **Árvore de Decisão**: Algoritmo de Machine Learning que faz perguntas sequenciais
- **Gini**: Medida de impureza (quanto menor, mais puro o nó)
- **CART**: Algoritmo para construir árvores de decisão
- **Média Móvel**: Média dos últimos N valores
- **Volatilidade**: Medida de variação/risco
- **Acurácia**: Porcentagem de predições corretas
- **Overfitting**: Modelo que "decora" os dados de treino mas não generaliza bem
