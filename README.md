# 📊 Resumo Executivo: Predição PETR3/PETR4 com CART

## 🎯 Arvore Gerada
![arvore](https://raw.githubusercontent.com/RafaelBassul/Modelo-Machine-Learning-Arvore-base-Gini/refs/heads/main/ArvoreGerada.png)

## 🎯 Objetivo
Prever se o fechamento das ações da Petrobras será **Alta** ou **Baixa** no dia seguinte, utilizando técnicas de Machine Learning em dados históricos da B3.

## 💾 Dados e Engenharia
* **Fonte:** Arquivo `Bovespa.csv` (28/09/2015 a 28/09/2016), filtrado exclusivamente para os tickers PETR3 e PETR4.
* **Dataset Final:** 253 registros válidos após o tratamento de dados e remoção de valores nulos.
* **Novos Atributos:** Foram gerados 5 indicadores técnicos para enriquecer o modelo, incluindo **Retorno Diário**, **Médias Móveis** (5 e 10 dias) e **Volatilidade**.

## ⚙️ O Modelo (CART)
* **Algoritmo:** Árvore de Decisão CART (*Classification and Regression Trees*) utilizando o **Índice de Gini** como métrica de impureza.
* **Estrutura:** A árvore foi configurada com profundidade máxima de 5 níveis para controlar a complexidade.
* **Funcionamento:** O modelo realiza divisões binárias recursivas buscando minimizar a impureza média ponderada dos nós resultantes.

## 📉 Resultados Chave
* **Atributos Decisivos:** O `retorno_diario` e a `volatilidade_5` foram as variáveis mais frequentes na árvore, indicando que o comportamento recente e o risco são os maiores preditores.
* **Acurácia:** **49,0%**. O desempenho foi estatisticamente equivalente ao acaso, indicando dificuldade em superar a eficiência de mercado com este modelo simples.
* **Viés de Predição:** A matriz de confusão mostrou uma tendência do modelo em prever a classe "Baixa", reflexo do desbalanceamento nos dados de treino (56,4% de amostras "Baixa").

## 💡 Conclusão
O trabalho foi eficaz na implementação modular do algoritmo e na estruturação do pipeline de dados, servindo como um exercício acadêmico robusto. Entretanto, como ferramenta financeira, o modelo sofreu devido à escassez de dados (apenas 1 ano) e à natureza não-linear do mercado, que exige algoritmos mais complexos.
