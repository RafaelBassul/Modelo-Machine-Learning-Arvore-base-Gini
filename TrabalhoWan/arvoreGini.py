"""
Script principal para execução do pipeline completo de machine learning.
Carrega dados, treina árvore de decisão e gera visualização.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from arvore import construir_arvore
from dados import (
    adicionar_atributos_tecnicos,
    carregar_dados_petroleo,
    dividir_treino_teste,
)
from modelo import acuracia, prever
from visualizacao import plotar_arvore


def executar_pipeline(caminho_csv: Path, max_profundidade: Optional[int] = None) -> None:
    """
    Executa o pipeline completo: carregamento, preparação, treinamento e avaliação.
    
    Args:
        caminho_csv: Caminho para o arquivo CSV com dados da Bovespa
        max_profundidade: Profundidade máxima da árvore (None = sem limite)
    """
    print("Carregando dados de PETR3 e PETR4...")
    df = carregar_dados_petroleo(caminho_csv)
    df, atributos = adicionar_atributos_tecnicos(df)

    print(f"Total de registros apos engenharia de atributos: {len(df)}")
    print(f"Atributos utilizados: {', '.join(atributos)}")

    X = df[atributos].to_numpy()
    y = df["classe_alvo"].to_numpy()

    X_treino, y_treino, X_teste, y_teste = dividir_treino_teste(X, y)
    print(f"Tamanho treino: {len(X_treino)} | Tamanho teste: {len(X_teste)}")

    arvore = construir_arvore((X_treino, y_treino), max_profundidade=max_profundidade)
    y_pred = prever(arvore, X_teste)
    desempenho = acuracia(y_teste, y_pred)

    print(f"Acuracia no conjunto de teste: {desempenho:.3f}")
    print("\nGerando grafico da arvore de decisao...")
    caminho_fig = caminho_csv.with_name("arvore_decisao.png")
    plotar_arvore(arvore, caminho_saida=caminho_fig, nomes_atributos=atributos)


if __name__ == "__main__":
    caminho_csv = Path(__file__).with_name("Bovespa.csv")
    executar_pipeline(caminho_csv, max_profundidade=5)
