"""
Módulo para carregamento e preparação de dados financeiros.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def carregar_dados_petroleo(caminho_csv: Path) -> pd.DataFrame:
    """
    Carrega e filtra dados das ações PETR3 e PETR4 do arquivo CSV.
    """
    df = pd.read_csv(
        caminho_csv,
        parse_dates=["Date"],
        dayfirst=True,
        decimal=",",
        dtype={"Ticker": "string"},
    )
    df = df[df["Ticker"].isin(["PETR3", "PETR4"])].copy()
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    colunas_utilizadas = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    df = df[colunas_utilizadas]
    colunas_numericas = ["Open", "High", "Low", "Close", "Volume"]
    df[colunas_numericas] = df[colunas_numericas].apply(
        pd.to_numeric, errors="coerce"
    )
    df.dropna(subset=colunas_numericas, inplace=True)
    return df


def adicionar_atributos_tecnicos(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adiciona indicadores técnicos e cria a variável alvo.
    
    A variável alvo "classe_alvo" indica se o preço de fechamento do dia seguinte
    será maior ("Alta") ou menor/igual ("Baixa") ao preço de fechamento atual.
    """
    agrupado = df.groupby("Ticker", group_keys=False)
    df["retorno_diario"] = agrupado["Close"].pct_change()
    df["mm_5_close"] = agrupado["Close"].transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    df["mm_10_close"] = agrupado["Close"].transform(lambda s: s.rolling(window=10, min_periods=10).mean())
    df["volatilidade_5"] = agrupado["retorno_diario"].transform(lambda s: s.rolling(window=5, min_periods=5).std())
    df["amplitude_pct"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    
    df["alvo_futuro"] = agrupado["Close"].shift(-1)
    df["classe_alvo"] = np.where(df["alvo_futuro"] > df["Close"], "Alta", "Baixa")
    
    df = df.dropna(subset=["mm_5_close", "mm_10_close", "volatilidade_5", "retorno_diario", "classe_alvo"]).copy()

    atributos = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "retorno_diario",
        "mm_5_close",
        "mm_10_close",
        "volatilidade_5",
        "amplitude_pct",
    ]
    return df, atributos


def dividir_treino_teste(
    X: np.ndarray, y: np.ndarray, proporcao_treino: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.
    """
    limite = int(len(X) * proporcao_treino)
    if limite == 0 or limite == len(X):
        raise ValueError("Proporcao de treino invalida para o tamanho do conjunto.")
    X_treino, X_teste = X[:limite], X[limite:]
    y_treino, y_teste = y[:limite], y[limite:]
    return X_treino, y_treino, X_teste, y_teste


