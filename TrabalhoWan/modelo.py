"""
Módulo para predição e avaliação do modelo de árvore de decisão.
"""
import numpy as np
from typing import Iterable, List, Sequence

from arvore import No


def prever_amostra(no: No, x: Sequence[float]) -> str:
    """
    Faz predição para uma única amostra.
    """
    atual = no
    while atual.classe is None:
        if x[atual.atributo] <= atual.corte:
            atual = atual.esquerda
        else:
            atual = atual.direita
    return atual.classe


def prever(no: No, X: np.ndarray) -> List[str]:
    """
    Faz predições para múltiplas amostras.
    """
    return [prever_amostra(no, x) for x in X]


def acuracia(y_verdadeiro: Iterable[str], y_predito: Iterable[str]) -> float:
    """
    Calcula a acurácia do modelo.
    """
    pares = list(zip(y_verdadeiro, y_predito))
    corretos = sum(1 for real, pred in pares if real == pred)
    return corretos / len(pares) if pares else 0.0


