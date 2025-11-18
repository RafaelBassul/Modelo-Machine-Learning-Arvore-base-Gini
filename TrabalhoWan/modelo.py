"""
Módulo para predição e avaliação do modelo de árvore de decisão.
"""
import numpy as np
from typing import Iterable, List, Sequence

from arvore import No


def prever_amostra(no: No, x: Sequence[float]) -> str:
    """
    Faz predição para uma única amostra.
    
    Args:
        no: Nó raiz da árvore de decisão
        x: Vetor de features de uma amostra
        
    Returns:
        Classe predita
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
    
    Args:
        no: Nó raiz da árvore de decisão
        X: Matriz de features (n_amostras, n_features)
        
    Returns:
        Lista de classes preditas
    """
    return [prever_amostra(no, x) for x in X]


def acuracia(y_verdadeiro: Iterable[str], y_predito: Iterable[str]) -> float:
    """
    Calcula a acurácia do modelo.
    
    Args:
        y_verdadeiro: Classes verdadeiras
        y_predito: Classes preditas
        
    Returns:
        Acurácia (proporção de predições corretas)
    """
    pares = list(zip(y_verdadeiro, y_predito))
    corretos = sum(1 for real, pred in pares if real == pred)
    return corretos / len(pares) if pares else 0.0

