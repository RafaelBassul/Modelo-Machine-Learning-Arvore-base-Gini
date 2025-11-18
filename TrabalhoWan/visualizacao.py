"""
Módulo para visualização da árvore de decisão.
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

from arvore import No

# Configura backend do matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")


def _atribuir_posicoes(no: No) -> Dict[No, Tuple[float, float]]:
    """Atribui posições (x, y) para cada nó da árvore para visualização."""
    posicoes: Dict[No, Tuple[float, float]] = {}
    x_corrente = 0.0
    espacamento_horizontal = 2.5  # Espaçamento horizontal entre nós
    espacamento_vertical = 1.5     # Fator para espaçamento vertical

    def helper(node: No, profundidade: int) -> None:
        nonlocal x_corrente
        if node.esquerda is not None:
            helper(node.esquerda, profundidade + 1)
        posicoes[node] = (x_corrente, -profundidade * espacamento_vertical)
        x_corrente += espacamento_horizontal
        if node.direita is not None:
            helper(node.direita, profundidade + 1)

    helper(no, 0)
    return posicoes


def plotar_arvore(no: No, figsize: Tuple[int, int] = (20, 10), caminho_saida: Optional[Path] = None, nomes_atributos: Optional[list] = None) -> None:
    """
    Plota a árvore de decisão em um gráfico.
    
    Args:
        no: Nó raiz da árvore de decisão
        figsize: Tamanho da figura (largura, altura)
        caminho_saida: Caminho para salvar o gráfico (opcional)
        nomes_atributos: Lista com nomes dos atributos (opcional, usa X1, X2... se não fornecido)
    """
    posicoes = _atribuir_posicoes(no)
    fig, ax = plt.subplots(figsize=figsize)

    def desenhar(node: No) -> None:
        x, y = posicoes[node]
        if node.esquerda is not None:
            xe, ye = posicoes[node.esquerda]
            ax.plot([x, xe], [y, ye], color="black", linewidth=1)
            desenhar(node.esquerda)
        if node.direita is not None:
            xd, yd = posicoes[node.direita]
            ax.plot([x, xd], [y, yd], color="black", linewidth=1)
            desenhar(node.direita)

        impureza_str = f"Gini: {node.impureza:.3f}" if node.impureza is not None else "Gini: N/A"
        
        if node.classe is None:
            # Nó interno: mostra nome do atributo, Gini e impureza média ponderada
            if nomes_atributos is not None and node.atributo < len(nomes_atributos):
                nome_atributo = nomes_atributos[node.atributo]
            else:
                nome_atributo = f"X{node.atributo + 1}"
            
            if node.impureza_media_ponderada is not None:
                imp_media_str = f"Imp.Med: {node.impureza_media_ponderada:.3f}"
                label = f"{nome_atributo} <= {node.corte:.3f}\n{impureza_str}\n{imp_media_str}"
            else:
                label = f"{nome_atributo} <= {node.corte:.3f}\n{impureza_str}"
            facecolor = "#1f77b4"
        else:
            # Folha: mostra apenas Gini (não tem divisão, então não tem impureza média)
            label = f"Classe {node.classe}\n{impureza_str}"
            facecolor = "#ff7f0e"

        bbox = dict(boxstyle="round,pad=0.5", fc=facecolor, ec="black", alpha=0.85, lw=1.5)
        # Ajusta tamanho da fonte: menor se tiver 3 linhas (nó interno com impureza média)
        fontsize = 7 if node.classe is None and node.impureza_media_ponderada is not None else 8
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="white", bbox=bbox, weight="bold")

    desenhar(no)
    ax.set_axis_off()
    fig.tight_layout()
    if caminho_saida is not None:
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        print(f"Grafico salvo em: {caminho_saida.resolve()}")
    backend = plt.get_backend().lower()
    if backend.startswith("agg"):
        plt.close(fig)
    else:
        plt.show()

