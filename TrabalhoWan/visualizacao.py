"""
Módulo para visualização da árvore de decisão.
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

from arvore import No, calcular_impureza_total

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")


def _atribuir_posicoes(no: No) -> Dict[No, Tuple[float, float]]:
    """Atribui posições (x, y) para cada nó da árvore para visualização.
    
    Usa um algoritmo que calcula a largura de cada subárvore e posiciona
    os nós de forma que os folhas não se sobreponham.
    """
    posicoes: Dict[No, Tuple[float, float]] = {}
    espacamento_horizontal = 4.0
    espacamento_vertical = 2.0
    
    def calcular_posicao(node: No, profundidade: int, x_inicial: float) -> float:
        """Calcula a posição x do nó e retorna a próxima posição x disponível."""
        if node.classe is not None:
            posicoes[node] = (x_inicial, -profundidade * espacamento_vertical)
            return x_inicial + espacamento_horizontal
        
        x_atual = x_inicial
        x_esquerda = None
        x_direita = None
        
        if node.esquerda is not None:
            x_atual = calcular_posicao(node.esquerda, profundidade + 1, x_atual)
            x_esquerda = posicoes[node.esquerda][0]
        
        if node.direita is not None:
            x_atual = calcular_posicao(node.direita, profundidade + 1, x_atual)
            x_direita = posicoes[node.direita][0]
        
        if x_esquerda is not None and x_direita is not None:
            x_medio = (x_esquerda + x_direita) / 2.0
        elif x_esquerda is not None:
            x_medio = x_esquerda
        elif x_direita is not None:
            x_medio = x_direita
        else:
            x_medio = x_inicial
        
        posicoes[node] = (x_medio, -profundidade * espacamento_vertical)
        
        return x_atual
    
    calcular_posicao(no, 0, 0.0)
    return posicoes


def plotar_arvore(no: No, figsize: Tuple[int, int] = (30, 15), caminho_saida: Optional[Path] = None, nomes_atributos: Optional[list] = None, acuracia: Optional[float] = None) -> None:
    """
    Plota a árvore de decisão em um gráfico.
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
        
        n_alta = node.n_alta if node.n_alta is not None else 0
        n_baixa = node.n_baixa if node.n_baixa is not None else 0
        amostras_str = f"classe: alta = {n_alta} // baixa = {n_baixa}"
        
        if node.classe is None:
            if nomes_atributos is not None and node.atributo < len(nomes_atributos):
                nome_atributo = nomes_atributos[node.atributo]
            else:
                nome_atributo = f"X{node.atributo + 1}"
            
            label = f"{nome_atributo} <= {node.corte:.3f}\n{impureza_str}\n{amostras_str}"
            facecolor = "#1f77b4"
        else:
            label = f"Classe {node.classe}\n{impureza_str}\n{amostras_str}"
            facecolor = "#ff7f0e"

        bbox = dict(boxstyle="round,pad=0.5", fc=facecolor, ec="black", alpha=0.85, lw=1.5)
        if node.classe is None:
            fontsize = 7
        else:
            fontsize = 7.5
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="white", bbox=bbox, weight="bold")
        
        if node.classe is None and node.impureza_media_ponderada is not None:
            imp_media_str = f"Imp.Med: {node.impureza_media_ponderada:.3f}"
            offset_y = -0.4
            ax.text(x, y + offset_y, imp_media_str, ha="center", va="top", 
                   fontsize=7, color="black", weight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", alpha=0.8, lw=1))

    desenhar(no)
    
    todas_posicoes = list(posicoes.values())
    if todas_posicoes:
        xs = [p[0] for p in todas_posicoes]
        ys = [p[1] for p in todas_posicoes]
        margem_x = (max(xs) - min(xs)) * 0.1 if max(xs) != min(xs) else 2.0
        margem_y = (max(ys) - min(ys)) * 0.1 if max(ys) != min(ys) else 1.0
        margem_y_inferior = 0.6
        ax.set_xlim(min(xs) - margem_x, max(xs) + margem_x)
        ax.set_ylim(min(ys) - margem_y - margem_y_inferior, max(ys) + margem_y)
    
    ax.set_axis_off()
    
    impureza_total = calcular_impureza_total(no)
    
    if acuracia is not None:
        texto_acuracia = f"Acurácia no conjunto de teste: {acuracia:.3f}"
        texto_impureza = f"Impureza total da árvore: {impureza_total:.3f}"
        
        fig.text(0.98, 0.98, texto_acuracia, 
                ha="right", va="top", 
                fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9, lw=1.5),
                transform=fig.transFigure)
        
        fig.text(0.98, 0.94, texto_impureza, 
                ha="right", va="top", 
                fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", alpha=0.9, lw=1.5),
                transform=fig.transFigure)
    
    if caminho_saida is not None:
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        print(f"Grafico salvo em: {caminho_saida.resolve()}")
    backend = plt.get_backend().lower()
    if backend.startswith("agg"):
        plt.close(fig)
    else:
        plt.show()

