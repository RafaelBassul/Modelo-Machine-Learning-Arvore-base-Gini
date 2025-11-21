"""
Módulo para visualização da árvore de decisão.
"""
import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

from arvore import No, calcular_impureza_total

# Configura backend do matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")


def _atribuir_posicoes(no: No) -> Dict[No, Tuple[float, float]]:
    """Atribui posições (x, y) para cada nó da árvore para visualização.
    
    Usa um algoritmo que calcula a largura de cada subárvore e posiciona
    os nós de forma que os folhas não se sobreponham.
    """
    posicoes: Dict[No, Tuple[float, float]] = {}
    espacamento_horizontal = 4.0  # Espaçamento horizontal entre folhas
    espacamento_vertical = 2.0    # Espaçamento vertical entre níveis
    
    def calcular_posicao(node: No, profundidade: int, x_inicial: float) -> float:
        """Calcula a posição x do nó e retorna a próxima posição x disponível."""
        if node.classe is not None:
            # Nó folha: ocupa apenas uma posição
            posicoes[node] = (x_inicial, -profundidade * espacamento_vertical)
            return x_inicial + espacamento_horizontal
        
        # Nó interno: precisa posicionar filhos primeiro
        x_atual = x_inicial
        x_esquerda = None
        x_direita = None
        
        # Posiciona subárvore esquerda
        if node.esquerda is not None:
            x_atual = calcular_posicao(node.esquerda, profundidade + 1, x_atual)
            x_esquerda = posicoes[node.esquerda][0]
        
        # Posiciona subárvore direita
        if node.direita is not None:
            x_atual = calcular_posicao(node.direita, profundidade + 1, x_atual)
            x_direita = posicoes[node.direita][0]
        
        # Centraliza o nó pai entre seus filhos
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
    
    Args:
        no: Nó raiz da árvore de decisão
        figsize: Tamanho da figura (largura, altura)
        caminho_saida: Caminho para salvar o gráfico (opcional)
        nomes_atributos: Lista com nomes dos atributos (opcional, usa X1, X2... se não fornecido)
        acuracia: Acurácia no conjunto de teste para exibir no gráfico (opcional)
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
        
        # Formata contagem de amostras por classe
        n_alta = node.n_alta if node.n_alta is not None else 0
        n_baixa = node.n_baixa if node.n_baixa is not None else 0
        amostras_str = f"classe: alta = {n_alta} // baixa = {n_baixa}"
        
        if node.classe is None:
            # Nó interno: mostra nome do atributo, Gini e contagem de amostras (sem impureza média dentro)
            if nomes_atributos is not None and node.atributo < len(nomes_atributos):
                nome_atributo = nomes_atributos[node.atributo]
            else:
                nome_atributo = f"X{node.atributo + 1}"
            
            label = f"{nome_atributo} <= {node.corte:.3f}\n{impureza_str}\n{amostras_str}"
            facecolor = "#1f77b4"
        else:
            # Folha: mostra classe, Gini e contagem de amostras
            label = f"Classe {node.classe}\n{impureza_str}\n{amostras_str}"
            facecolor = "#ff7f0e"

        bbox = dict(boxstyle="round,pad=0.5", fc=facecolor, ec="black", alpha=0.85, lw=1.5)
        # Ajusta tamanho da fonte
        if node.classe is None:
            fontsize = 7  # 3 linhas: atributo, Gini, amostras
        else:
            fontsize = 7.5  # 3 linhas: classe, Gini, amostras
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="white", bbox=bbox, weight="bold")
        
        # Adiciona impureza média abaixo do nó (apenas para nós internos que têm essa informação)
        if node.classe is None and node.impureza_media_ponderada is not None:
            imp_media_str = f"Imp.Med: {node.impureza_media_ponderada:.3f}"
            # Posiciona o texto abaixo do nó (ajusta o offset vertical)
            offset_y = -0.4  # Espaçamento abaixo do nó
            ax.text(x, y + offset_y, imp_media_str, ha="center", va="top", 
                   fontsize=7, color="black", weight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", alpha=0.8, lw=1))

    desenhar(no)
    
    # Ajusta os limites do eixo para garantir que todos os nós sejam visíveis
    # Inclui espaço extra na parte inferior para a impureza média
    todas_posicoes = list(posicoes.values())
    if todas_posicoes:
        xs = [p[0] for p in todas_posicoes]
        ys = [p[1] for p in todas_posicoes]
        margem_x = (max(xs) - min(xs)) * 0.1 if max(xs) != min(xs) else 2.0
        margem_y = (max(ys) - min(ys)) * 0.1 if max(ys) != min(ys) else 1.0
        # Adiciona espaço extra na parte inferior para a impureza média
        margem_y_inferior = 0.6
        ax.set_xlim(min(xs) - margem_x, max(xs) + margem_x)
        ax.set_ylim(min(ys) - margem_y - margem_y_inferior, max(ys) + margem_y)
    
    ax.set_axis_off()
    
    # Calcula a impureza total da árvore
    impureza_total = calcular_impureza_total(no)
    
    # Adiciona acurácia e impureza total no canto superior direito
    if acuracia is not None:
        # Usa coordenadas da figura (0-1) para posicionar no canto superior direito
        texto_acuracia = f"Acurácia no conjunto de teste: {acuracia:.3f}"
        texto_impureza = f"Impureza total da árvore: {impureza_total:.3f}"
        
        # Posiciona a acurácia
        fig.text(0.98, 0.98, texto_acuracia, 
                ha="right", va="top", 
                fontsize=12, weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9, lw=1.5),
                transform=fig.transFigure)
        
        # Posiciona a impureza total logo abaixo da acurácia
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

