"""
Módulo contendo a implementação da árvore de decisão CART com índice de Gini.
"""
import numpy as np
from collections import Counter
from typing import Optional, Tuple


class No:
    """Estrutura do nó da árvore de decisão."""
    def __init__(self, atributo=None, corte=None, esquerda=None, direita=None, classe=None, impureza=None, impureza_media_ponderada=None, n_alta=None, n_baixa=None):
        self.atributo = atributo  # indice do atributo (j)
        self.corte = corte        # valor de corte (s)
        self.esquerda = esquerda  # subarvore: x_j <= s
        self.direita = direita    # subarvore: x_j > s
        self.classe = classe      # classe da folha (se for no terminal)
        self.impureza = impureza  # impureza de Gini do nó (antes da divisão)
        self.impureza_media_ponderada = impureza_media_ponderada  # impureza média após a melhor divisão
        self.n_alta = n_alta      # número de amostras da classe "Alta" no nó
        self.n_baixa = n_baixa    # número de amostras da classe "Baixa" no nó


def gini(y):
    """Calcula o índice de impureza de Gini."""
    if len(y) == 0:
        return 0.0
    contagem = Counter(y)
    proporcoes = [count / len(y) for count in contagem.values()]
    return 1.0 - sum(p ** 2 for p in proporcoes)


def classe_majoritaria(y):
    """Retorna a classe majoritária."""
    return Counter(y).most_common(1)[0][0]


def criterio_parada(dados, profundidade, max_profundidade):
    """Verifica se deve parar a construção da árvore."""
    X, y = dados
    # No puro: todas as classes iguais
    if len(set(y)) <= 1:
        return True
    # Profundidade maxima
    if max_profundidade is not None and profundidade >= max_profundidade:
        return True
    # Poucas amostras (opcional)
    if len(y) < 2:
        return True
    return False


def dividir_dados(dados, j, s):
    """Divide os dados baseado no atributo j e valor de corte s."""
    X, y = dados
    mask = X[:, j] <= s
    dados_esq = (X[mask], y[mask])
    dados_dir = (X[~mask], y[~mask])
    return dados_esq, dados_dir


def impureza_media_ponderada(dados_esq, dados_dir):
    """Calcula a impureza média ponderada após uma divisão."""
    X_esq, y_esq = dados_esq
    X_dir, y_dir = dados_dir
    n = len(y_esq) + len(y_dir)
    if n == 0:
        return 0.0
    i_esq = gini(y_esq) if len(y_esq) > 0 else 0.0
    i_dir = gini(y_dir) if len(y_dir) > 0 else 0.0
    return (len(y_esq) / n) * i_esq + (len(y_dir) / n) * i_dir


def melhor_divisao(dados):
    """Encontra a melhor divisão (atributo j*, valor s*) que minimiza a impureza."""
    X, y = dados
    n_amostras, n_atributos = X.shape
    melhor_j, melhor_s = None, None
    melhor_impureza = float('inf')
    melhor_no_puro_tamanho = -1
    melhor_s_maior = -1

    for j in range(n_atributos):
        valores_unicos = np.unique(X[:, j])
        for s in valores_unicos:
            dados_esq, dados_dir = dividir_dados(dados, j, s)
            y_esq, y_dir = dados_esq[1], dados_dir[1]
            if len(y_esq) == 0 or len(y_dir) == 0:
                continue
            
            imp = impureza_media_ponderada(dados_esq, dados_dir)
            puro_esq = len(set(y_esq)) == 1 if len(y_esq) > 0 else False
            puro_dir = len(set(y_dir)) == 1 if len(y_dir) > 0 else False
            tamanho_puro = len(y_esq) if puro_esq else (len(y_dir) if puro_dir else 0)

            melhorar = False
            if imp < melhor_impureza:
                melhorar = True
            elif imp == melhor_impureza:
                if tamanho_puro > melhor_no_puro_tamanho:
                    melhorar = True
                elif tamanho_puro == melhor_no_puro_tamanho:
                    if s > melhor_s_maior:
                        melhorar = True
            
            if melhorar:
                melhor_impureza = imp
                melhor_j, melhor_s = j, s
                melhor_no_puro_tamanho = tamanho_puro
                melhor_s_maior = s

    if melhor_j is None:
        raise ValueError("Nenhuma divisao valida encontrada.")
    return melhor_j, melhor_s, melhor_impureza


def construir_arvore(dados, profundidade=0, max_profundidade=None):
    """Constrói a árvore de decisão recursivamente usando o algoritmo CART."""
    X, y = dados
    impureza_atual = gini(y)
    
    # Conta amostras por classe
    contagem = Counter(y)
    n_alta = contagem.get("Alta", 0)
    n_baixa = contagem.get("Baixa", 0)

    if criterio_parada(dados, profundidade, max_profundidade):
        classe_folha = classe_majoritaria(y)
        return No(classe=classe_folha, impureza=impureza_atual, impureza_media_ponderada=None, n_alta=n_alta, n_baixa=n_baixa)

    # Escolhe a melhor divisao (atributo j, valor s) e retorna também a impureza média
    j, s, imp_media = melhor_divisao(dados)
    dados_esq, dados_dir = dividir_dados(dados, j, s)

    # Cria o no atual e constroi as subarvores recursivamente
    no = No(atributo=j, corte=s, impureza=impureza_atual, impureza_media_ponderada=imp_media, n_alta=n_alta, n_baixa=n_baixa)
    no.esquerda = construir_arvore(dados_esq, profundidade + 1, max_profundidade)
    no.direita = construir_arvore(dados_dir, profundidade + 1, max_profundidade)

    return no


def calcular_impureza_total(no: No) -> float:
    """
    Calcula a impureza total da árvore (erro médio ponderado de classificação).
    
    A impureza total é calculada recursivamente a partir das folhas,
    usando a soma ponderada das impurezas de todas as folhas.
    Alternativamente, pode ser obtida recursivamente usando as impurezas médias
    ponderadas dos nós internos (raízes de subárvores).
    
    Args:
        no: Nó raiz da árvore de decisão
        
    Returns:
        Impureza total da árvore
    """
    def coletar_folhas(node: No) -> list:
        """Coleta todas as folhas da árvore."""
        folhas = []
        if node.classe is not None:
            folhas.append(node)
        else:
            if node.esquerda is not None:
                folhas.extend(coletar_folhas(node.esquerda))
            if node.direita is not None:
                folhas.extend(coletar_folhas(node.direita))
        return folhas
    
    # Calcula usando as folhas (método mais direto)
    folhas = coletar_folhas(no)
    n_total = sum((folha.n_alta or 0) + (folha.n_baixa or 0) for folha in folhas)
    
    if n_total == 0:
        return 0.0
    
    # Soma ponderada das impurezas das folhas
    impureza_total = 0.0
    for folha in folhas:
        n_folha = (folha.n_alta or 0) + (folha.n_baixa or 0)
        if n_folha > 0:
            peso = n_folha / n_total
            impureza_folha = folha.impureza if folha.impureza is not None else 0.0
            impureza_total += peso * impureza_folha
    
    return impureza_total


def imprimir_arvore(no, indent=""):
    """Imprime a árvore de decisão em formato textual (visualização didática)."""
    if no.classe is not None:
        print(f"{indent}--> Prediz: Classe {no.classe}")
        return

    print(f"{indent}[X_{no.atributo + 1} <= {no.corte}]")
    print(f"{indent}|- Sim ", end="")
    imprimir_arvore(no.esquerda, indent + "|  ")
    print(f"{indent}|_ Nao ", end="")
    imprimir_arvore(no.direita, indent + "   ")

