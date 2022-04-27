# IMPORTANDO BIBLIOTECAS
import numpy as np
import random
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ESTABELECENDO PARÂMETROS
#precos = [1.99, 2.49, 2.99, 3.49, 3.99, 5.99]


precos = [5, 5, 5, 5, 5, 5]
demanda_observada = np.array([100, 200, 300, 400, 500, 600])


precos * demanda_observada







def resolve(result_teta):
    receita = precos * demanda_observada
    prop_receita = receita / np.sum(receita)

    alpha = prop_receita + (result_teta / np.sum(result_teta))

    return alpha / 2












# CRIANDO DEMANDA OBSERVADA
"""
demanda_observada = precos[::-1]
demanda_observada = np.array(demanda_observada) * 5
demanda_observada = [int(obs)*0.5 for obs in demanda_observada]
demanda_observada[-1] = 0.5
demanda_observada = np.array(demanda_observada)
"""


# prior distribution for each price - gamma(α, β)
# α = parâmetro de forma - Para valores de α muito altos, a distribuição gamma tende à Gaussiana
# β = parâmetro de escala - tem a função de ESTICAR OU ENCOLHER




def gamma(df):
    
    k = np.array([linha[2] for linha in df]) # forma
    theta = np.array([linha[3] for linha in df]) # escala

    return np.random.gamma(k, theta)



def funcao_objetivo(precos, demandas, estoque): 

    """
    sobre c ser negativo:
    Como as elasticidades-preço são negativas, o problema de otimização acima é maximizar
    uma função côncava sujeita a restrições convexas. Portanto, é uma otimização convexa
    problema que pode ser resolvido usando técnicas padrão Bertsekas & Scientific
    """

    receita = np.multiply(precos, demandas)
    
    #L = len(precos)
    #M = np.full([1, L], 1)
    #B = [[1]]
    #Df = [demandas]

    res = linprog(
        c = -np.array(receita).flatten(), # Os coeficientes da função objetivo linear a serem minimizados
        A_eq = np.array([[1] * len(precos)]), # A matriz de restrição de igualdade. Cada linha de A_eq especifica os coeficientes de uma restrição de igualdade linear em x
        b_eq = [[1]], # O vetor de restrição de igualdade. Cada elemento de A_eq @ x deve ser igual ao elemento correspondente de b_eq.
        A_ub = [demandas], # A matriz de restrições de desigualdade. Cada linha de A_ub especifica os coeficientes de uma restrição de desigualdade linear em x.
        b_ub = np.array([estoque]), # O vetor de restrição de desigualdade. Cada elemento representa um limite superior no valor correspondente de A_ub @ x.
        bounds = (0, None)) # Uma sequência de pares (min, max) para cada elemento em x, definindo os valores mínimo e máximo dessa variável de decisão

    resultado = np.array(res.x).reshape(1, len(precos)).flatten()
    return resultado





















T = 100
history = []

lista_preco_escolhido = []

teta = []
for p in precos:
    # PREÇO, DEMANDA, ALPHA, BETA, MÉDIA
    teta.append(np.array([p, None, 50.00, 1.00, 50.00]))

for t in range(0, T):             
    # MODELANDO A DEMANDA
    demandas = gamma(teta)

 
    #demandas = demandas + (demandas * ((demanda_observada * 0.2) / 100 ))
    #demandas = demandas + np.log(demanda_observada)

    # RECEITA
    price_probs = resolve(demandas)


    # seleção aleatória de um preço
    preco_index = random.choices(list(range(len(precos))), weights = price_probs, k = 1)[0]


    """
    lista_preco_escolhido.append(preco_index)
    preco_escolhido = precos[preco_index]
    
    # sell at the selected price and observe demand
    

    print('preço selecionado %.2f => demanda %.2f, receita %.2f' % (preco_escolhido, demanda_escolhida, demanda_escolhida*preco_escolhido))
    
    """


    # ATUALIZANDO TETA
    #teta[preco_index][1] = demanda_escolhida
    teta[preco_index][2] += 5
    teta[preco_index][3] += 1

    history.append(teta)


pd.Series(lista_preco_escolhido).value_counts()

pd.DataFrame(teta)

# DEMANDA
distribuicoes_alpha = [np.random.normal(valores[2], valores[3], 100) for valores in history[0]]

for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(precos[i]))
plt.legend()
plt.show()


