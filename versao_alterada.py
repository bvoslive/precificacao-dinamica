# IMPORTANDO BIBLIOTECAS
import numpy as np
import random
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


# ESTABELECENDO PARÂMETROS


df = pd.read_csv('commodities_agrupados.csv')

COD_COMMODITY = '55072'
teste_precos = df[df['COMMODITY'] == COD_COMMODITY]

teste_precos = teste_precos.drop(50640)
teste_precos = teste_precos.drop(50648)

precos = teste_precos['PRECO'].tolist()

demands_df = teste_precos['QUANTIDADE'].tolist()
demands_df = np.array(demands_df) / 80
demanda_observada = demands_df


#------------------------------

df = pd.read_csv('commodities_agrupados.csv')
df.sort_values('COMMODITY')



commodities = df['COMMODITY'].unique().tolist()


commodity_selecionada = commodities[6]

df_fragmentos = df[df['COMMODITY'] == commodity_selecionada]


precos = df_fragmentos['PRECO'].tolist()

precos = [5] * 6



quantidades = df_fragmentos['QUANTIDADE'].tolist()
demanda_observada = np.array([int(qnt) for qnt in quantidades])

demanda_observada = np.array([100, 200, 300, 400, 500, 600])


np.array(precos) * np.array(demanda_observada)





# Hidden (true) demand parameters - a linear demans function is assumed
demand_a = 50
demand_b = 21

# prior distribution for each price - gamma(α, β)
# α = parâmetro de forma - Para valores de α muito altos, a distribuição gamma tende à Gaussiana
# β = parâmetro de escala - tem a função de ESTICAR OU ENCOLHER

teta = []
for p in range(len(precos)):
    # PREÇO, DEMANDA, ALPHA, BETA, MÉDIA
    teta.append([precos[p], None, 30.00 + demands_df[p], 1.00, 30.00])


def gamma(df):
    
    k = np.array([linha[2] for linha in df]) # forma
    theta = 1/np.array([linha[3] for linha in df]) # escala

    return np.random.gamma(k, theta)



gamma(teta)




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
lista_teta = []
for t in range(0, T):              # simulation loop

    # MODELANDO A DEMANDA
    demandas = gamma(teta)

    # quando o estoque for baixo, ele vai tentar priorizar aqueles que possuem valores mais altos


    price_probs = funcao_objetivo(precos, demandas, np.mean(demandas))

    print('DEMANDA = ', demandas)
    print('PROB PREÇOS = ', price_probs)

    # seleção aleatória de um preço
    preco_index = random.choices(range(len(precos)), weights = price_probs, k = 1)[0]
    preco_escolhido = precos[preco_index]
    
    # sell at the selected price and observe demand
    demanda_avariar = demandas[preco_index]
    demanda_escolhida = np.random.poisson(demanda_avariar, 1)[0]

    #demanda_escolhida = demanda_observada[preco_index]

    print('preço selecionado %.2f => demanda %.2f, receita %.2f' % (preco_escolhido, demanda_escolhida, demanda_escolhida*preco_escolhido))

    # ATUALIZANDO TETA
    teta[preco_index][1] = demanda_escolhida
    teta[preco_index][2] = teta[preco_index][2] + (demanda_escolhida * 0.2)
    teta[preco_index][3] = teta[preco_index][3] + (demanda_escolhida * 0.05)
    teta[preco_index][4] = teta[preco_index][2] / teta[preco_index][3]

    lista_teta.append(teta)



pd.DataFrame(teta)




# DEMANDA
distribuicoes_alpha = [np.random.normal(valores[2] , valores[3], int(valores[1])) for valores in teta]

for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(precos[i]))
plt.legend()
plt.show()









# RECEITA
distribuicoes_alpha = [np.random.normal(valores[2] * valores[0], valores[3] * valores[0], 100) for valores in teta]


for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(precos[i]))
plt.legend()
plt.show()


