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

prices = [1.99, 2.49, 2.99, 3.49, 3.99, 5.99]
demanda_observada = prices[::-1]
demanda_observada = np.array(demanda_observada) * 5
demanda_observada = [int(obs)*0.5 for obs in demanda_observada]
demanda_observada[-1] = 0.5
demanda_observada = np.array(demanda_observada)


# Hidden (true) demand parameters - a linear demans function is assumed
demand_a = 50
demand_b = 21

# prior distribution for each price - gamma(α, β)

# α = parâmetro de forma - Para valores de α muito altos, a distribuição gamma tende à Gaussiana
# β = parâmetro de escala - tem a função de ESTICAR OU ENCOLHER

teta = []
for p in prices:
    # PREÇO, DEMANDA, ALPHA, BETA, MÉDIA
    teta.append(np.array([p, None, 30.00, 1.00, 30.00]))


def gamma(df):
    
    shape = np.array([linha[2] for linha in df])
    scale = np.array([linha[3] for linha in df])

    return np.random.gamma(shape, scale)

def optimal_price_probabilities(prices, demands, inventory): 

    """
    sobre c ser negativo:
    Como as elasticidades-preço são negativas, o problema de otimização acima é maximizar
    uma função côncava sujeita a restrições convexas. Portanto, é uma otimização convexa
    problema que pode ser resolvido usando técnicas padrão Bertsekas & Scientific
    """

    revenues = np.multiply(prices, demands)
    
    L = len(prices)
    M = np.full([1, L], 1)
    B = [[1]]
    Df = [demands]

    res = linprog(-np.array(revenues).flatten(), 
                  A_eq=M, 
                  b_eq=B, 
                  A_ub=Df, 
                  b_ub=np.array([inventory]), 
                  bounds=(0, None))

    price_prob = np.array(res.x).reshape(1, L).flatten()
    return price_prob


T = 100
history = []
for t in range(0, T):              # simulation loop

    # MODELANDO A DEMANDA
    demands = gamma(teta)
    demands = demands - (demands * 0.2) + demanda_observada 

    print(tabulate(np.array(teta), tablefmt="fancy_grid"))
    #print("demands = ", np.array(demands))

    # RECEITA
    revenues = np.multiply(prices, demands)
    
    #valor das variáveis de decisão
    price_probs = optimal_price_probabilities(prices, demands, 70)

    print('DEMANDA = ', demands)
    print('RECEITA = ', revenues)
    print('PROB PREÇOS = ', price_probs)

    # seleção aleatória de um preço
    price_index_t = random.choices(range(len(prices)), weights = price_probs, k = 1)[0]
    preco_t = prices[price_index_t]
    
    # venda sobre o preço observado e a demanda observada
    #demand = demand_a - demand_b * preco_t

    # sell at the selected price and observe demand
    demand = demanda_observada[price_index_t] + (0.2 * demanda_observada[price_index_t])
    demand_t = np.random.poisson(demand, 1)[0]


    print('preço selecionado %.2f => demanda %.2f, receita %.2f' % (preco_t, demand_t, demand_t*preco_t))

    # ATUALIZANDO TETA
    teta[price_index_t][1] = demand_t
    teta[price_index_t][2] = teta[price_index_t][2] + demand_t
    teta[price_index_t][3] += (demand_t * 0.1)
    teta[price_index_t][4] = teta[price_index_t][2] / teta[price_index_t][3]

    history.append(teta)





"""
teta['demanda'][price_index_t] = demand_t
teta['alpha'][price_index_t] = teta['alpha'][price_index_t] + demand_t
teta['beta'][price_index_t] += 1
teta['mean'][price_index_t] = teta['alpha'][price_index_t] / teta['beta'][price_index_t]
"""





# IMPRIMINDO HISTORIA

df_demanda = history[-1]['demanda']
df_alpha = history[-1]['alpha']
df_beta = history[-1]['beta']





# DEMANDA
distribuicoes_alpha = [np.random.normal(valores[2], valores[3], 100) for valores in teta]









for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(prices[i]))
plt.legend()
plt.show()







# RECEITA
distribuicoes_receita = [np.random.normal(valores[0] * valores[2], valores[3], int(valores[1])) for valores in teta.values]

for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(teta['price'][i]))
plt.legend()
plt.show()




x = np.linspace(0, 200, 200)


for i, params in enumerate(history[43][3]):
    y = stats.gamma.pdf(x, a=params['alpha'], scale=1.0/params['beta']) 
    plt.plot(x, y, "-", label='price %.2f' % params['price'])

len(y)


x = np.linspace(30, 110, 200) 
for i, params in enumerate(history[t][3]):
    y = stats.gamma.pdf(x, a=params['alpha']*params['price'], scale=1.0/params['beta']) 
    plt.plot(x, y, "-", label='price %.2f' % params['price']) 



plt.legend(loc='upper left')
plt.ylim([0, 0.5])
plt.grid(True)
plt.show()



