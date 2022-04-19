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
precos = [1.99, 2.49, 2.99, 3.49, 3.99, 5.99]


precos = [19.9, 24.9, 29.9, 34.9, 39.9, 59.9]

# CRIANDO DEMANDA OBSERVADA
demanda_observada = precos[::-1]
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
for p in precos:
    # PREÇO, DEMANDA, ALPHA, BETA, MÉDIA
    teta.append(np.array([p, None, 30.00, 1.00, 30.00]))


def gamma(df):
    
    k = np.array([linha[2] for linha in df]) # shape
    theta = np.array([linha[3] for linha in df]) # scale

    return np.random.gamma(k, theta)



def optimal_price_probabilities(precos, demandas, estoque): 

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
        bounds = (0, None)) # A sequence of (min, max) pairs for each element in x, defining the minimum and maximum values of that decision variable

    price_prob = np.array(res.x).reshape(1, len(precos)).flatten()
    return price_prob



T = 150
history = []
for t in range(0, T):              # simulation loop

    # MODELANDO A DEMANDA
    demandas = gamma(teta)
    demandas = demandas - (demandas * 0.2) + demanda_observada 

    print(tabulate(np.array(teta), tablefmt="fancy_grid"))
    #print("demands = ", np.array(demands))

    # RECEITA
    revenues = np.multiply(precos, demandas)
    
    #valor das variáveis de decisão
    price_probs = optimal_price_probabilities(precos, demandas, 70)

    print('DEMANDA = ', demandas)
    print('RECEITA = ', revenues)
    print('PROB PREÇOS = ', price_probs)

    # seleção aleatória de um preço
    price_index_t = random.choices(range(len(precos)), weights = price_probs, k = 1)[0]
    preco_t = precos[price_index_t]
    
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







# DEMANDA
distribuicoes_alpha = [np.random.normal(valores[2] , valores[3], 100) for valores in teta]

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



# RECEITA
distribuicoes_receita = [np.random.normal(valores[0] * valores[2], valores[3], int(valores[1])) for valores in teta.values]

for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(teta['price'][i]))
plt.legend()
plt.show()


#---------------------------------------------------

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



