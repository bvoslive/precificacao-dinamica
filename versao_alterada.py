# IMPORTANDO BIBLIOTECAS
import numpy as np
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# ESTABELECENDO PARÂMETROS

prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]

# Hidden (true) demand parameters - a linear demans function is assumed
demand_a = 50
demand_b = 7

# prior distribution for each price - gamma(α, β)
teta = []
for p in prices:
    teta.append({'price': p, 'alpha': 30.00, 'beta': 1.00, 'mean': 30.00})


# PREDIÇÃO DE DEMANDA
def gamma(alpha, beta):
    shape = alpha
    scale = 1/beta
    return np.random.gamma(shape, scale)

T = 50
history = []
for t in range(0, T):              # simulation loop


    # MODELANDO A DEMANDA
    demands = list(map(lambda v: gamma(v['alpha'], v['beta']), teta))
    print(tabulate(np.array(teta), tablefmt="fancy_grid"))
    print("demands = ", np.array(demands))
    

    estoque = 60

    # RECEITA
    revenues = np.multiply(prices, demands)
    
    L = len(prices)
    M = np.full([1, L], 1)
    B = [[1]]
    Df = [demands]


    """
    sobre c ser negativo:
    Como as elasticidades-preço são negativas, o problema de otimização acima é maximizar
    uma função côncava sujeita a restrições convexas. Portanto, é uma otimização convexa
    problema que pode ser resolvido usando técnicas padrão Bertsekas & Scientific
    """

    # OTIMIZADOR
    res = linprog(

        # coeficientes da função objetivo
        c = -np.array(revenues).flatten(),

        # especifificando restrição de igualdade
        A_eq=M,

        # Cada elemento de A_eq deve ser igual ao elemento correspondente de b_eq
        b_eq=B,

        # demandas
        A_ub=Df,

        # estoque
        b_ub=np.array([estoque]),

        # sequência de pares (min, max) para cada elemento em x
        bounds=(0, None))


    #valor das variáveis de decisão
    price_probs = np.array(res.x).reshape(1, L).flatten()


    print(demands)
    print(revenues)
    print(price_probs)




    # seleção aleatória de um preço
    price_index_t = np.random.choice(len(prices), 1, p=price_probs)[0]
    price_t = prices[price_index_t]
    
    # sell at the selected price and observe demand
    demand = demand_a - demand_b * price_t
    demand_t = np.random.poisson(demand, 1)[0]

    print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t, demand_t, demand_t*price_t))
    
    theta_trace = []
    for v in teta:
        theta_trace.append(v.copy())
    history.append([price_t, demand_t, demand_t*price_t, theta_trace])

    # update model parameters
    v = teta[price_index_t]
    v['alpha'] = v['alpha'] + demand_t
    v['beta'] = v['beta'] + 1
    v['mean'] = v['alpha'] / v['beta']
    
    print("")








# -----> INSIGHTS <-----

# GERAL
prices = [h[0] for h in history]
demands = [h[1] for h in history]



# DEMANDA

x = np.linspace(0, 60, 200) 

t = 48


plt.xlabel('Demand')
plt.ylabel('Demand PDF')
plt.title('Demand PDFs for different prices')
x = np.linspace(0, 60, 200)


for i, params in enumerate(history[0][3]):
    y = stats.gamma.pdf(x, a=params['alpha'], scale=1.0/params['beta']) 
    plt.plot(x, y, "-", label='price %.2f' % params['price'])
plt.show()



params['alpha']


params['beta']


# FAZER AQUI O BOXPLOT

history[48][3][5]



import scipy
scipy.stats.gamma.pdf(4, 3)



for i, params in enumerate(history[t][3]):
    print(i)
    print(params)






plt.legend(loc='upper left')
plt.ylim([0, 0.5])
plt.grid(True)
plt.show()



history[48][3]


import seaborn


demand_a = 57
demand_b = 7

price = 4.99

def sample_demand(price):
    demand = demand_a - demand_b * price
    return np.random.poisson(demand, 100)


distribuicao = sample_demand(8)


sns.distplot(distribuicao)
plt.show()



sample_demand(4.99)




import pandas as pd

datas = pd.date_range('2020-03-01', '2022-01-01')

len_numeros_aleatorios = len(datas)

num_aleatorios = np.random.rand(len_numeros_aleatorios)
num_aleatorios = np.cumsum(num_aleatorios)

plt.plot(datas, num_aleatorios)
plt.show()