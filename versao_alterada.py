# IMPORTANDO BIBLIOTECAS
import numpy as np
import random
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns

# ESTABELECENDO PARÂMETROS

prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]

# Hidden (true) demand parameters - a linear demans function is assumed
demand_a = 50
demand_b = 2



# prior distribution for each price - gamma(α, β)

# α = parâmetro de forma - Para valores de α muito altos, a distribuição gamma tende à Gaussiana
# β = parâmetro de escala - tem a função de ESTICAR OU ENCOLHER


colunas = ['price', 'demanda', 'alpha', 'beta', 'mean']
teta = pd.DataFrame(columns = colunas)
for p in prices:
    teta = teta.append(pd.Series([p, None, 30.00, 1.00, 30.00], index=colunas), ignore_index=True)


# PREDIÇÃO DE DEMANDA
def gamma(df):
    
    shape = df['alpha']
    scale = 1/df['beta']
    result_dist_gama = np.random.gamma(shape, scale)
    return result_dist_gama




demanda_observada = 50
estoque = 40

T = 60
history = []
for t in range(0, T):              # simulation loop

    # MODELANDO A DEMANDA
    demands = gamma(teta)

    print(tabulate(np.array(teta), tablefmt="fancy_grid"))
    print("demands = ", np.array(demands))

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

    #np.sum(price_probs)

    print(demands)
    print(revenues)
    print(price_probs)

    # seleção aleatória de um preço
    #price_index_t = np.random.choice(len(prices), 1, p=price_probs)[0]

    price_index_t = random.choices(range(len(prices)), weights = price_probs, k = 1)[0]
    preco_t = prices[price_index_t]
    
    # venda sobre o preço observado e a demanda observada
    #demand = demand_a - demand_b * preco_t

    demanda_t = np.random.poisson(demanda_observada, 1)[0]

    print('preço selecionado %.2f => demanda %.2f, receita %.2f' % (preco_t, demanda_t, demanda_t*preco_t))

    """
    teta_filtro = []
    for v in teta:
        teta_filtro.append(v.copy())
    history.append([preco_t, demanda_t, demanda_t*preco_t, teta_filtro])
    """

    # ATUALIZANDO TETA
    
    teta['demanda'][price_index_t] = demanda_t
    teta['alpha'][price_index_t] += demanda_t
    teta['beta'][price_index_t] += 1
    teta['mean'][price_index_t] = teta['alpha'][price_index_t] / teta['beta'][price_index_t]



# IMPRIMINDO HISTORIA

df_demanda = teta['demanda']
df_alpha = teta['alpha']
df_beta = teta['beta']



# DEMANDA
distribuicoes_alpha = [np.random.normal(valores[2], valores[3], int(valores[1])) for valores in teta.values]


for i, distribuicao in enumerate(distribuicoes_alpha):
    sns.distplot(distribuicao, label=str(teta['price'][i]))
plt.legend()
plt.show()


# RECEITA
distribuicoes_receita = [np.random.normal(valores[0] * valores[2], valores[3], int(valores[1])) for valores in teta.values]

for i, distribuicao in enumerate(distribuicoes_receita):
    sns.distplot(distribuicao, label=str(teta['price'][i]))
plt.legend()
plt.show()









distribuicoes[0]




x = np.linspace(0, 200, 200)



history[43][3]

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






