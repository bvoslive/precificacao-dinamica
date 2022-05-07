
import numpy as np
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np


precos = []



df = pd.read_csv('commodities_agrupados.csv')

COD_COMMODITY = '55072'
teste_precos = df[df['COMMODITY'] == COD_COMMODITY]

teste_precos = teste_precos.drop(50640)
teste_precos = teste_precos.drop(50648)

prices = teste_precos['PRECO'].tolist()

demands_df = teste_precos['QUANTIDADE'].tolist()
demands_df = np.array(demands_df) / 80
demands = demands_df



# VISUALIZANDO PREÇOS E DEMANDA
plt.scatter(prices, demands_df)
plt.show()


# PREÇO MULTIPLICANDO DEMANDA
pd.DataFrame(np.array(prices) * np.array(demands_df)).sort_values(0)






def tabprint(msg, A):
    print(msg)
    print(tabulate(A, tablefmt="fancy_grid"))


def optimal_price_probabilities(prices, demands, inventory):   
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



# Optimization procedure test
prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]
demands = list(map(lambda p: 50 - 7*p, prices))






revenues = np.multiply(prices, demands)
print(demands)
print(revenues)
print(optimal_price_probabilities(prices, demands, 232))



# -----> ETAPA 2 <-----

#prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]

# Hidden (true) demand parameters - a linear demans function is assumed
#demand_a = 50
#demand_b = 7

# prior distribution for each price - gamma(α, β)
teta = []
for p in range(len(prices)):
    teta.append({'price': prices[p], 'alpha': 30.00 + demands_df[p], 'beta': 1.00, 'mean': 30.00})



pd.DataFrame(teta)



def gamma(alpha, beta):
    shape = alpha
    scale = 1/beta
    return np.random.gamma(shape, scale)


def sample_demand(price):
    demand = demand_a - demand_b * price
    return np.random.poisson(demand, 1)[0]



def sample_demands_from_model(θ):
    return list(map(lambda v: gamma(v['alpha'], v['beta']), θ))


import random

T = 100
history = []



for t in range(0, T):              # simulation loop
    demands = sample_demands_from_model(teta)
    print(tabulate(np.array(teta), tablefmt="fancy_grid"))
    
    print("demands = ", np.array(demands))
    

    price_probs = optimal_price_probabilities(prices, demands, np.mean(demands))
    
    # select one best price
    #price_index_t = np.random.choice(len(prices), 1, p=price_probs)[0]

    price_index_t = random.choices(range(len(prices)), weights = price_probs, k = 1)[0]


    price_t = prices[price_index_t]
    

    history.append(price_index_t)


    # sell at the selected price and observe demand
    demand_t = demands[price_index_t]

    print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t, demand_t, demand_t*price_t))


    # update model parameters
    procurando = teta[price_index_t]
    procurando['alpha'] = procurando['alpha'] + (demand_t * 0.2)
    procurando['beta'] = procurando['beta'] + (demand_t * 0.05)
    procurando['mean'] = procurando['alpha'] / procurando['beta']
    
    print("")




pd.Series(history).value_counts()




pd.DataFrame(history[-1])




# -----> VISUALIZAÇÃO <-----

def visualize_snapshot(t):
    plt.subplot(3, 1, 1)
    plt.xlabel('Demand')
    plt.ylabel('Demand PDF')
    plt.title('Demand PDFs for different prices')
    x = np.linspace(0, 60, 200) 
    for i, params in enumerate(history[t][3]):
        y = stats.gamma.pdf(x, a=params['alpha'], scale=1.0/params['beta']) 
        plt.plot(x, y, "-", label='price %.2f' % params['price']) 
    plt.legend(loc='upper left')
    plt.ylim([0, 0.5])
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.xlabel('Revenue')
    plt.ylabel('Revenue PDF')
    plt.title('Revenue PDFs for different prices')
    x = np.linspace(30, 110, 200) 
    for i, params in enumerate(history[t][3]):
        y = stats.gamma.pdf(x, a=params['alpha']*params['price'], scale=1.0/params['beta']) 
        plt.plot(x, y, "-", label='price %.2f' % params['price']) 
    plt.legend(loc='upper left')
    plt.ylim([0, 0.3])

    plt.subplot(3, 1, 3)
    plt.xlabel('Time')
    plt.ylabel('Demand/price')
    plt.title('Realized demand and price')
    prices = [h[0] for h in history]
    demands = [h[1] for h in history]
    plt.plot(range(0, t+1), np.array(prices)[0:t+1], 'r-') 
    plt.bar(range(0, T-1), np.pad(np.array(demands)[0:t+1], (0, T-2-t), 'constant'), 0.35, color='#9999ee')
    plt.ylim([0, 40])



fig = plt.figure(figsize = (10, 12))
plt.subplots_adjust(hspace = 0.5)
visualize_snapshot(T - 2)                 # fisualize the final state of the simulation
plt.show()