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


def gamma(alpha, beta):
    shape = alpha
    scale = 1/beta
    return np.random.gamma(shape, scale)




T = 50
history = []
for t in range(0, T):              # simulation loop

    demands = list(map(lambda v: gamma(v['alpha'], v['beta']), teta))

    print(tabulate(np.array(teta), tablefmt="fancy_grid"))
    print("demands = ", np.array(demands))
    
    inventory = 60

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

    price_probs = np.array(res.x).reshape(1, L).flatten()

    print(demands)
    print(revenues)
    print(price_probs)


    # select one best price
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






















history[0][3]



prices = [h[0] for h in history]
demands = [h[1] for h in history]

procura_beta = history[4][3]



procura_beta



procura_beta['price']
