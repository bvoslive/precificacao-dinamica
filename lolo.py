import numpy as np
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import scipy


from sklearn.preprocessing import MinMaxScaler


precos = []

df = pd.read_csv('commodities_agrupados.csv')
df.iloc[20:40]
df['COMMODITY'].unique().tolist()[1:120]
df['COMMODITY'].value_counts()[380:400]


teste_precos = df[df['COMMODITY'] == '55072']
teste_precos.sort_values('PRECO')[:50]


teste_precos = teste_precos.drop(50640)
teste_precos = teste_precos.drop(50648)


teste_precos['QUANTIDADE'].mean()




prices = teste_precos['PRECO'].tolist()
prices

scaler = MinMaxScaler((0, 10))

prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

prices = [price[0] for price in prices]



demands_df = teste_precos['QUANTIDADE'].tolist()

scaler2 = MinMaxScaler((0, 10))

demands_df = scaler2.fit_transform([demands_df].reshape(-1, 1))











plt.scatter(prices, demands_df)
plt.show()



pd.DataFrame(np.array(prices) * np.array(demands_df)).sort_values(0)



np.log(np.array(demands) * np.array(prices))





"""
x = [5, 20, 50, 100, 120, 150]
y = [900, 550, 290, 220, 200, 180]
np.random.poisson(10, 1)[0]
plt.scatter(x, y)
plt.show()
poly_resultado = np.polyfit(x, y, 3)
from scipy.misc import derivative
def f(x, poly_resultado=poly_resultado):
    return poly_resultado[0] / (x ** 3 ) + poly_resultado[1] / (x ** 2) + poly_resultado[2] / x 
# plot 4/(x^(1/4)) + 3/(x^(1/3)) + 2/(x^(1/2)) + 81
def f(x, poly_resultado=poly_resultado):
    
    poly_resultado[0]
    poly_resultado[0]
    poly_resultado[0]
    poly_resultado[0]
-np.log(100000)
price = 5
y_k = f(price)
alpha = -derivative(f, price, dx=1e-6)
c = y_k - (alpha * price)
price = 15
alpha = derivative(f, price, dx=1e-6)
from scipy.misc import derivative
def f(x):
    return -x**3 - x**2
derivative(f, 1.0, dx=1e-6)
i = 1
alpha = derivative(f, x[i])
c = y[i] - alpha * x[i]
price = x[i]
slope = alpha * price
demanda_poisson = c + slope
from scipy.stats import linregress
resultado = linregress(x=[20, 10], y=[200, 100])
resultado.slope
resultado.intercept
"""





np.set_printoptions(precision=2)

def tabprint(msg, A):
    print(msg)
    print(tabulate(A, tablefmt="fancy_grid"))


def optimal_price_probabilities(prices, demands, inventory):   
    revenues = np.multiply(prices, demands)
    
    L = len(prices)
    M = np.full([1, L], 1)
    B = [[0.99]]
    Df = [demands]

    res = linprog(-np.array(revenues).flatten(), 
                  A_eq=M, 
                  b_eq=B, 
                  #A_ub=Df, 
                  #b_ub=np.array([inventory]), 
                  bounds=(0, None),
                  method='revised_simplex')

    price_prob = np.array(res.x).reshape(1, L).flatten()
    return price_prob


print(optimal_price_probabilities(prices, demands, demands[4]))





# Optimization procedure test
prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]
demands = list(map(lambda p: 50 - 7*p, prices))






revenues = np.multiply(prices, demands)
print(demands)
print(revenues)


# -----> ETAPA 2 <-----

#prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]

# Hidden (true) demand parameters - a linear demans function is assumed
#demand_a = 50
#demand_b = 7

# prior distribution for each price - gamma(α, β)
θ = []
for p in range(len(prices)):
    θ.append({'price': prices[p], 'alpha': 10.00 + demands_df[p], 'beta': 1.00, 'mean': 30.00})



pd.DataFrame(θ)



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
    demands = sample_demands_from_model(θ)
    print(tabulate(np.array(θ), tablefmt="fancy_grid"))
    
    print("demands = ", np.array(demands))
    

    price_probs = optimal_price_probabilities(prices, demands, demands[4])
    
    # select one best price
    #price_index_t = np.random.choice(len(prices), 1, p=price_probs)[0]

    price_index_t = random.choices(range(len(prices)), weights = price_probs, k = 1)[0]


    price_t = prices[price_index_t]
    
    # sell at the selected price and observe demand
    demand_t = demands[price_index_t]
    print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t, demand_t, demand_t*price_t))


    # update model parameters
    procurando = θ[price_index_t]
    procurando['alpha'] = procurando['alpha'] + (demand_t * 0.2)
    procurando['beta'] = procurando['beta'] + (demand_t * 0.05)
    procurando['mean'] = procurando['alpha'] / procurando['beta']
    
    print("")













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