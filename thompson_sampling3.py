# IMPORTANDO BIBLIOTECAS
import numpy as np
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import random
from tqdm import tqdm

# IMPORTANDO DADOS
df = pd.read_csv('commodities_agrupados.csv')

# SELECIONANDO UMA COMMOODITY
COD_COMMODITY = '55072'
df = df[df['COMMODITY'] == COD_COMMODITY]
df = df.drop(50640)
df = df.drop(50648)
precos = df['PRECO'].tolist()
demanda_observada = df['QUANTIDADE'].tolist()
receita = np.multiply(precos, demanda_observada)
pesos_receita = receita / receita.sum()

# VISUALIZANDO PREÇOS E DEMANDA
#plt.scatter(precos, demanda_observada)
#plt.show()

# PREÇO MULTIPLICANDO DEMANDA
pd.DataFrame(np.array(precos) * np.array(demanda_observada)).sort_values(0)

# -----> ETAPA 2 <-----

# α = parâmetro de forma - Para valores de α muito altos, a distribuição gamma tende à Gaussiana
# β = parâmetro de escala - tem a função de ESTICAR OU ENCOLHER

def gamma(df):
    
    k = np.array([linha[2] for linha in df]) # forma
    theta = np.array([linha[3] for linha in df]) # escala

    return np.random.gamma(k, theta)





def gamma(alpha, beta):
    shape = alpha
    scale = 1/beta
    return np.random.gamma(shape, scale)


def calcula_gamma(teta):
    resultado_gamma = []
    for i in range(len(teta)):

        alpha = teta[i][2]
        beta = teta[i][3]

        resultado = gamma(alpha, beta)
        resultado_gamma.append(resultado)
    
    return resultado_gamma






# prior distribution for each price - gamma(α, β)
teta = []
for p in precos:
    # PREÇO, DEMANDA, ALPHA, BETA, MÉDIA
    teta.append(np.array([p, None, 30.00, 1.00, 30.00]))





T = 3000
lista_precos_index_escolhido= []
history = []

for t in tqdm(range(0, T)):
    
    # MODELANDO A DEMANDA
    demandas = calcula_gamma(teta)

    # ESTABELECENDO PROB DOS PREÇOS
    precos_log = np.log(precos)
    preco_probs = np.array(demandas) * np.array(precos_log)
    preco_probs = preco_probs / preco_probs.sum()

    # ESCOLHENDO UM PREÇO
    preco_index = random.choices(range(len(precos)), weights = preco_probs, k = 1)[0]

    # SELECIONANDO A DEMANDA ESCCOLHIDA
    demanda_escolhida = demandas[preco_index]
    demanda_escolhida = np.random.poisson(demanda_escolhida, 1)[0]

    # APPEND EM LISTAS
    history.append(teta)
    lista_precos_index_escolhido.append(preco_index)

    # ATUALIZANDO TETA
    teta[preco_index][1] = demanda_escolhida
    teta[preco_index][2] = teta[preco_index][2] + (demanda_escolhida * pesos_receita[preco_index])
    teta[preco_index][3] = teta[preco_index][3] + (demanda_escolhida * pesos_receita[preco_index] * 0.05)
    teta[preco_index][4] = teta[preco_index][2] / teta[preco_index][3]






pd.DataFrame(teta)

pd.Series(lista_precos_index_escolhido).value_counts()












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