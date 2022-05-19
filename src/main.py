# IMPORTANDO BIBLIOTECAS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import seaborn as sns

# IMPORTANDO DADOS
df = pd.read_csv('./data/commodities_agrupados.csv')

# SELECIONANDO UMA COMMOODITY
COD_COMMODITY = '55072'
df = df[df['COMMODITY'] == COD_COMMODITY]
df = df.drop(50640)
df = df.drop(50648)
precos = df['PRECO'].tolist()
demanda_observada = df['QUANTIDADE'].tolist()
receita = np.multiply(precos, demanda_observada)
pesos_receita = receita / receita.sum()


# PREÇO MULTIPLICANDO DEMANDA

# -----> ETAPA 2 <-----

# α = parâmetro de forma - Para valores de α muito altos, a distribuição gamma tende à Gaussiana
# β = parâmetro de escala - tem a função de ESTICAR OU ENCOLHER
def gamma(teta):
    resultado_gamma = []
    for i in range(len(teta)):

        alpha = teta[i][2]
        beta = teta[i][3]

        resultado = np.random.gamma(alpha, 1/beta)
        resultado_gamma.append(resultado)
    
    return resultado_gamma


teta = []
for p in precos:
    # PREÇO, DEMANDA, ALPHA, BETA, MÉDIA
    teta.append(np.array([p, None, 30.00, 1.00, 0]))

iters = 3000
lista_precos_index_escolhido= []
lista_alpha = []

for i in tqdm(range(0, iters)):
    
    # MODELANDO A DEMANDA
    demandas = gamma(teta)

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
    historico_alpha = [teta[i][2] for i in range(len(teta))]
    lista_alpha.append(historico_alpha)
    lista_precos_index_escolhido.append(preco_index)

    # ATUALIZANDO TETA
    teta[preco_index][1] = demanda_escolhida
    teta[preco_index][2] = teta[preco_index][2] + (demanda_escolhida * pesos_receita[preco_index])
    teta[preco_index][3] = teta[preco_index][3] + (demanda_escolhida * pesos_receita[preco_index] * 0.05)
    teta[preco_index][4] += 1

# -----> ETAPA 3 <-----

# HISTÓRICO ALPHA
df_lista_alpha = pd.DataFrame(lista_alpha)

for i in range(len(df_lista_alpha)):
    plt.plot(df_lista_alpha[i], label=f'R${precos[i]}')
plt.legend()
plt.show()

colunas = ['PRECO', 'D_ESCOLHIDA', 'ALPHA', 'BETA', 'SUPORTE']
df_teta = pd.DataFrame(teta, columns = colunas)
plt.bar(pd.DataFrame(teta).index, pd.DataFrame(teta)[2])
plt.show()

sns.barplot(x='ALPHA', y='PRECO', data=df_teta, color='green', orient = 'h')
plt.show()
