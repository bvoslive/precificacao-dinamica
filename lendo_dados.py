import numpy as np
import pandas as pd


arquivo_nome = 'purchase-order-quantity-price-detail-for-commodity-goods-procurements-1.csv'

df = pd.read_csv(arquivo_nome)


df = df[['COMMODITY', 'COMMODITY_DESCRIPTION', 'QUANTITY', 'UNIT_PRICE', 'ITM_TOT_AM']]

df_com_preco = df.groupby(['COMMODITY', 'UNIT_PRICE'])['QUANTITY'].sum()
df_com_preco = df_com_preco[df_com_preco != 0]
df_com_preco.reset_index(inplace=True)
quantidade = df_com_preco.values
lista_com_preco = df_com_preco.index.tolist()
df_transacoes = pd.DataFrame(lista_com_preco, columns=['COMMODITY', 'PRECO'])
df_transacoes['QUANTIDADE'] = quantidade

df_transacoes.to_csv('commodities_agrupados.csv', index=False)