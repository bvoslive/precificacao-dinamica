import pandas as pd


arquivo_nome = 'purchase-order-quantity-price-detail-for-commodity-goods-procurements-1.csv'

df = pd.read_csv(arquivo_nome)


df = df[['COMMODITY', 'COMMODITY_DESCRIPTION', 'QUANTITY', 'UNIT_PRICE', 'ITM_TOT_AM']]


df.iloc[1]
