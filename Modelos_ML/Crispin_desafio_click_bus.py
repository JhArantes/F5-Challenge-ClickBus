# modelo_predicitivo_compras_csv.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Carregar CSV
df = pd.read_csv('C:\\Users\\joaocrispin\\Documents\\faculdade\\desafio clickbus (1)\\desafio clickbus\\dados_desafio_fiap\\hash\\df_t.csv', parse_dates=['date_purchase'])

# 2. Agregar dados por mês
# Criando coluna ano-mês
df['AnoMes'] = df['date_purchase'].dt.to_period('M')
# Soma total de compras por mês
df_agg = df.groupby('AnoMes').agg({
    'gmv_success': 'sum',
    'total_tickets_quantity_success': 'sum'
}).reset_index()

# Transformar AnoMes em datetime para plot
df_agg['AnoMes'] = df_agg['AnoMes'].dt.to_timestamp()

# 3. Preparar dados para regressão
df_agg['mes_num'] = np.arange(len(df_agg))
X = df_agg['mes_num'].values.reshape(-1, 1)
y = df_agg['total_tickets_quantity_success'].values

# 4. Treinar modelo Linear
modelo = LinearRegression()
modelo.fit(X, y)

# 5. Previsão para 2025
meses_futuros = np.arange(len(df_agg), len(df_agg)+12).reshape(-1,1)
previsao_2025 = modelo.predict(meses_futuros)

# Criar DataFrame da previsão
datas_2025 = pd.date_range(start='2025-01-01', periods=12, freq='MS')
df_previsao = pd.DataFrame({
    'Data': datas_2025,
    'Quantidade_prevista': previsao_2025
})

# 6. Plot histórico + previsão
plt.figure(figsize=(10,6))
plt.plot(df_agg['AnoMes'], df_agg['total_tickets_quantity_success'], label='Histórico', marker='o')
plt.plot(df_previsao['Data'], df_previsao['Quantidade_prevista'], label='Previsão 2025', marker='x', linestyle='--')
plt.title('Previsão de Compras de Passagens para 2025')
plt.xlabel('Data')
plt.ylabel('Quantidade de Passagens')
plt.legend()
plt.grid(True)
plt.tight_layout()



# 7. Exibir tabela de previsão
print("Previsão de Compras para 2025:")
print(df_previsao)