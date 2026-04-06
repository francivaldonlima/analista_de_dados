# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║                   UTR221_ML_validacao.py                        ║
║     Validação e Análise Comparativa do Modelo — UTR-221         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OBJETIVO:                                                       ║
║    Este script é o "laboratório" de análise do modelo.           ║
║    Ele vai além do treinamento simples e realiza:                ║
║                                                                  ║
║    1. Treinamento de DOIS modelos:                               ║
║       - Modelo de 1 minuto (referência de curto prazo)           ║
║       - Modelo de 30 minutos (horizonte longo para simulação)    ║
║    2. Comparação das features importantes em cada horizonte      ║
║    3. Avaliação de métricas de acurácia                          ║
║    4. Previsão recursiva das próximas 7 HORAS (420 minutos)      ║
║       passo a passo, simulando o comportamento futuro            ║
║    5. Exportação dos resultados e gráficos                       ║
║                                                                  ║
║  DIFERENÇA PARA treinar_modelo.py:                               ║
║    - Este é para ANÁLISE E VALIDAÇÃO (estudo)                    ║
║    - treinar_modelo.py é para PRODUÇÃO (gera os .pkl)            ║
║    - Aqui usamos horizonte de 30 min (mais desafiador)           ║
║    - Aqui geramos a previsão futura de 7 horas                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DE ENCODING
# ═══════════════════════════════════════════════════════════════════
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Suporte a caracteres especiais do português

# ═══════════════════════════════════════════════════════════════════
# IMPORTAÇÃO DAS BIBLIOTECAS
# ═══════════════════════════════════════════════════════════════════

import pandas as pd   # Manipulação de dados em tabelas (DataFrames)
import numpy as np    # Operações matemáticas e arrays

# scikit-learn: biblioteca de Machine Learning
from sklearn.model_selection import train_test_split
    # Função para dividir dados em treino/teste (não usada diretamente,
    # mas importada para disponibilidade — usamos divisão manual)

from sklearn.linear_model import LinearRegression
    # Regressão Linear: modelo mais simples (não usado diretamente,
    # mas importado para referência e comparação futura)

from sklearn.ensemble import RandomForestRegressor
    # Random Forest: algoritmo principal deste projeto
    # Combina 200 árvores de decisão para fazer previsões robustas

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # Funções de avaliação:
    # mean_absolute_error = MAE: erro médio absoluto em BAR
    # mean_squared_error  = MSE: erro quadrático médio (penaliza erros grandes)
    # r2_score            = R²: coeficiente de determinação (0 a 1)

import matplotlib.pyplot as plt
    # Matplotlib: criação de gráficos para visualização dos resultados


# ═══════════════════════════════════════════════════════════════════
# ETAPA 1 — CARREGAMENTO DOS DADOS
# ═══════════════════════════════════════════════════════════════════
# Idêntico ao treinar_modelo.py. Lemos os dados brutos dos arquivos
# Excel e preparamos o DataFrame base para a engenharia de features.

# ── Carregar URT 220 (dados das bombas e do reservatório) ──────────
# usecols: ler apenas as colunas necessárias (economiza memória e tempo)
df220 = pd.read_excel('urt220.xlsx', usecols=[
    'E3TIMESTAMP',      # Data e hora da leitura
    'CMB_220_S2A_EST',  # Estado da Bomba 2A: 0=desligada, 1=ligada
    'CMB_220_S2B_EST',  # Estado da Bomba 2B: 0=desligada, 1=ligada
    'LIT_220_RA2_000'   # Nível do reservatório RA2 em metros
])

# O SCADA exporta o timestamp em formato não padrão:
# "01/02/26 00:00:27,954000000" — vírgula como decimal, 9 dígitos após o ponto
# Precisamos converter para o formato Python: "01/02/26 00:00:27.954000"
ts220 = (df220['E3TIMESTAMP'].astype(str)
         # Trocar vírgula por ponto: "27,954000000" → "27.954000000"
         .str.replace(',', '.', regex=False)
         # Truncar para 6 dígitos (microssegundos): "27.954000000" → "27.954000"
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))

# Converter o texto tratado para datetime do Python
df220['E3TIMESTAMP'] = pd.to_datetime(ts220, format='%d/%m/%y %H:%M:%S.%f')

# Definir timestamp como índice e ordenar do mais antigo ao mais recente
df220 = df220.set_index('E3TIMESTAMP').sort_index()

# ── Carregar URT 221 (pressão — variável alvo do modelo) ──────────
df221 = pd.read_excel('urt221.xlsx', usecols=['E3TIMESTAMP', 'PIT_221_S01_000'])
    # PIT_221_S01_000 = Pressão da URT 221 — é isso que queremos prever

ts221 = (df221['E3TIMESTAMP'].astype(str)
         .str.replace(',', '.', regex=False)
         .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))
df221['E3TIMESTAMP'] = pd.to_datetime(ts221, format='%d/%m/%y %H:%M:%S.%f')
df221 = df221.set_index('E3TIMESTAMP').sort_index()

# ── Regularizar para 1 leitura por minuto ─────────────────────────
# resample('1min').mean() = agrupa em janelas de 1 minuto, calcula média
# dropna() = remove linhas onde todos os valores são NaN (gaps reais)
df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

# ── Combinar os dois DataFrames num único ─────────────────────────
# join com how='inner' = manter apenas timestamps presentes em AMBAS as UTRs
# Isso garante que para cada linha temos tanto as features (220) quanto o alvo (221)
df = df220.join(df221, how='inner')

# ── Filtrar período de análise ────────────────────────────────────
# Usar apenas dados após esta data/hora (sistema estável e com dados completos)
df = df.loc['2026-02-28 17:30:38':]

# ── Renomear colunas para nomes mais descritivos ───────────────────
df = df.rename(columns={
    'CMB_220_S2A_EST': 'BOMBA_1',         # Bomba principal
    'CMB_220_S2B_EST': 'BOMBA_2',         # Bomba do rodízio
    'LIT_220_RA2_000': 'NIVEL-UTR-220',   # Nível do reservatório (metros)
    'PIT_221_S01_000': 'PRESSAO-UTR-221'  # Pressão da URT 221 (BAR) — TARGET
})


# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DO HORIZONTE DE PREVISÃO
# ═══════════════════════════════════════════════════════════════════
# HORIZONTE_MIN define quantos minutos à frente o modelo prevê.
# Altere este valor para testar diferentes horizontes:
#   1  = prever o próximo minuto (mais fácil, mais preciso)
#   15 = prever 15 min à frente (médio prazo)
#   30 = prever 30 min à frente (longo prazo, mais desafiador)
# ──────────────────────────────────────────────────────────────────
HORIZONTE_MIN = 30   # minutos à frente para prever (modelo principal)
# ──────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
# ETAPA 2 — ENGENHARIA DE FEATURES (CRIAÇÃO DE VARIÁVEIS)
# ═══════════════════════════════════════════════════════════════════
# Criamos variáveis derivadas que ajudam o modelo a entender o contexto.
# Quanto maior o horizonte de previsão, mais o modelo precisa de contexto
# histórico para fazer uma boa estimativa.

# ── Variáveis Alvo (Target) ────────────────────────────────────────
# Criamos DOIS targets para comparar os horizontes de previsão:
# shift(-1)         = pressão daqui a 1 minuto (referência)
# shift(-HORIZONTE) = pressão daqui a N minutos (principal)
df['target_1min']  = df['PRESSAO-UTR-221'].shift(-1)
    # Referência de curto prazo — fácil de prever, serve como base
df['target']       = df['PRESSAO-UTR-221'].shift(-HORIZONTE_MIN)
    # Target principal — previsão de 30 minutos à frente

# ── Lags (Memória Temporal) ───────────────────────────────────────
# Lags trazem o histórico de pressão para o modelo.
# O modelo não "vê" a série temporal; ele só vê as features de cada linha.
# Os lags traduzem a dependência temporal em features numéricas.
df['nivel_221_lag1']  = df['PRESSAO-UTR-221'].shift(1)
    # Pressão de 1 minuto atrás — valor mais recente conhecido
df['nivel_221_lag5']  = df['PRESSAO-UTR-221'].shift(5)
    # Pressão de 5 minutos atrás — tendência de curtíssimo prazo
df['nivel_221_lag15'] = df['PRESSAO-UTR-221'].shift(15)
    # Pressão de 15 minutos atrás — ciclo de bomba (geralmente dura ~15min)
df['nivel_220_lag1']  = df['NIVEL-UTR-220'].shift(1)
    # Nível do reservatório 1 minuto atrás — afeta a pressão indiretamente

# ── Tendências (Taxa de Mudança da Pressão) ───────────────────────
# A tendência indica se a pressão está subindo ou descendo.
# Para horizonte de 30 min, a tendência de 1h pode ser mais informativa.
df['tendencia_1min'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(2)
    # Variação minuto a minuto: captura mudanças imediatas
df['tendencia_5min'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(6)
    # Variação em 5 minutos: captura tendência de curto prazo
df['tendencia_1h']   = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(60)
    # Variação em 1 hora: captura padrão diário (muito útil para 30 min)

# ── Médias Móveis (Valores Suavizados) ────────────────────────────
# Médias móveis suavizam ruídos e revelam a tendência real.
# SEMPRE usar shift(1) antes do rolling para evitar data leakage!
df['media_movel_1h']  = df['PRESSAO-UTR-221'].shift(1).rolling(window=60).mean()
    # Média da pressão na última hora — contexto de médio prazo
df['media_movel_15m'] = df['PRESSAO-UTR-221'].shift(1).rolling(window=15).mean()
    # Média da pressão nos últimos 15 minutos — contexto recente

# ── Features Temporais ────────────────────────────────────────────
# A hora do dia é fundamental: à noite o consumo é menor,
# durante o dia há picos de demanda que afetam a pressão.
df['hora'] = df.index.hour  # Hora de 0 a 23

# Turnos do dia (categorias para facilitar a aprendizagem):
# madrugada (0h-5h), manhã (6h-11h), tarde (12h-17h), noite (18h-23h)
df['turno'] = pd.cut(df.index.hour,
                     bins=[-1, 5, 11, 17, 23],
                     labels=[0, 1, 2, 3]).astype(int)
    # Converter para int pois o modelo espera números, não categorias

# ── Features de Rodízio das Bombas ────────────────────────────────
# Este bloco captura o comportamento específico do rodízio entre Bomba1 e Bomba2.
# O rodízio é um padrão operacional importante: as bombas se alternam para
# distribuir o desgaste, e cada bomba pode ter uma curva de pressão diferente.

# Transição da Bomba 2: momento em que a bomba liga ou desliga
# diff() = diferença entre valor atual e anterior
# +1 = bomba ligou, -1 = bomba desligou, 0 = sem mudança
df['transicao_B2'] = df['BOMBA_2'].diff().fillna(0)
    # fillna(0) = o primeiro valor não tem anterior → sem transição

# Tempo em minutos que a Bomba 2 está continuamente ligada
# Algoritmo:
#   1. Detectar cada vez que o estado muda (True/False)
#   2. Numerar cada "grupo" de estado constante com cumsum()
#   3. Dentro de cada grupo, contar 0, 1, 2, 3... (minutos decorridos)
#   4. Multiplicar por 0 (se bomba desligada) para zerar o contador
grupo_b2 = (df['BOMBA_2'] != df['BOMBA_2'].shift()).cumsum()
    # cumsum() acumula o número de mudanças → ID único por período de estado constante
df['tempo_ligada_B2'] = (
    df.groupby(grupo_b2)['BOMBA_2']
    # transform com lambda: aplicar uma função a cada grupo
    # range(len(x)) = sequência 0,1,2... dentro do grupo
    .transform(lambda x: range(len(x)))
    # Multiplicar pelo estado: se BOMBA_2=0, zera o contador
    * df['BOMBA_2']
)

# Indicador de rodízio: qual bomba está em operação no momento
# np.where = IF vetorizado: np.where(condição, se_verdadeiro, se_falso)
# Resultado: 1 se Bomba1 ligada, 2 se Bomba2 ligada, 0 se ambas desligadas
df['rodizio'] = np.where(df['BOMBA_1'] == 1, 1,
               np.where(df['BOMBA_2'] == 1, 2, 0))

# Estado da Bomba 2 no minuto anterior (lag 1)
df['BOMBA_2_lag1'] = df['BOMBA_2'].shift(1)

# Proporção do tempo ativo nas últimas 15 leituras (0 a 1)
# Útil para capturar padrões de funcionamento das bombas
df['B2_media_15m'] = df['BOMBA_2'].shift(1).rolling(window=15).mean()
df['B1_media_15m'] = df['BOMBA_1'].shift(1).rolling(window=15).mean()

# Remover linhas com NaN (criadas pelos shift e rolling dos primeiros/últimos registros)
df = df.dropna()

# ── Filtros de qualidade dos dados ────────────────────────────────
# df2: dataset principal para treino — remove casos inválidos
# target=0 pode indicar falha de sensor ou período sem operação
# nivel_221_lag1=0 indica que não havia pressão histórica válida
df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

# df3: dataset auxiliar para análise — linhas onde a pressão era zero
# Pode ser usado para entender quando e por que o sistema fica sem pressão
df3 = df[df['PRESSAO-UTR-221'] == 0].copy()


# ═══════════════════════════════════════════════════════════════════
# ETAPA 3 — DEFINIÇÃO DAS FEATURES E DIVISÃO DOS DADOS
# ═══════════════════════════════════════════════════════════════════

# Lista de features (variáveis de entrada) — ordem importa para reprodutibilidade
features = [
    'BOMBA_1',          # Estado atual da Bomba 1 (0 ou 1)
    'BOMBA_2',          # Estado atual da Bomba 2 (0 ou 1)
    'BOMBA_2_lag1',     # Estado da Bomba 2 no minuto anterior
    'B1_media_15m',     # Fração do tempo ligada da Bomba 1 nos últimos 15 min
    'B2_media_15m',     # Fração do tempo ligada da Bomba 2 nos últimos 15 min
    'nivel_220_lag1',   # Nível do reservatório UTR-220 (1 min atrás)
    'nivel_221_lag1',   # Pressão UTR-221 1 min atrás (lag mais importante)
    'nivel_221_lag5',   # Pressão UTR-221 5 min atrás
    'nivel_221_lag15',  # Pressão UTR-221 15 min atrás
    'hora',             # Hora do dia (0 a 23)
    'turno',            # Turno: 0=madrugada, 1=manhã, 2=tarde, 3=noite
    'tendencia_1min',   # Variação de pressão no último minuto
    'tendencia_5min',   # Variação de pressão nos últimos 5 minutos
    'tendencia_1h',     # Variação de pressão na última hora
    'media_movel_1h',   # Média de pressão na última hora
    'media_movel_15m',  # Média de pressão nos últimos 15 minutos
    'transicao_B2',     # Mudança de estado da Bomba 2
    'tempo_ligada_B2',  # Minutos consecutivos com Bomba 2 ligada
    'rodizio',          # Identificador da bomba em operação (1, 2 ou 0)
]

# Extrair matrizes de features e vetores de alvo
X2 = df2[features]      # Matriz X: linhas = minutos, colunas = 19 features
y2 = df2['target']      # Vetor y: pressão daqui a HORIZONTE_MIN minutos

# Diagnóstico inicial: tamanho dos datasets
print(f"DF  (original):      {len(df)} linhas")
print(f"DF2 (sem target=0):  {len(df2)} linhas  ({len(df) - len(df2)} removidas)")
print(f"DF3 (NIVEL-221=0):   {len(df3)} linhas")
print(f"Horizonte de previsão: {HORIZONTE_MIN} minutos à frente")


# ═══════════════════════════════════════════════════════════════════
# ETAPA 4 — DIVISÃO TEMPORAL E TREINAMENTO DOS DOIS MODELOS
# ═══════════════════════════════════════════════════════════════════
# Dividimos os dados em treino (80% iniciais) e teste (20% finais).
# Treinamos DOIS modelos para comparação:
#   - modelo_1min: prevê 1 minuto à frente (baseline fácil)
#   - modelo:      prevê HORIZONTE_MIN minutos à frente (desafiador)

# Ponto de corte: 80% dos dados para treino
split_point = int(len(df2) * 0.8)

# Divisão temporal dos dados de features
X_train, X_test = X2.iloc[:split_point], X2.iloc[split_point:]

# Divisão temporal do target principal (HORIZONTE_MIN minutos)
y_train, y_test = y2.iloc[:split_point], y2.iloc[split_point:]

# ── Modelo Principal: HORIZONTE_MIN minutos à frente ──────────────
# n_estimators=200: 200 árvores de decisão (bom balanço precisão/velocidade)
# random_state=42:  reprodutibilidade (mesmos resultados em qualquer execução)
# n_jobs=-1:        usar todos os núcleos do CPU em paralelo
modelo = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)   # .fit() = fase de aprendizado

# ── Modelo de Referência: 1 minuto à frente ───────────────────────
# Usamos as mesmas features (X_train) mas com targets de 1 minuto
# Isso permite comparar: "o que o modelo aprendeu que é importante
# para prever em 1 min vs. em 30 min?"
y2_1min      = df2['target_1min']   # Target de 1 minuto
y_train_1min = y2_1min.iloc[:split_point]
y_test_1min  = y2_1min.iloc[split_point:]

modelo_1min = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
modelo_1min.fit(X_train, y_train_1min)  # Mesmo X_train, target diferente


# ═══════════════════════════════════════════════════════════════════
# ETAPA 5 — PREVISÃO E COMPARAÇÃO DE IMPORTÂNCIA DAS FEATURES
# ═══════════════════════════════════════════════════════════════════
# Aplicar os modelos treinados nos dados de TESTE (nunca vistos).
# .predict() = usar o modelo para gerar previsões
previsoes      = modelo.predict(X_test)       # Previsões do modelo principal
previsoes_1min = modelo_1min.predict(X_test)  # Previsões do modelo de referência

# ── Análise de Importância das Features ───────────────────────────
# feature_importances_ = quanto cada feature contribuiu para as previsões
# Valores entre 0 e 1, somando 1.0 no total
# Comparar 1 min vs. 30 min revela quais padrões são importantes em cada escala
imp_1min = pd.Series(modelo_1min.feature_importances_, index=features).sort_values(ascending=False)
    # Ordenar da mais para a menos importante (no modelo de 1 min)
imp_Nmin = pd.Series(modelo.feature_importances_,     index=features).sort_values(ascending=False)
    # Ordenar da mais para a menos importante (no modelo de N min)

# Exibir tabela comparativa de importância
print("\n" + "=" * 65)
print(f"   COEFICIENTES — 1 min  ×  {HORIZONTE_MIN} min à frente")
print("=" * 65)
# Cabeçalho da tabela formatada
print(f"{'Feature':<20}  {'1 min':>7}  {str(HORIZONTE_MIN)+' min':>7}  {'Variação':>9}")
print("-" * 65)

for feat in imp_Nmin.index:  # Iterar na ordem do modelo principal
    v1 = imp_1min[feat]  # Importância no modelo de 1 min
    vN = imp_Nmin[feat]  # Importância no modelo de N min

    # Seta indicando se a importância subiu, desceu ou manteve
    # 0.001 = tolerância para evitar falsas variações por arredondamento
    seta = "↑" if vN > v1 + 0.001 else ("↓" if vN < v1 - 0.001 else " =")

    # Barra visual proporcional (até 30 caracteres)
    barra = "█" * int(vN * 30)

    # Imprimir linha formatada: feature, importância 1min, importância Nmin, variação, barra
    print(f"{feat:<20}  {v1:>7.4f}  {vN:>7.4f}  {seta:>4}  {barra}")

print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# ETAPA 6 — MÉTRICAS DE ACURÁCIA DOS MODELOS
# ═══════════════════════════════════════════════════════════════════
# Calculamos as métricas para AMBOS os modelos para comparação.

def metricas(y_real, y_pred, label):
    """
    Calcula e exibe as métricas de avaliação de um modelo de regressão.

    PARÂMETROS:
      y_real: valores reais (medidos pelo sensor)
      y_pred: valores previstos pelo modelo
      label:  nome do modelo para identificação no console

    RETORNA:
      tuple: (mae, rmse, r2, mape) — valores calculados

    MÉTRICAS EXPLICADAS:
      MAE  (Mean Absolute Error):
        Erro médio absoluto. Se MAE=0.03, o modelo erra em média 0.03 BAR.
        Interpretação intuitiva, mesma unidade que a variável.

      RMSE (Root Mean Squared Error):
        Raiz do erro quadrático médio. Penaliza mais os erros grandes.
        Se RMSE > MAE, há alguns erros grandes (outliers de previsão).

      R² (Coeficiente de Determinação):
        Quanto da variação real o modelo explica.
        R²=1.0 = perfeito, R²=0.0 = não melhor que prever a média.

      MAPE (Mean Absolute Percentage Error):
        Erro médio em porcentagem. Independente da escala.
        MAPE=2.0% significa que o modelo erra em média 2% do valor real.
    """
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    # Calcular MAPE apenas onde o valor real não é zero (evitar divisão por zero)
    mask = y_real != 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100

    # Exibir resultados formatados
    print(f"\n  [{label}]")
    print(f"  MAE  : {mae:.4f} bar    RMSE : {rmse:.4f} bar")
    print(f"  R²   : {r2:.4f}         MAPE : {mape:.2f}%    Acurácia: {100-mape:.2f}%")

    return mae, rmse, r2, mape

# Exibir métricas para comparação dos dois horizontes
print("\n" + "=" * 65)
print("   MÉTRICAS")
print("=" * 65)
metricas(y_test_1min, previsoes_1min, f"Modelo  1 min")
metricas(y_test,      previsoes,      f"Modelo {HORIZONTE_MIN} min")
print("=" * 65)

# Calcular métricas do modelo principal (serão usadas na previsão futura)
mae      = mean_absolute_error(y_test, previsoes)
rmse_val = np.sqrt(mean_squared_error(y_test, previsoes))
r2_val   = r2_score(y_test, previsoes)
mask     = y_test != 0
mape     = np.mean(np.abs((y_test[mask] - previsoes[mask]) / y_test[mask])) * 100
acuracia = 100 - mape


# ═══════════════════════════════════════════════════════════════════
# ETAPA 7 — PREVISÃO FUTURA: PRÓXIMAS 7 HORAS (420 MINUTOS)
# ═══════════════════════════════════════════════════════════════════
# Esta é a etapa mais sofisticada: previsão recursiva (passo a passo).
#
# COMO FUNCIONA A PREVISÃO RECURSIVA?
#   1. Iniciamos com os últimos dados reais conhecidos
#   2. Para cada minuto futuro:
#      a. Calculamos as features usando o histórico disponível
#      b. O modelo prevê a pressão daquele minuto
#      c. A previsão é adicionada ao histórico
#      d. No próximo passo, usamos esse histórico atualizado
#   3. Repetimos por 420 minutos (7 horas)
#
# POR QUÊ É DESAFIADOR?
#   Os erros se acumulam: cada previsão errada alimenta a próxima.
#   Por isso, para horizontes muito longos, as previsões ficam menos precisas.
#   Incluímos uma lógica de controle das bombas (setpoints) para
#   simular o comportamento real do sistema de automação.

print("\n" + "=" * 70)
print("   PREVISÃO FUTURA: PRÓXIMAS 7 HORAS (420 min)")
print("=" * 70)

# Último timestamp com dados reais disponíveis
ultimo_ts = df.index[-1]
print(f"Último dado real: {ultimo_ts}")
print(f"Modelo utilizado: {HORIZONTE_MIN} min à frente")

# Total de passos futuros: 7 horas × 60 minutos = 420 passos
passos_futuro = 420

# Buffer de histórico da pressão: últimos 60 minutos reais
# Necessário para calcular lags de até lag60 (tendência de 1 hora)
historico = list(df['PRESSAO-UTR-221'].tail(60).values)
    # .tail(60) = 60 últimas linhas
    # .values = converter para array numpy
    # list() = converter para lista Python (mais fácil de fazer .append())

# ── Estado inicial das bombas ──────────────────────────────────────
# Iniciamos a previsão com o estado real das bombas no momento atual
bomba1     = float(df['BOMBA_1'].iloc[-1])    # Estado da Bomba 1 (0.0 ou 1.0)
bomba2     = float(df['BOMBA_2'].iloc[-1])    # Estado da Bomba 2 (0.0 ou 1.0)
bomba2_ant = float(df['BOMBA_2'].iloc[-2])    # Estado anterior da Bomba 2 (para calcular transição)
nivel220_const = float(df['NIVEL-UTR-220'].iloc[-1])  # Nível do reservatório (assumido constante)
tempo_b2   = int(df['tempo_ligada_B2'].iloc[-1])      # Minutos que a Bomba 2 já está ligada

# Histórico das bombas nos últimos 15 minutos (para as médias B1_media_15m e B2_media_15m)
hist_b1 = list(df['BOMBA_1'].tail(15).values)
hist_b2 = list(df['BOMBA_2'].tail(15).values)

# Setpoints de controle (lógica de acionamento automático das bombas)
SETPOINT_ALTO  = 1.68  # BAR: pressão que DESLIGA as bombas
SETPOINT_BAIXO = 1.55  # BAR: pressão que LIGA as bombas

# Lista para acumular os resultados de cada passo futuro
registros_futuro = []

# ── Loop de previsão recursiva ─────────────────────────────────────
for passo in range(1, passos_futuro + 1):
    # Calcular o timestamp do próximo minuto previsto
    proximo_ts = ultimo_ts + pd.Timedelta(minutes=passo)

    # ── Lags da pressão (valores do histórico) ────────────────────
    nivel_lag1  = historico[-1]    # Último valor disponível (t-1)
    nivel_lag2  = historico[-2]    # 2 minutos atrás (t-2)
    nivel_lag6  = historico[-6]  if len(historico) >= 6  else historico[0]
        # 6 minutos atrás — ou o primeiro valor se não tiver 6 registros
    nivel_lag15 = historico[-15] if len(historico) >= 15 else historico[0]
        # 15 minutos atrás
    nivel_lag60 = historico[-60] if len(historico) >= 60 else historico[0]
        # 60 minutos atrás (para tendência de 1 hora)

    # Guardar o estado anterior da Bomba 2 para calcular a transição
    bomba2_ant = bomba2

    # ── Lógica de controle por setpoints ──────────────────────────
    # Simular o comportamento do controlador automático do sistema:
    # - Se pressão >= setpoint alto: desligar as bombas (sistema pressurizado)
    # - Se pressão <= setpoint baixo: ligar as bombas (sistema com baixa pressão)
    if nivel_lag1 >= SETPOINT_ALTO:
        bomba1, bomba2 = 0.0, 0.0  # Desligar ambas as bombas
        tempo_b2 = 0                # Zerar o contador de tempo da Bomba 2
    elif nivel_lag1 <= SETPOINT_BAIXO:
        bomba1, bomba2 = 1.0, 1.0  # Ligar ambas as bombas

    # Atualizar o contador de tempo da Bomba 2
    if bomba2 == 1:
        tempo_b2 += 1  # Incrementar se ligada
    else:
        tempo_b2 = 0   # Zerar se desligada

    # Atualizar históricos das bombas (janela deslizante de 15 posições)
    hist_b1.append(bomba1)
    hist_b1 = hist_b1[-15:]  # Manter apenas os últimos 15 valores
    hist_b2.append(bomba2)
    hist_b2 = hist_b2[-15:]

    # Calcular features de turno (dependem do timestamp futuro)
    rodizio_val  = 1 if bomba1 == 1 else (2 if bomba2 == 1 else 0)
    transicao_b2 = bomba2 - bomba2_ant  # +1 se ligou, -1 se desligou, 0 se estável

    hora_val  = proximo_ts.hour
    # Classificar a hora em turno: 0=madrugada, 1=manhã, 2=tarde, 3=noite
    turno_val = 0 if hora_val < 6 else (1 if hora_val < 12 else (2 if hora_val < 18 else 3))

    # ── Montar o dicionário de features para este passo ───────────
    # Usamos um dicionário para clareza — será convertido em DataFrame
    feat_values = {
        'BOMBA_1':         bomba1,
        'BOMBA_2':         bomba2,
        'BOMBA_2_lag1':    bomba2_ant,
        'B1_media_15m':    np.mean(hist_b1),   # Média do histórico de 15 min
        'B2_media_15m':    np.mean(hist_b2),
        'nivel_220_lag1':  nivel220_const,      # Assumimos nível constante do reservatório
        'nivel_221_lag1':  nivel_lag1,
        'nivel_221_lag5':  nivel_lag6,          # Nota: lag5 usa posição -6 no histórico
        'nivel_221_lag15': nivel_lag15,
        'hora':            hora_val,
        'turno':           turno_val,
        'tendencia_1min':  nivel_lag1 - nivel_lag2,    # Variação no último minuto
        'tendencia_5min':  nivel_lag1 - nivel_lag6,    # Variação nos últimos 5 min
        'tendencia_1h':    nivel_lag1 - nivel_lag60,   # Variação na última hora
        'media_movel_1h':  np.mean(historico[-60:]),   # Média da última hora
        'media_movel_15m': np.mean(historico[-15:]),   # Média dos últimos 15 min
        'transicao_B2':    transicao_b2,
        'tempo_ligada_B2': tempo_b2,
        'rodizio':         rodizio_val,
    }

    # Converter dicionário para DataFrame de 1 linha
    # [features] garante a ordem correta das colunas (igual ao treino)
    X_futuro = pd.DataFrame([feat_values])[features]

    # Fazer a previsão com o modelo treinado
    # max(..., 0) = pressão não pode ser negativa (limite físico)
    previsao = max(modelo.predict(X_futuro)[0], 0)

    # Salvar os resultados deste passo
    registros_futuro.append({
        'Data':            proximo_ts,
        'Previsto_Futuro': round(previsao, 4),  # Pressão prevista em BAR
        'BOMBA_1':         bomba1,
        'BOMBA_2':         bomba2,
        'Rodizio':         rodizio_val,
        'Passo':           passo,                # Número do passo (1 a 420)
        'Hora':            proximo_ts.strftime('%H:%M'),  # Hora formatada
    })

    # Adicionar a previsão ao histórico para o próximo passo
    # É aqui que acontece a recursividade: previsão vira dado histórico
    historico.append(previsao)

# ── Criar DataFrame com todos os resultados futuros ───────────────
df_futuro = pd.DataFrame(registros_futuro).set_index('Data')
print(f"Previsão gerada: {len(df_futuro)} passos (7h futuras)")

# Exibir os primeiros 10 e últimos 10 passos da previsão
print(df_futuro[['Previsto_Futuro', 'Hora']].head(10))
print(df_futuro[['Previsto_Futuro', 'Hora']].tail(10))

# Exportar previsão futura para Excel
df_futuro.to_excel('DF6_futuro_7h.xlsx')
print("✅ Arquivo gerado: DF6_futuro_7h.xlsx")


# ═══════════════════════════════════════════════════════════════════
# ETAPA 8 — GERAÇÃO DE GRÁFICOS
# ═══════════════════════════════════════════════════════════════════
# Criamos 2 gráficos para visualização dos resultados da análise.

# ── Gráfico 1: Comparação de Importância das Features ─────────────
# Mostra lado a lado a importância de cada feature para prever em
# 1 minuto vs. HORIZONTE_MIN minutos. Muito útil para entender o modelo.

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    # 1 linha, 2 colunas; sharey=True = mesmo eixo Y nos dois gráficos

# Ordenar features pela importância no modelo principal (N min)
feats_ord = imp_Nmin.index.tolist()

# Colorir features relacionadas à Bomba 2 de vermelho, o resto de azul
# Isso destaca visualmente quais features estão relacionadas ao rodízio
cores = ['#e74c3c' if 'BOMBA_2' in f or 'B2' in f or 'rodizio' in f
         else '#3498db' for f in feats_ord]
    # List comprehension: para cada feature, escolher a cor com base no nome

# Subplot esquerdo: modelo de 1 minuto (cinza neutro)
axes[0].barh(feats_ord, [imp_1min[f] for f in feats_ord], color='#95a5a6')
    # barh = gráfico de barras horizontais
axes[0].set_title('Importância — 1 min à frente', fontsize=12)
axes[0].set_xlabel('Importância')
axes[0].invert_yaxis()  # Inverte eixo Y: feature mais importante fica no topo
axes[0].grid(axis='x', alpha=0.3)  # Grade vertical suave

# Subplot direito: modelo de N minutos (colorido por categoria)
axes[1].barh(feats_ord, [imp_Nmin[f] for f in feats_ord], color=cores)
axes[1].set_title(f'Importância — {HORIZONTE_MIN} min à frente\n(vermelho = BOMBA_2 / rodízio)', fontsize=12)
axes[1].set_xlabel('Importância')
axes[1].grid(axis='x', alpha=0.3)

# Título geral da figura (acima dos dois subplots)
plt.suptitle('Comparação de Importância das Features', fontsize=14, fontweight='bold')
plt.tight_layout()   # Ajustar espaçamento automático
plt.savefig('coeficientes_comparativo.png', dpi=200, bbox_inches='tight')
    # dpi=200: alta resolução para apresentações e relatórios
print("✅ Gráfico salvo: coeficientes_comparativo.png")

# ── Gráfico 2: Série Real + Previsão Futura 7 Horas ───────────────
# Mostra a última hora de dados reais e as 7 horas de previsão futura.
# É a visualização mais impactante do projeto!

# Definir janela de contexto: 1 hora antes do início da previsão
janela_grafico_inicio = ultimo_ts - pd.Timedelta(hours=1)
dados_grafico = df.loc[janela_grafico_inicio:ultimo_ts].copy()
    # .loc[inicio:fim] = selecionar por intervalo de índice datetime

fig2, ax = plt.subplots(figsize=(16, 7))  # Figura grande para visualização clara

# Linha 1: dados REAIS da última hora (referência)
ax.plot(dados_grafico.index, dados_grafico['PRESSAO-UTR-221'],
        label='Última 1h Real', color='steelblue', linewidth=2.5)

# Linha 2: previsão das próximas 7 horas (tracejada laranja)
ax.plot(df_futuro.index, df_futuro['Previsto_Futuro'],
        label=f'Previsão 7h (modelo {HORIZONTE_MIN} min)', color='darkorange',
        linewidth=2, linestyle='--')  # linestyle='--' = linha tracejada

# Linha vertical marcando onde começa a previsão
ax.axvline(ultimo_ts, color='red', linestyle=':', alpha=0.8, label='Início da previsão')

# Linhas horizontais dos setpoints de controle das bombas
ax.axhline(1.68, color='gray', linestyle='--', alpha=0.5, label='Setpoint alto (1.68)')
ax.axhline(1.55, color='gray', linestyle=':',  alpha=0.5, label='Setpoint baixo (1.55)')

# Título com informações do contexto
ax.set_title(f'Previsão Futura 7 Horas — Modelo {HORIZONTE_MIN} min\nA partir de {ultimo_ts}',
             fontsize=13)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Pressão (BAR)')
ax.legend()               # Mostrar legenda
ax.grid(True, alpha=0.3)  # Grade suave
fig2.autofmt_xdate()      # Rotacionar rótulos do eixo X para não sobrepor
plt.tight_layout()
plt.savefig('previsao_futura_7h.png', dpi=300, bbox_inches='tight')
    # dpi=300: máxima qualidade para publicação
print("✅ Gráfico salvo: previsao_futura_7h.png")

# Exibir os gráficos na tela (abre janela interativa)
plt.show()

print("=" * 70)
