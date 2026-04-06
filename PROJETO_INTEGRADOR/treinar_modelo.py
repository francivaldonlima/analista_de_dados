# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║                      treinar_modelo.py                          ║
║        Treinamento do Modelo Random Forest — UTR-221            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OBJETIVO:                                                       ║
║    Treinar um modelo de Machine Learning (Random Forest) capaz   ║
║    de prever a pressão da UTR-221 1 minuto à frente,             ║
║    usando como entrada os dados históricos das UTRs 220 e 221.   ║
║                                                                  ║
║  O QUE É RANDOM FOREST?                                         ║
║    Random Forest (Floresta Aleatória) é um algoritmo que cria    ║
║    muitas "árvores de decisão" (n_estimators=200 neste caso)    ║
║    e combina suas previsões. É robusto, eficiente e funciona     ║
║    bem com dados reais e ruidosos de sistemas industriais.       ║
║                                                                  ║
║  ARTEFATOS GERADOS (arquivos de saída):                          ║
║    modelo_utr221.pkl    → modelo treinado (Random Forest)        ║
║    historico_utr221.pkl → série histórica completa               ║
║                          (usada para inicializar o dashboard)    ║
║    meta_utr221.pkl      → metadados do modelo:                   ║
║                          features usadas, métricas, timestamps   ║
║                                                                  ║
║  COMO EXECUTAR:                                                  ║
║    python treinar_modelo.py                                      ║
║                                                                  ║
║  DEPENDÊNCIAS:                                                   ║
║    pip install pandas numpy scikit-learn openpyxl joblib         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DE ENCODING (CODIFICAÇÃO DE CARACTERES)
# ═══════════════════════════════════════════════════════════════════
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Garante exibição correta de acentos no Windows


# ═══════════════════════════════════════════════════════════════════
# IMPORTAÇÃO DAS BIBLIOTECAS
# ═══════════════════════════════════════════════════════════════════

import pandas as pd    # Manipulação de tabelas de dados (DataFrames)
import numpy as np     # Operações matemáticas e arrays numéricos

# scikit-learn: a principal biblioteca de Machine Learning do Python
from sklearn.ensemble import RandomForestRegressor
# RandomForestRegressor = Random Forest para problemas de REGRESSÃO
# (quando queremos prever um valor numérico contínuo, como pressão em BAR)
# Diferença: Classification = categorias (ex: sim/não), Regression = números

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Funções para medir a qualidade das previsões do modelo:
# mean_absolute_error (MAE) = erro médio absoluto em BAR
# mean_squared_error (MSE)  = erro quadrático médio (penaliza erros grandes)
# r2_score (R²)             = coeficiente de determinação (0 a 1, quanto maior melhor)

import joblib
# joblib: biblioteca para salvar e carregar objetos Python (como modelos treinados)
# Equivalente a "serializar" o modelo em um arquivo .pkl (pickle)
# Isso permite usar o modelo treinado no dashboard sem precisar re-treinar


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONFIGURAÇÕES DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Parâmetros que definem o comportamento do modelo e do sistema.
# Altere aqui para experimentar diferentes configurações.

HORIZONTE_MIN = 1   # Quantos minutos à frente o modelo vai prever
                    # Ex: se agora são 14:00, o modelo prevê para 14:01
                    # No script de validação (UTR221_ML_validacao.py), este
                    # valor é 30 — aqui usamos 1 para o modelo do dashboard.

SETPOINT_ALTO  = 1.68  # Pressão em BAR que DESLIGA as bombas automaticamente
                       # Quando a pressão atinge 1.68 BAR, as bombas param
                       # para evitar sobrepressão na rede

SETPOINT_BAIXO = 1.55  # Pressão em BAR que LIGA as bombas automaticamente
                       # Quando a pressão cai a 1.55 BAR, as bombas ligam
                       # para manter a pressão adequada no sistema


# ═══════════════════════════════════════════════════════════════════
# ETAPA 1/5 — CARREGAMENTO DOS DADOS
# ═══════════════════════════════════════════════════════════════════
# Lemos os arquivos Excel com os dados históricos das duas UTRs.
# Usamos apenas as colunas necessárias para economizar memória.

# Cabeçalho visual no console
print("=" * 60)
print("  TREINAMENTO DO MODELO UTR-221")
print(f"  Horizonte: {HORIZONTE_MIN} minutos à frente")
print("=" * 60)

print("\n[1/5] Carregando dados dos arquivos Excel...")

# Carregar apenas as colunas necessárias da URT 220
# usecols = lista de colunas a ler (mais eficiente que ler o arquivo inteiro)
df220 = pd.read_excel(
    'urt220.xlsx',
    usecols=['E3TIMESTAMP',       # Data e hora da leitura
             'CMB_220_S2A_EST',   # Estado da Bomba 2A (0=desligada, 1=ligada)
             'CMB_220_S2B_EST',   # Estado da Bomba 2B (0=desligada, 1=ligada)
             'LIT_220_RA2_000']   # Nível do reservatório RA2 em metros
)

def parse_timestamp(serie):
    """
    Converte uma coluna de texto de timestamp para o formato datetime do Python.

    O SCADA exporta no formato: "01/02/26 00:00:27,954000000"
    Python espera:              "01/02/26 00:00:27.954000"

    Transformações necessárias:
      1. Trocar vírgula por ponto (separador decimal)
      2. Truncar para 6 dígitos decimais (microssegundos)
      3. Converter para datetime

    PARÂMETROS:
      serie: coluna (Series) do pandas com os timestamps como texto

    RETORNA:
      Series com valores do tipo datetime
    """
    return pd.to_datetime(
        # Passo 1: garantir que é texto
        serie.astype(str)
             # Passo 2: "27,954000000" → "27.954000000"
             .str.replace(',', '.', regex=False)
             # Passo 3: "27.954000000" → "27.954000" (máximo 6 dígitos após o ponto)
             .str.replace(r'(\.\d{6})\d+', r'\1', regex=True),
        # Especificar o formato exato para conversão mais rápida e correta
        format='%d/%m/%y %H:%M:%S.%f'
    )

# Aplicar a conversão de timestamp na URT 220
df220['E3TIMESTAMP'] = parse_timestamp(df220['E3TIMESTAMP'])

# Definir E3TIMESTAMP como índice e ordenar cronologicamente
# O índice datetime permite operações temporais como resample e shift
df220 = df220.set_index('E3TIMESTAMP').sort_index()

# Carregar dados da URT 221 (apenas a pressão, que é nossa variável alvo)
df221 = pd.read_excel('urt221.xlsx', usecols=['E3TIMESTAMP', 'PIT_221_S01_000'])
df221['E3TIMESTAMP'] = parse_timestamp(df221['E3TIMESTAMP'])
df221 = df221.set_index('E3TIMESTAMP').sort_index()

# Regularizar ambas para intervalos de 1 minuto
# resample('1min').mean() = calcular média de cada janela de 1 minuto
# dropna() = remover linhas onde TODOS os valores são NaN (apenas gaps reais)
df220 = df220.resample('1min').mean().dropna()
df221 = df221.resample('1min').mean().dropna()

# Combinar URT 220 e URT 221 num único DataFrame
# join(..., how='inner') = manter apenas os timestamps que existem em AMBAS as tabelas
# Isso garante que sempre temos os dados de entrada (URT220) E o alvo (URT221)
df = df220.join(df221, how='inner')

# Filtrar para usar apenas dados a partir de um ponto específico
# Esta data foi escolhida pois o sistema estava em condição estável após esse momento
df = df.loc['2026-02-28 17:30:38':]

# Renomear colunas para nomes mais legíveis e intuitivos
# Isso facilita a leitura do código e a interpretação dos resultados
df = df.rename(columns={
    'CMB_220_S2A_EST': 'BOMBA_1',        # Estado da Bomba 1 (ligada/desligada)
    'CMB_220_S2B_EST': 'BOMBA_2',        # Estado da Bomba 2 (ligada/desligada)
    'LIT_220_RA2_000': 'NIVEL-UTR-220',  # Nível do reservatório (metros)
    'PIT_221_S01_000': 'PRESSAO-UTR-221',# Pressão da UTR 221 (BAR) — ALVO DO MODELO
})

print(f"    Dados carregados: {len(df)} minutos  |  {df.index[0]} a {df.index[-1]}")


# ═══════════════════════════════════════════════════════════════════
# ETAPA 2/5 — ENGENHARIA DE FEATURES (CRIAÇÃO DAS VARIÁVEIS)
# ═══════════════════════════════════════════════════════════════════
# Feature Engineering é o processo de criar novas variáveis (features)
# a partir dos dados brutos para que o modelo possa aprender padrões
# mais complexos. É uma das etapas mais importantes em ML.
#
# POR QUÊ CRIAR FEATURES DERIVADAS?
#   O modelo precisa de contexto. A pressão atual não basta para prever
#   a futura — precisamos saber a tendência, o histórico recente,
#   o estado das bombas, a hora do dia, etc.

print("\n[2/5] Calculando features...")

# ── Variável Alvo (Target) ─────────────────────────────────────────
# O "target" é o que queremos prever — a pressão DAQUI A N MINUTOS.
# shift(-N) = deslocar os valores N posições para CIMA (para o futuro)
# Isso alinha: "no momento T, o target é o valor em T + N minutos"
df['target'] = df['PRESSAO-UTR-221'].shift(-HORIZONTE_MIN)

# ── Lags (Valores Passados) ────────────────────────────────────────
# Lags são os valores de minutos anteriores. São como "memória" do sistema.
# shift(1) = valor de 1 minuto atrás
# shift(5) = valor de 5 minutos atrás
# POR QUÊ? A pressão futura depende muito da pressão passada recente.
df['nivel_221_lag1']  = df['PRESSAO-UTR-221'].shift(1)   # Pressão 1 minuto atrás
df['nivel_221_lag5']  = df['PRESSAO-UTR-221'].shift(5)   # Pressão 5 minutos atrás
df['nivel_221_lag15'] = df['PRESSAO-UTR-221'].shift(15)  # Pressão 15 minutos atrás
df['nivel_220_lag1']  = df['NIVEL-UTR-220'].shift(1)     # Nível do reservatório 1 min atrás

# ── Tendências (Taxa de Variação) ──────────────────────────────────
# A tendência mostra se a pressão está subindo, descendo ou estável.
# Tendência = valor atual - valor anterior (positivo = subindo)
# POR QUÊ? Se a pressão está caindo rapidamente, é mais provável que
# continue caindo no próximo minuto.
df['tendencia_1min'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(2)
    # Variação no último minuto: diferença entre t-1 e t-2
df['tendencia_5min'] = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(6)
    # Variação nos últimos 5 minutos: diferença entre t-1 e t-6
df['tendencia_1h']   = df['PRESSAO-UTR-221'].shift(1) - df['PRESSAO-UTR-221'].shift(60)
    # Variação na última hora: diferença entre t-1 e t-60

# ── Médias Móveis (Suavização) ─────────────────────────────────────
# Médias móveis suavizam oscilações e mostram a tendência de médio prazo.
# rolling(window=N) = janela deslizante de N períodos
# .mean() = média dentro da janela
# POR QUÊ USAR shift(1) ANTES DO ROLLING?
#   Para evitar "data leakage" (vazamento de dados do futuro).
#   Se usássemos os valores atuais, o modelo "saberia" o futuro durante o treino,
#   o que é trapaça e resultaria em métricas falsamente boas.
df['media_movel_1h']  = df['PRESSAO-UTR-221'].shift(1).rolling(window=60).mean()
    # Média da pressão na última hora (60 minutos anteriores)
df['media_movel_15m'] = df['PRESSAO-UTR-221'].shift(1).rolling(window=15).mean()
    # Média da pressão nos últimos 15 minutos

# ── Features Temporais (Hora e Turno) ─────────────────────────────
# O comportamento da pressão pode variar ao longo do dia:
# - Madrugada: menor consumo → pressão mais estável
# - Dia: maior consumo → pressão mais volátil
df['hora']  = df.index.hour  # Hora do dia: 0 a 23

# Turno de trabalho (categorizado em 4 períodos):
# pd.cut: divide valores em intervalos (bins) e atribui rótulos (labels)
# bins=[-1, 5, 11, 17, 23] cria 4 intervalos:
#   0 = madrugada (00h-05h), 1 = manhã (06h-11h)
#   2 = tarde (12h-17h),     3 = noite (18h-23h)
df['turno'] = pd.cut(df.index.hour,
                     bins=[-1, 5, 11, 17, 23],
                     labels=[0, 1, 2, 3]).astype(int)

# ── Features Específicas da BOMBA_2 (Rodízio) ──────────────────────
# O sistema opera em rodízio: às vezes a Bomba 1 está ativa,
# às vezes a Bomba 2. Isso afeta diretamente a pressão produzida.

# Transição da Bomba 2: detecta quando a bomba liga (+1) ou desliga (-1)
# diff() = valor atual menos o anterior
# fillna(0) = preencher o primeiro valor (sem anterior) com 0 (sem transição)
df['transicao_B2'] = df['BOMBA_2'].diff().fillna(0)

# Tempo contínuo que a Bomba 2 está ligada (em minutos)
# Lógica: criar grupos de períodos consecutivos com mesmo estado
# (df['BOMBA_2'] != df['BOMBA_2'].shift()) → True onde há mudança de estado
# .cumsum() → conta as mudanças acumuladas = ID do grupo
grupo_b2 = (df['BOMBA_2'] != df['BOMBA_2'].shift()).cumsum()
df['tempo_ligada_B2'] = (
    df.groupby(grupo_b2)['BOMBA_2']
    # range(len(x)) = 0, 1, 2, 3... (minutos no grupo)
    .transform(lambda x: range(len(x)))
    # Multiplicar pelo estado da bomba: se 0 (desligada), o tempo fica 0
    * df['BOMBA_2']
)

# Indicador de qual bomba está operando (rodízio):
# 1 = Bomba 1 ligada, 2 = Bomba 2 ligada, 0 = ambas desligadas
# np.where = equivalente a IF...ELSE vetorizado (aplica em todo o array de uma vez)
df['rodizio']      = np.where(df['BOMBA_1'] == 1, 1,
                    np.where(df['BOMBA_2'] == 1, 2, 0))

# Estado da Bomba 2 no minuto anterior (lag 1)
df['BOMBA_2_lag1'] = df['BOMBA_2'].shift(1)

# Média da fração do tempo em que cada bomba ficou ligada nos últimos 15 min
# Valores entre 0 (desligada todo o tempo) e 1 (ligada todo o tempo)
df['B2_media_15m'] = df['BOMBA_2'].shift(1).rolling(window=15).mean()
df['B1_media_15m'] = df['BOMBA_1'].shift(1).rolling(window=15).mean()

# Remover linhas com NaN criados pelos shifts e rolling windows
# NaN aparece nos primeiros/últimos registros onde não há histórico suficiente
df = df.dropna()

# Criar df2: remover linhas onde target=0 ou pressão lag1=0
# Valores zero podem indicar falhas de sensor ou períodos sem operação
# que não devem influenciar o treinamento do modelo
df2 = df[(df['target'] != 0) & (df['nivel_221_lag1'] != 0)].copy()

print(f"    Total de linhas válidas para treino: {len(df2)}")


# ═══════════════════════════════════════════════════════════════════
# ETAPA 3/5 — DIVISÃO TEMPORAL E TREINAMENTO DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Antes de treinar, precisamos dividir os dados em:
#   - Conjunto de TREINO: dados usados para o modelo aprender (80%)
#   - Conjunto de TESTE:  dados usados para avaliar o modelo (20%)
#
# IMPORTANTE: Usamos divisão TEMPORAL (não aleatória)!
# Por quê? Em séries temporais, o futuro não pode treinar o passado.
# Se misturássemos dados de dezembro para prever novembro, o modelo
# "trapacearia" usando informações do futuro.

print("\n[3/5] Treinando o modelo...")

# Lista completa de features (variáveis de entrada) do modelo
# A ordem importa: precisa ser a mesma durante treino e previsão
FEATURES = [
    'BOMBA_1',          # Estado atual da Bomba 1 (0 ou 1)
    'BOMBA_2',          # Estado atual da Bomba 2 (0 ou 1)
    'BOMBA_2_lag1',     # Estado da Bomba 2 no minuto anterior
    'B1_media_15m',     # Fração do tempo ligada da Bomba 1 (15 min)
    'B2_media_15m',     # Fração do tempo ligada da Bomba 2 (15 min)
    'nivel_220_lag1',   # Nível do reservatório 1 min atrás
    'nivel_221_lag1',   # Pressão UTR-221 1 minuto atrás
    'nivel_221_lag5',   # Pressão UTR-221 5 minutos atrás
    'nivel_221_lag15',  # Pressão UTR-221 15 minutos atrás
    'hora',             # Hora do dia (0-23)
    'turno',            # Turno (0=madrugada, 1=manhã, 2=tarde, 3=noite)
    'tendencia_1min',   # Variação de pressão no último minuto
    'tendencia_5min',   # Variação de pressão nos últimos 5 minutos
    'tendencia_1h',     # Variação de pressão na última hora
    'media_movel_1h',   # Média da pressão na última hora
    'media_movel_15m',  # Média da pressão nos últimos 15 minutos
    'transicao_B2',     # Mudança de estado da Bomba 2 (+1 ligou, -1 desligou)
    'tempo_ligada_B2',  # Minutos consecutivos que a Bomba 2 está ligada
    'rodizio',          # Qual bomba está operando (1, 2 ou 0)
]

# Extrair as features (X) e o alvo (y) do DataFrame
X2 = df2[FEATURES]  # Matriz de features: 19 colunas, uma linha por minuto
y2 = df2['target']  # Vetor de alvos: pressão daqui a HORIZONTE_MIN minutos

# Divisão temporal 80/20
# int(len(df2) * 0.8) = índice que corresponde a 80% dos dados
split_point = int(len(df2) * 0.8)

# iloc[:split_point]  = todas as linhas DO INÍCIO até split_point (treino)
# iloc[split_point:]  = todas as linhas DE split_point ao FIM (teste)
X_train, X_test = X2.iloc[:split_point], X2.iloc[split_point:]
y_train, y_test = y2.iloc[:split_point], y2.iloc[split_point:]

# Criar e treinar o modelo Random Forest
# n_estimators=200: criar 200 árvores de decisão diferentes
#   Mais árvores = mais robusto, mas mais lento para treinar
# random_state=42: semente aleatória para reprodutibilidade
#   (mesma semente = mesmo modelo em qualquer máquina)
# n_jobs=-1: usar todos os núcleos do processador em paralelo
#   (treina as 200 árvores simultaneamente, muito mais rápido)
modelo = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# .fit(X_train, y_train) = fase de aprendizado
# O modelo analisa os dados de treino e ajusta seus parâmetros internos
modelo.fit(X_train, y_train)

print("    Modelo treinado com sucesso.")


# ═══════════════════════════════════════════════════════════════════
# ETAPA 4/5 — MÉTRICAS DE AVALIAÇÃO DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Avaliar o modelo no conjunto de TESTE (dados que ele nunca viu).
# Isso nos diz o quão bem o modelo generaliza para novos dados.
#
# MÉTRICAS UTILIZADAS:
#   MAE  (Mean Absolute Error): erro médio em BAR.
#          Ex: MAE=0.02 → em média erra 0.02 BAR
#   RMSE (Root Mean Squared Error): similar ao MAE, mas penaliza mais
#          erros grandes. Útil para detectar previsões muito erradas.
#   R²   (R-Squared): coeficiente de determinação.
#          1.0 = perfeito, 0.0 = não melhor que a média, <0 = péssimo
#   MAPE (Mean Absolute Percentage Error): erro em porcentagem.
#          Ex: MAPE=1.5% → erra em média 1.5% do valor real
#   Acurácia = 100% - MAPE

print("\n[4/5] Calculando métricas...")

# Fazer previsões no conjunto de teste (dados não vistos no treino)
# .predict(X_test) = aplicar o modelo aprendido aos dados de teste
previsoes = modelo.predict(X_test)

# Calcular cada métrica de avaliação
mae      = mean_absolute_error(y_test, previsoes)
    # Média dos |erros absolutos|: mean(|real - previsto|)

rmse     = np.sqrt(mean_squared_error(y_test, previsoes))
    # Raiz do erro quadrático médio: sqrt(mean((real - previsto)²))

r2       = r2_score(y_test, previsoes)
    # Coeficiente de determinação: 1 - (SS_res / SS_tot)

# Para o MAPE, evitar divisão por zero (onde real = 0)
mask_nz  = y_test != 0  # Máscara: True apenas onde o valor real não é zero

mape     = np.mean(np.abs((y_test[mask_nz] - previsoes[mask_nz]) / y_test[mask_nz])) * 100
    # Erro percentual médio absoluto

acuracia = 100 - mape  # Acurácia simples = 100% - MAPE

# Organizar as métricas num dicionário (será salvo nos metadados)
metricas = {
    'MAE':      round(mae, 4),       # Arredondar para 4 casas decimais
    'RMSE':     round(rmse, 4),
    'R2':       round(r2, 4),
    'MAPE':     round(mape, 2),      # Porcentagem: 2 casas
    'Acuracia': round(acuracia, 2),
}

# Exibir as métricas no console de forma organizada
print("=" * 45)
print(f"  Horizonte : {HORIZONTE_MIN} minutos a frente")
print(f"  MAE       : {mae:.4f} BAR")
print(f"  RMSE      : {rmse:.4f} BAR")
print(f"  R²        : {r2:.4f}")
print(f"  MAPE      : {mape:.2f}%")
print(f"  Acuracia  : {acuracia:.2f}%")
print("=" * 45)

# Importância das features: quanto cada variável contribuiu para o modelo
# feature_importances_ = atributo do Random Forest após o treino
# Valores somam 1.0 (100%) — maior valor = mais importante para a previsão
print("\nImportancia das features:")
importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)

for feat, imp in importancias.items():
    # Criar uma barra visual proporcional à importância
    # int(imp * 40) = converter para número de caracteres '#' (escala de 0 a 40)
    barra = "#" * int(imp * 40)
    # :<20s = alinhar texto à esquerda em 20 caracteres
    print(f"  {feat:<20s} {imp:.4f}  {barra}")


# ═══════════════════════════════════════════════════════════════════
# ETAPA 5/5 — SALVAR OS ARTEFATOS DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Salvamos 3 arquivos (.pkl = pickle) que o dashboard vai carregar:
#
#   1. modelo_utr221.pkl:    O modelo treinado — contém as 200 árvores
#                            de decisão prontas para fazer previsões.
#
#   2. historico_utr221.pkl: Os dados históricos usados para inicializar
#                            a simulação no dashboard. O dashboard precisa
#                            dos últimos 60 minutos de dados reais para
#                            calcular os lags e médias móveis.
#
#   3. meta_utr221.pkl:      Metadados — informações sobre o modelo:
#                            quais features usar, métricas de qualidade,
#                            período dos dados, setpoints das bombas.

print("\n[5/5] Salvando artefatos...")

# joblib.dump(objeto, arquivo) = serializar e salvar no disco
# .pkl é a extensão convencional para arquivos pickle
joblib.dump(modelo, 'modelo_utr221.pkl')
print("    OK modelo_utr221.pkl")

# Histórico completo com as colunas usadas no dashboard para:
# - inicializar o buffer de pressão (lags e médias móveis)
# - conhecer o estado inicial das bombas
# - calcular o tempo que a bomba 2 já está ligada
historico = df[[
    'PRESSAO-UTR-221',   # Série histórica de pressão (alimenta os lags)
    'NIVEL-UTR-220',     # Nível do reservatório (feature do modelo)
    'BOMBA_1',           # Estado histórico da Bomba 1
    'BOMBA_2',           # Estado histórico da Bomba 2
    'rodizio',           # Indicador de rodízio histórico
    'tempo_ligada_B2',   # Tempo contínuo que a Bomba 2 estava ligada
]].copy()

joblib.dump(historico, 'historico_utr221.pkl')
print("    OK historico_utr221.pkl")

# Metadados: todas as informações necessárias para reproduzir e usar o modelo
meta = {
    'features':       FEATURES,         # Lista das features na ordem correta
    'metricas':       metricas,          # Métricas de qualidade do modelo
    'horizonte_min':  HORIZONTE_MIN,     # Quantos minutos à frente o modelo prevê
    'setpoint_alto':  SETPOINT_ALTO,     # Pressão de desligamento das bombas
    'setpoint_baixo': SETPOINT_BAIXO,    # Pressão de acionamento das bombas
    'ultimo_ts':      str(df.index[-1]), # Último timestamp dos dados de treino
    'primeiro_ts':    str(df.index[0]),  # Primeiro timestamp dos dados de treino
    'total_minutos':  len(df),           # Total de minutos no dataset
}

joblib.dump(meta, 'meta_utr221.pkl')
print("    OK meta_utr221.pkl")

# Mensagem final com instruções de uso
print("\n" + "=" * 60)
print("  TREINAMENTO CONCLUIDO")
print("  Execute o dashboard com:")
print("    python -m streamlit run dashboard_utr221.py")
print("=" * 60)
