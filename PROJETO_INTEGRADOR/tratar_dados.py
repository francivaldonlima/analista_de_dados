# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║                        tratar_dados.py                          ║
║           Tratamento e Imputação de Dados — URT220 e URT221     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OBJETIVO:                                                       ║
║    Preparar os dados brutos exportados do sistema SCADA          ║
║    (Supervisory Control and Data Acquisition) para que possam    ║
║    ser usados no treinamento de modelos de Machine Learning.     ║
║                                                                  ║
║  O QUE ESTE SCRIPT FAZ — PASSO A PASSO:                         ║
║    1. Carrega os arquivos Excel com os dados históricos          ║
║    2. Converte a coluna de data/hora para o formato correto      ║
║    3. Marca como inválidos (NaN) os valores com QUALITY = 0,     ║
║       pois o SCADA sinaliza que esses dados não são confiáveis   ║
║    4. Regulariza a série temporal para intervalos de 1 minuto    ║
║       (preenche os buracos onde não há leitura)                  ║
║    5. Imputa (preenche) os valores NaN:                          ║
║       - Sensores analógicos (pressão, nível, vazão):             ║
║         interpolação linear — estima o valor entre dois pontos   ║
║       - Status/bombas (valores binários 0 ou 1):                 ║
║         forward-fill — mantém o último valor conhecido           ║
║    6. Gera relatório de diagnóstico comparando antes x depois    ║
║    7. Exporta os arquivos tratados prontos para o modelo         ║
║                                                                  ║
║  ARQUIVOS DE ENTRADA:  urt220.xlsx  |  urt221.xlsx               ║
║  ARQUIVOS DE SAÍDA:    urt220_tratado.xlsx  |  urt221_tratado.xlsx ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DE ENCODING (CODIFICAÇÃO DE CARACTERES)
# ═══════════════════════════════════════════════════════════════════
# Esta linha garante que o terminal exiba corretamente caracteres
# especiais do português (ã, ç, é, etc.) no Windows.
# sys = módulo do sistema operacional do Python
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════
# IMPORTAÇÃO DAS BIBLIOTECAS
# ═══════════════════════════════════════════════════════════════════
# As bibliotecas são "caixas de ferramentas" que já foram criadas
# por outros programadores e que podemos reutilizar.

import pandas as pd          # Pandas: manipulação de tabelas (DataFrames)
                              # É a principal ferramenta de análise de dados em Python
import numpy as np           # NumPy: operações matemáticas e arrays numéricos
                              # NaN (Not a Number) = valor ausente/inválido
import matplotlib.pyplot as plt   # Matplotlib: criação de gráficos e visualizações
import matplotlib.dates as mdates # Formatação de datas nos eixos dos gráficos
from pathlib import Path          # Pathlib: manipulação de caminhos de arquivos


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONFIGURAÇÕES GLOBAIS DO SCRIPT
# ═══════════════════════════════════════════════════════════════════
# Aqui definimos as "constantes" — valores que não mudam durante
# a execução, mas que podem ser facilmente ajustados pelo analista.

# --- Arquivos de entrada e saída ---
ARQUIVO_220 = 'urt220.xlsx'         # Arquivo Excel com dados da URT 220
ARQUIVO_221 = 'urt221.xlsx'         # Arquivo Excel com dados da URT 221
SAIDA_220   = 'urt220_tratado.xlsx' # Nome do arquivo tratado que será gerado para URT220
SAIDA_221   = 'urt221_tratado.xlsx' # Nome do arquivo tratado que será gerado para URT221

# --- Parâmetros de tratamento temporal ---
FREQ_RESAMPLE = '1min'  # Frequência de reamostragem: 1 leitura por minuto
                        # O SCADA pode registrar em intervalos irregulares;
                        # padronizamos para 1 minuto para ter série contínua.

MAX_GAP_INTERP = 30     # Limite máximo de minutos para interpolar valores ausentes.
                        # Se um sensor ficou sem leitura por mais de 30 minutos,
                        # NÃO tentamos adivinhar o valor — deixamos como NaN.
                        # Isso evita estimativas muito imprecisas em falhas longas.


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 2 — DEFINIÇÃO DAS COLUNAS POR TIPO
# ═══════════════════════════════════════════════════════════════════
# Separamos as colunas em dois grupos porque cada tipo precisa de
# um tratamento diferente para valores ausentes (NaN):
#
#   - ANALÓGICAS: medidas contínuas (pressão em BAR, nível em metros,
#     corrente em Ampères). Usamos interpolação linear.
#
#   - STATUS/DISCRETAS: valores binários (0 = desligado, 1 = ligado).
#     Usamos forward-fill (mantém o último estado conhecido).

# Colunas analógicas da URT 220 — sensores de medição contínua:
ANALOGICAS_220 = [
    'PIT_220_S01_000',  # PIT = Pressure Indicator Transmitter (pressão na saída S01)
    'LIT_220_RA2_000',  # LIT = Level Indicator Transmitter (nível no reservatório RA2)
    'LIT_220_RE1_000',  # LIT = Level no reservatório RE1
    'CMB_220_S1A_AMP',  # CMB = Bomba, S1A = bomba 1A, AMP = corrente elétrica (Ampères)
    'CMB_220_S1B_AMP',  # Corrente elétrica da bomba 1B
    'CMB_220_S2A_AMP',  # Corrente elétrica da bomba 2A
    'CMB_220_S2B_AMP',  # Corrente elétrica da bomba 2B
]

# Colunas analógicas da URT 221:
ANALOGICAS_221 = [
    'LIT_221_RE4_000',  # Nível do reservatório RE4 da URT 221
    'PIT_221_S01_000',  # Pressão na saída S01 da URT 221 — VARIÁVEL ALVO DO MODELO
    'FIT_221_S01_000',  # FIT = Flow Indicator Transmitter (vazão instantânea)
    'FIT_221_S01_TLL',  # Vazão acumulada (totalizador)
]

# Colunas de status/discretas da URT 220 — valores 0 ou 1:
STATUS_220 = [
    # Status gerais da UTR 220 (a própria unidade de telemetria):
    'UTR_220_STS_PRT',  # Status de proteção da UTR (0=normal, 1=alarme)
    'UTR_220_STS_PWR',  # Status de energia (0=sem energia, 1=com energia)
    'UTR_220_STS_UPS',  # Status do nobreak UPS (0=sem bateria, 1=ok)
    # Status da Bomba S1A (grupo 1, bomba A):
    'CMB_220_S1A_LRT',  # Ligada em Remoto (1=operação remota ativa)
    'CMB_220_S1A_AMN',  # Alarme de Manutenção
    'CMB_220_S1A_EST',  # Estado (0=desligada, 1=ligada)
    'CMB_220_S1A_DEF',  # Defeito elétrico detectado
    'CMB_220_S1A_TMO',  # Timeout (tempo excedido)
    'CMB_220_S1A_EXT',  # Status externo
    'CMB_220_S1A_ERR',  # Erro genérico
    'CMB_220_S1A_STS',  # Status geral da bomba
    # Status da Bomba S1B (grupo 1, bomba B):
    'CMB_220_S1B_LRT', 'CMB_220_S1B_AMN', 'CMB_220_S1B_EST',
    'CMB_220_S1B_DEF', 'CMB_220_S1B_TMO', 'CMB_220_S1B_EXT',
    'CMB_220_S1B_ERR', 'CMB_220_S1B_STS',
    # Status da Bomba S2A (grupo 2, bomba A — bomba principal do modelo):
    'CMB_220_S2A_LRT', 'CMB_220_S2A_AMN', 'CMB_220_S2A_EST',
    'CMB_220_S2A_DEF', 'CMB_220_S2A_TMO', 'CMB_220_S2A_EXT',
    'CMB_220_S2A_ERR', 'CMB_220_S2A_STS',
    # Status da Bomba S2B (grupo 2, bomba B — bomba secundária do modelo):
    'CMB_220_S2B_LRT', 'CMB_220_S2B_AMN', 'CMB_220_S2B_EST',
    'CMB_220_S2B_DEF', 'CMB_220_S2B_TMO', 'CMB_220_S2B_EXT',
    'CMB_220_S2B_ERR', 'CMB_220_S2B_STS',
]


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 3 — DEFINIÇÃO DAS FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════
# Uma "função" é um bloco de código reutilizável. Ao invés de
# repetir o mesmo código várias vezes, criamos funções que podem
# ser chamadas com diferentes dados.

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBJETIVO: Converte a coluna E3TIMESTAMP para o formato datetime do Python
              e define essa coluna como o índice (referência) do DataFrame.

    POR QUÊ É NECESSÁRIO?
      O SCADA exporta o timestamp como texto no formato:
        "01/02/26 00:00:27,954000000"
      Precisamos converter esse texto para datetime para:
        1. Ordenar os dados cronologicamente
        2. Fazer operações temporais (resample, diff, etc.)

    PARÂMETROS:
      df (pd.DataFrame): tabela com a coluna E3TIMESTAMP como texto

    RETORNA:
      pd.DataFrame: mesma tabela com E3TIMESTAMP como índice datetime
    """
    # Passo 1: Converter a coluna para string (texto) para garantir o formato correto
    # e aplicar substituições de texto:
    ts = (df['E3TIMESTAMP'].astype(str)
          # Substituir vírgula por ponto: "27,954000000" → "27.954000000"
          # O Python usa ponto como separador decimal, não vírgula
          .str.replace(',', '.', regex=False)
          # Truncar para 6 dígitos após o ponto: "27.954000000" → "27.954000"
          # datetime do Python aceita no máximo microssegundos (6 dígitos)
          .str.replace(r'(\.\d{6})\d+', r'\1', regex=True))

    # Passo 2: Criar uma cópia do DataFrame para não alterar o original
    df = df.copy()

    # Passo 3: Converter o texto processado para datetime usando o formato especificado
    # %d/%m/%y = dia/mês/ano com 2 dígitos
    # %H:%M:%S.%f = hora:minuto:segundo.microsegundo
    df['E3TIMESTAMP'] = pd.to_datetime(ts, format='%d/%m/%y %H:%M:%S.%f')

    # Passo 4: Definir E3TIMESTAMP como índice e ordenar do mais antigo ao mais recente
    return df.set_index('E3TIMESTAMP').sort_index()


def mascarar_quality_zero(df: pd.DataFrame, colunas_valor: list) -> pd.DataFrame:
    """
    OBJETIVO: Invalidar (transformar em NaN) os valores de sensores
              que o SCADA marcou como ruins (QUALITY = 0).

    O QUE É QUALITY?
      O SCADA atribui um código de qualidade a cada leitura:
        - QUALITY = 192: dado válido e confiável
        - QUALITY = 0:   dado inválido (sensor com falha, comunicação perdida, etc.)
      Para cada sensor (ex: PIT_221_S01_000), existe uma coluna correspondente
      (ex: PIT_221_S01_000_QUALITY) indicando a qualidade daquela leitura.

    PARÂMETROS:
      df (pd.DataFrame): tabela com os dados brutos
      colunas_valor (list): lista de nomes das colunas de sensores a verificar

    RETORNA:
      pd.DataFrame: tabela com valores ruins substituídos por NaN
    """
    # Criar cópia para não alterar o DataFrame original
    df = df.copy()

    # Iterar sobre cada coluna de sensor
    for col in colunas_valor:
        # Montar o nome da coluna de qualidade correspondente
        # Ex: 'PIT_221_S01_000' → 'PIT_221_S01_000_QUALITY'
        qual_col = col + '_QUALITY'

        # Verificar se a coluna de qualidade existe no DataFrame
        if qual_col in df.columns:
            # Criar uma máscara booleana (True onde QUALITY = 0)
            # Uma máscara é um array de True/False com o mesmo tamanho do DataFrame
            mascara_ruim = df[qual_col] == 0

            # Onde a máscara é True (dado ruim), substituir o valor por NaN
            # np.nan é o valor especial do NumPy para "não é um número" / dado ausente
            df.loc[mascara_ruim, col] = np.nan

    return df


def diagnostico_nulos(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    OBJETIVO: Gerar um relatório mostrando quantos valores ausentes (NaN)
              existem em cada coluna do DataFrame.

    POR QUÊ É IMPORTANTE?
      Antes de imputar, precisamos saber ONDE estão os dados faltantes
      e QUANTOS são. Isso ajuda a avaliar a qualidade dos dados.

    PARÂMETROS:
      df (pd.DataFrame): tabela a ser analisada
      label (str): nome/descrição para identificar o relatório no console

    RETORNA:
      pd.DataFrame: tabela apenas com colunas que têm NaN, ordenada por quantidade
    """
    # Contar o total de linhas do DataFrame
    total = len(df)

    # Contar quantos NaN existem em cada coluna
    # isnull() retorna True onde há NaN; sum() conta os True (que valem 1)
    nulos = df.isnull().sum()

    # Calcular o percentual de NaN em cada coluna
    # round(2) = arredondar para 2 casas decimais
    pct = (nulos / total * 100).round(2)

    # Criar um novo DataFrame com as colunas 'NaN' e '%'
    result = pd.DataFrame({'NaN': nulos, '%': pct})

    # Filtrar apenas colunas que TÊM valores ausentes, ordenar do mais grave ao menor
    result = result[result['NaN'] > 0].sort_values('NaN', ascending=False)

    # Exibir o relatório no console
    print(f"\n{'='*55}")
    print(f"  {label} — Dados ausentes após mascaramento QUALITY=0")
    print(f"{'='*55}")
    if result.empty:
        print("  Nenhum valor ausente.")
    else:
        print(result.to_string())

    return result


def imputar(df: pd.DataFrame,
            analogicas: list,
            status: list,
            max_gap: int = MAX_GAP_INTERP) -> pd.DataFrame:
    """
    OBJETIVO: Preencher os valores ausentes (NaN) com estimativas razoáveis.

    O QUE É IMPUTAÇÃO?
      Imputar significa estimar e preencher valores que estão faltando.
      Não podemos simplesmente deletar as linhas com NaN, pois precisamos
      de uma série temporal contínua para o modelo de Machine Learning.

    ESTRATÉGIAS UTILIZADAS:
      1. Interpolação linear (para sensores analógicos):
         - Método: desenha uma linha reta entre o último valor válido e
           o próximo valor válido, e estima os pontos intermediários.
         - Limite: máximo de MAX_GAP_INTERP minutos. Se a lacuna for maior,
           os pontos além do limite ficam como NaN (sem estimativa).
         - Exemplo: pressão de 1.5 BAR → 60 min sem dados → pressão 1.7 BAR
           A interpolação estimaria: 1.52, 1.54, 1.56, ... 1.68 BAR

      2. Forward-fill (para colunas de status binário):
         - Método: repete o último valor conhecido até encontrar o próximo.
         - Lógica: se uma bomba estava LIGADA (1) e perdemos o sinal por
           alguns minutos, é razoável assumir que ela ainda estava LIGADA.
         - bfill() = backward-fill, usado no início da série (sem valor anterior)

    PARÂMETROS:
      df (pd.DataFrame): tabela após resample (com NaN nos gaps)
      analogicas (list): colunas de sensores contínuos
      status (list): colunas de status binários (0/1)
      max_gap (int): máximo de períodos para interpolar (default = 30 min)

    RETORNA:
      pd.DataFrame: tabela com NaN preenchidos
    """
    # Criar cópia para não alterar o original
    df = df.copy()

    # ── Passo 1: Interpolação para sensores analógicos ────────────────
    for col in analogicas:
        # Verificar se a coluna existe E se tem algum NaN (otimização)
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].interpolate(
                method='time',          # Considera o tempo real entre pontos (não apenas índice)
                limit=max_gap,          # Máximo de pontos consecutivos a preencher
                limit_direction='forward'  # Preenche apenas para frente (do passado ao futuro)
            )

    # ── Passo 2: Forward-fill para colunas de status ──────────────────
    for col in status:
        if col in df.columns and df[col].isnull().any():
            # ffill() = forward-fill: propaga o último valor não-NaN para frente
            # bfill() = backward-fill: usado para o início onde não há valor anterior
            df[col] = df[col].ffill().bfill()

    # ── Passo 3: Preencher colunas QUALITY com valor padrão ──────────
    # Identificar todas as colunas que terminam com '_QUALITY'
    qual_cols = [c for c in df.columns if c.endswith('_QUALITY')]
    for qc in qual_cols:
        # Onde a QUALITY está ausente (era um gap preenchido), definir 192
        # 192 é o código padrão do SCADA para "dado válido"
        # Isso indica que o dado foi imputado mas é considerado aceitável
        df[qc] = df[qc].fillna(192)

    return df


def lacunas_temporais(df: pd.DataFrame, label: str, threshold_min: int = 2):
    """
    OBJETIVO: Identificar e reportar as lacunas (buracos) na série temporal.

    O QUE É UMA LACUNA TEMPORAL?
      Quando o SCADA ou a comunicação falha, pode não haver registros por
      vários minutos ou horas. Isso cria "buracos" na série temporal.
      É importante saber onde estão e quão grandes são, pois:
        - Lacunas pequenas (< 30 min): interpolamos com segurança
        - Lacunas grandes (> 30 min): podem indicar falhas graves

    PARÂMETROS:
      df (pd.DataFrame): tabela com índice datetime
      label (str): nome para identificação no relatório
      threshold_min (int): mínimo de minutos para considerar uma lacuna (default = 2 min)
    """
    # Calcular a diferença de tempo entre linhas consecutivas
    # diff() calcula o valor atual menos o anterior (resultado em Timedelta)
    diffs = df.index.to_series().diff()

    # Filtrar apenas os gaps maiores que o threshold
    # pd.Timedelta(minutes=threshold_min) cria um objeto de duração de tempo
    gaps = diffs[diffs > pd.Timedelta(minutes=threshold_min)]

    # Exibir relatório
    print(f"\n{'='*55}")
    print(f"  {label} — Lacunas > {threshold_min} min: {len(gaps)}")
    print(f"{'='*55}")

    if not gaps.empty:
        # Para cada lacuna encontrada, mostrar o timestamp e a duração
        for ts, gap in gaps.items():
            print(f"  {ts}  →  {gap}")


def regularizar(df: pd.DataFrame,
                freq: str = FREQ_RESAMPLE,
                cols_binarias: list = None) -> pd.DataFrame:
    """
    OBJETIVO: Criar uma série temporal com intervalos regulares de 1 minuto.

    O PROBLEMA DA IRREGULARIDADE:
      O SCADA registra dados em intervalos irregulares — às vezes a cada
      30 segundos, às vezes com lacunas de vários minutos. Para análise
      e modelos de ML, precisamos de dados em intervalos fixos.

    COMO FUNCIONA O RESAMPLE?
      É como uma "grade" de tempo: criamos uma linha para cada minuto,
      e para cada minuto calculamos a média dos valores registrados
      naquele intervalo. Se não houve registro num dado minuto, o valor
      fica como NaN (será preenchido na imputação posterior).

    TRATAMENTOS ESPECIAIS:
      - Colunas binárias (0 ou 1): ao calcular a média de valores 0/1,
        podemos obter 0.5 (se houve uma transição no minuto). Arredondamos
        para manter apenas 0 ou 1.
      - Colunas QUALITY: usamos o MÍNIMO, pois se qualquer leitura naquele
        minuto foi ruim (QUALITY=0), o minuto todo deve ser marcado como ruim.

    PARÂMETROS:
      df (pd.DataFrame): tabela com índice datetime irregular
      freq (str): frequência desejada ('1min', '5min', etc.)
      cols_binarias (list): colunas que devem permanecer apenas 0 ou 1

    RETORNA:
      pd.DataFrame: tabela com índice temporal regular (uma linha por minuto)
    """
    # Tratar argumento mutável padrão (boa prática em Python)
    if cols_binarias is None:
        cols_binarias = []

    # Separar colunas de qualidade das colunas de valor
    qual_cols  = [c for c in df.columns if c.endswith('_QUALITY')]
    valor_cols = [c for c in df.columns if c not in qual_cols]

    # Resample das colunas de valor: agregar por média no período de 1 minuto
    # resample('1min') = agrupa as leituras em janelas de 1 minuto
    # .mean() = calcula a média dentro de cada janela
    df_resamp = df[valor_cols].resample(freq).mean()

    # Arredondar colunas binárias para eliminar valores intermediários (ex: 0.5)
    # round() arredonda para o inteiro mais próximo: 0.5 → 1, 0.4 → 0
    # astype('Int64') = tipo inteiro nullable (suporta NaN, diferente de int)
    for col in cols_binarias:
        if col in df_resamp.columns:
            df_resamp[col] = df_resamp[col].round().astype('Int64')

    # Resample das colunas QUALITY usando o MÍNIMO
    # Se num minuto houve uma leitura com QUALITY=0, o mínimo será 0
    # Isso "contamina" o minuto inteiro se qualquer dado foi ruim
    for qc in qual_cols:
        df_resamp[qc] = df[qc].resample(freq).min()

    return df_resamp


def plot_comparativo(df_orig: pd.DataFrame,
                     df_trat: pd.DataFrame,
                     col: str,
                     label: str,
                     arquivo_saida: str):
    """
    OBJETIVO: Criar um gráfico comparando a série ANTES e DEPOIS do tratamento.

    POR QUÊ VISUALIZAR?
      É fundamental verificar visualmente se o tratamento fez sentido.
      O gráfico nos permite ver se as interpolações criaram valores plausíveis
      ou se há alguma distorção nos dados.

    PARÂMETROS:
      df_orig (pd.DataFrame): dados originais (com NaN nos gaps)
      df_trat (pd.DataFrame): dados após imputação
      col (str): nome da coluna a plotar
      label (str): título do gráfico
      arquivo_saida (str): nome do arquivo PNG a salvar
    """
    # Criar figura com 2 subplots empilhados verticalmente (nrows=2)
    # sharex=True = ambos os gráficos compartilham o mesmo eixo X (tempo)
    # figsize=(16, 8) = largura 16 polegadas, altura 8 polegadas
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # ── Gráfico superior: dados ORIGINAIS ─────────────────────────────
    axes[0].plot(df_orig.index, df_orig[col],
                 color='steelblue',  # cor azul-aço
                 linewidth=0.8,      # linha fina para muitos pontos
                 label='Original')
    axes[0].set_title(f'{label} — Original', fontsize=11)
    axes[0].set_ylabel(col)          # rótulo do eixo Y = nome da coluna
    axes[0].legend()                 # mostrar legenda
    axes[0].grid(True, alpha=0.3)    # grade suave (alpha=0.3 = 30% opacidade)

    # ── Gráfico inferior: dados TRATADOS ──────────────────────────────
    axes[1].plot(df_trat.index, df_trat[col],
                 color='darkorange',  # cor laranja para diferenciar
                 linewidth=0.8,
                 label='Tratado')
    axes[1].set_title(f'{label} — Após Imputação', fontsize=11)
    axes[1].set_ylabel(col)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Formatar automaticamente os rótulos do eixo X (datas) para não sobrepor
    fig.autofmt_xdate()

    # Ajustar o espaçamento automático entre os subplots
    plt.tight_layout()

    # Salvar como PNG com resolução de 150 DPI (qualidade para relatório)
    # bbox_inches='tight' = não cortar bordas do gráfico
    plt.savefig(arquivo_saida, dpi=150, bbox_inches='tight')

    # Fechar a figura para liberar memória (importante em loops)
    plt.close()

    print(f"  Gráfico salvo: {arquivo_saida}")


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 4 — PIPELINE PRINCIPAL DE EXECUÇÃO
# ═══════════════════════════════════════════════════════════════════
# Aqui chamamos todas as funções na ordem correta para processar
# os dados. O pipeline é a sequência de operações que transforma
# os dados brutos em dados tratados prontos para análise.

# Cabeçalho no console para indicar o início do processamento
print("\n" + "="*55)
print("  TRATAMENTO DE DADOS — URT220 e URT221")
print("="*55)


# ═══════════════════════════════════════════════════════════════════
# BLOCO A — PROCESSAMENTO DA URT 220
# ═══════════════════════════════════════════════════════════════════
# A URT 220 contém dados de bombas e reservatórios que serão usados
# como variáveis preditoras (entrada) no modelo de ML.

print("\n>>> Carregando urt220.xlsx ...")

# pd.read_excel: ler arquivo Excel e criar um DataFrame (tabela)
# Cada linha do Excel vira uma linha no DataFrame
df220_raw = pd.read_excel(ARQUIVO_220)

# Converter timestamp e definir como índice
df220 = parse_timestamp(df220_raw)

# Informações de diagnóstico: quantas linhas e qual o período
print(f"    Linhas carregadas: {len(df220):,}")   # :, = formatar número com separador de milhar
print(f"    Período: {df220.index[0]}  →  {df220.index[-1]}")

# Verificar e reportar lacunas temporais antes de qualquer tratamento
lacunas_temporais(df220, 'URT220')

# Criar lista com TODAS as colunas de valor da URT 220
# (analógicas + status = tudo que tem coluna QUALITY)
todas_220 = ANALOGICAS_220 + STATUS_220  # juntar as duas listas

# Mascarar valores com QUALITY=0 (dados inválidos do SCADA)
df220_masked = mascarar_quality_zero(df220, todas_220)

# Diagnóstico: ver quantos NaN temos ANTES da imputação
diag_220_antes = diagnostico_nulos(df220_masked, 'URT220 [antes da imputação]')

# Definir quais colunas são binárias (para o resample arredondar para 0 ou 1)
BINARIAS_220 = STATUS_220  # todas as colunas de status são binárias

# Regularizar para série temporal de 1 minuto
# Após o resample, os gaps viram NaN — serão tratados na imputação
df220_reg = regularizar(df220_masked, cols_binarias=BINARIAS_220)
print(f"\n    Linhas após resample 1min: {len(df220_reg):,}")

# Imputar valores ausentes usando as estratégias definidas
# - Analógicas: interpolação linear com limite de 30 min
# - Status: forward-fill + backward-fill
df220_trat = imputar(df220_reg, ANALOGICAS_220, STATUS_220)

# Diagnóstico: ver quantos NaN restaram APÓS a imputação
diag_220_depois = diagnostico_nulos(df220_trat, 'URT220 [após imputação]')


# ═══════════════════════════════════════════════════════════════════
# BLOCO B — PROCESSAMENTO DA URT 221
# ═══════════════════════════════════════════════════════════════════
# A URT 221 contém a pressão que é a VARIÁVEL ALVO (target) do modelo.
# É o que queremos prever durante falhas de comunicação.

print("\n>>> Carregando urt221.xlsx ...")
df221_raw = pd.read_excel(ARQUIVO_221)
df221 = parse_timestamp(df221_raw)
print(f"    Linhas carregadas: {len(df221):,}")
print(f"    Período: {df221.index[0]}  →  {df221.index[-1]}")

lacunas_temporais(df221, 'URT221')

# A URT 221 só tem colunas analógicas (sem status de bombas)
todas_221 = ANALOGICAS_221
df221_masked = mascarar_quality_zero(df221, todas_221)
diag_221_antes = diagnostico_nulos(df221_masked, 'URT221 [antes da imputação]')

# Resample sem colunas binárias (a URT 221 não tem status de equipamentos)
df221_reg = regularizar(df221_masked)
print(f"\n    Linhas após resample 1min: {len(df221_reg):,}")

# Imputar — a lista de status é vazia [] pois a URT 221 não tem status de bomba
# max_gap=MAX_GAP_INTERP garante o mesmo limite de 30 minutos
df221_trat = imputar(df221_reg, ANALOGICAS_221, [], max_gap=MAX_GAP_INTERP)
diag_221_depois = diagnostico_nulos(df221_trat, 'URT221 [após imputação]')


# ═══════════════════════════════════════════════════════════════════
# BLOCO C — GERAÇÃO DE GRÁFICOS COMPARATIVOS
# ═══════════════════════════════════════════════════════════════════
# Geramos 4 gráficos para validação visual do tratamento.
# IMPORTANTE: Chamamos resample().mean() nos dados mascarados para
# criar a versão "original em 1min" com os NaN visíveis nos gaps.

print("\n>>> Gerando gráficos comparativos ...")

# Gráfico 1: Pressão da URT 220 — antes x depois do tratamento
plot_comparativo(
    df220_masked.resample('1min').mean(),  # Original em 1min (com NaN nos gaps)
    df220_trat,                            # Tratado (com NaN preenchidos)
    'PIT_220_S01_000',                     # Coluna a plotar
    'URT220 — Pressão (PIT_220_S01_000)', # Título do gráfico
    'graf_220_pressao.png'                 # Arquivo de saída
)

# Gráfico 2: Nível do reservatório da URT 220
plot_comparativo(
    df220_masked.resample('1min').mean(),
    df220_trat,
    'LIT_220_RA2_000',
    'URT220 — Nível Reservatório (LIT_220_RA2_000)',
    'graf_220_nivel.png'
)

# Gráfico 3: Pressão da URT 221 — variável alvo do modelo
plot_comparativo(
    df221_masked.resample('1min').mean(),
    df221_trat,
    'PIT_221_S01_000',
    'URT221 — Pressão (PIT_221_S01_000)',
    'graf_221_pressao.png'
)

# Gráfico 4: Vazão acumulada da URT 221 (totalizador de fluxo)
plot_comparativo(
    df221_masked.resample('1min').mean(),
    df221_trat,
    'FIT_221_S01_TLL',
    'URT221 — Vazão Acumulada (FIT_221_S01_TLL)',
    'graf_221_vazao_acum.png'
)


# ═══════════════════════════════════════════════════════════════════
# BLOCO D — RESUMO ESTATÍSTICO
# ═══════════════════════════════════════════════════════════════════
# describe() é uma função do pandas que calcula automaticamente:
# - count: quantidade de valores não-nulos
# - mean: média
# - std: desvio padrão (quanto os valores variam em torno da média)
# - min: valor mínimo
# - 25%, 50%, 75%: percentis (quartis)
# - max: valor máximo

print("\n" + "="*55)
print("  RESUMO ESTATÍSTICO — COLUNAS PRINCIPAIS")
print("="*55)

# Selecionar as colunas mais importantes para o relatório
cols_220 = ['PIT_220_S01_000', 'LIT_220_RA2_000', 'CMB_220_S2A_EST', 'CMB_220_S2B_EST']
cols_221 = ['PIT_221_S01_000', 'FIT_221_S01_000', 'FIT_221_S01_TLL']

print("\n[URT220 — Tratado]")
# [c for c in cols_220 if c in df220_trat.columns] = list comprehension:
# seleciona apenas as colunas que existem no DataFrame (evita erro se alguma faltar)
print(df220_trat[[c for c in cols_220 if c in df220_trat.columns]].describe().round(4))

print("\n[URT221 — Tratado]")
print(df221_trat[[c for c in cols_221 if c in df221_trat.columns]].describe().round(4))


# ═══════════════════════════════════════════════════════════════════
# BLOCO E — EXPORTAÇÃO DOS DADOS TRATADOS
# ═══════════════════════════════════════════════════════════════════
# Salvamos os DataFrames tratados em arquivos Excel para uso posterior
# no script de treinamento do modelo (treinar_modelo.py).

print("\n>>> Exportando arquivos tratados ...")

# to_excel: salvar DataFrame como arquivo .xlsx
# O índice (timestamps) é incluído automaticamente na primeira coluna
df220_trat.to_excel(SAIDA_220)
print(f"  ✅ {SAIDA_220}  ({len(df220_trat):,} linhas)")

df221_trat.to_excel(SAIDA_221)
print(f"  ✅ {SAIDA_221}  ({len(df221_trat):,} linhas)")


# ═══════════════════════════════════════════════════════════════════
# BLOCO F — RELATÓRIO CONSOLIDADO DE IMPUTAÇÃO
# ═══════════════════════════════════════════════════════════════════
# Função inline para resumir o resultado da imputação de cada URT.

print("\n" + "="*55)
print("  RELATÓRIO CONSOLIDADO")
print("="*55)

def resumo_imputacao(df_antes, df_depois, label):
    """
    Exibe um resumo comparando a quantidade de NaN antes e depois da imputação.

    PARÂMETROS:
      df_antes: DataFrame com NaN (após resample, antes de imputar)
      df_depois: DataFrame imputado (após preencher os NaN)
      label: nome da URT para identificação no relatório
    """
    total = len(df_depois)  # Total de linhas no dataset final

    # sum().sum() = somar todos os NaN em TODAS as colunas (total global)
    nulos_antes  = df_antes.isnull().sum().sum()
    nulos_depois = df_depois.isnull().sum().sum()

    # Calcular quantos valores foram efetivamente preenchidos
    imputados = nulos_antes - nulos_depois

    print(f"\n  {label}")
    print(f"    Linhas no dataset final : {total:>8,}")        # :>8, = alinhar à direita em 8 caracteres
    print(f"    NaN antes da imputação  : {nulos_antes:>8,}")
    print(f"    NaN após  a imputação   : {nulos_depois:>8,}")
    print(f"    Valores imputados       : {imputados:>8,}")

# Chamar o resumo para cada URT
resumo_imputacao(df220_reg, df220_trat, 'URT220')
resumo_imputacao(df221_reg, df221_trat, 'URT221')

# Mensagem final confirmando todos os arquivos gerados
print("\n" + "="*55)
print("  Concluído! Arquivos gerados:")
print(f"    {SAIDA_220}")
print(f"    {SAIDA_221}")
print("    graf_220_pressao.png")
print("    graf_220_nivel.png")
print("    graf_221_pressao.png")
print("    graf_221_vazao_acum.png")
print("="*55 + "\n")
