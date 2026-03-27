import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def converter_timestamp(df):
    """Converte a coluna E3TIMESTAMP de texto para datetime."""
    df['E3TIMESTAMP'] = pd.to_datetime(
        df['E3TIMESTAMP'], format='%d-%b-%y %I.%M.%S.%f000 %p', errors='coerce'
    )
    return df


def filtrar_quality(df):
    """Remove linhas onde qualquer coluna QUALITY == 0 (leitura ruim do sensor)."""
    cols_quality = [c for c in df.columns if 'QUALITY' in c]
    mascara = (df[cols_quality] == 0).any(axis=1)
    removidos = mascara.sum()
    df = df[~mascara].reset_index(drop=True)
    print(f"  Linhas removidas por QUALITY=0: {removidos}")
    return df


def tratar_nulos(df):
    """Preenche valores nulos com interpolação linear."""
    nulos_antes = df.isnull().sum().sum()
    df = df.interpolate(method='linear', limit_direction='both')
    nulos_depois = df.isnull().sum().sum()
    print(f"  Nulos tratados: {nulos_antes} -> {nulos_depois}")
    return df


def remover_outliers(df):
    """
    Remove outliers nas colunas de sensores (sem QUALITY, sem TIMESTAMP).
    Usa o método IQR (interquartil): remove valores abaixo de Q1-3*IQR
    ou acima de Q3+3*IQR (fator 3 para ser conservador com dados industriais).
    """
    cols_sensor = [c for c in df.columns if 'QUALITY' not in c and c != 'E3TIMESTAMP']
    total_removidos = 0
    for col in cols_sensor:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 3 * IQR
        limite_sup = Q3 + 3 * IQR
        antes = len(df)
        df = df[(df[col] >= limite_inf) & (df[col] <= limite_sup)]
        removidos = antes - len(df)
        if removidos > 0:
            print(f"  [{col}] outliers removidos: {removidos}")
        total_removidos += removidos
    print(f"  Total de linhas removidas por outlier: {total_removidos}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# PROCESSAMENTO
# ─────────────────────────────────────────────

arquivos = {
    'urt220': 'tratamento_de_dados/urt220.xlsx',
    'urt221': 'tratamento_de_dados/urt221.xlsx',
}

for nome, caminho in arquivos.items():
    print(f"\n{'='*50}")
    print(f"Processando: {caminho}")
    print(f"{'='*50}")

    df = pd.read_excel(caminho)
    print(f"Shape original: {df.shape}")

    # 1. Converter timestamp
    df = converter_timestamp(df)
    print(f"  E3TIMESTAMP convertido para datetime.")

    # 2. Filtrar quality ruim
    df = filtrar_quality(df)

    # 3. Tratar nulos
    df = tratar_nulos(df)

    # 4. Remover outliers
    df = remover_outliers(df)

    print(f"Shape final: {df.shape}")

    # 5. Salvar resultado
    saida = f'tratamento_de_dados/{nome}_tratado.csv'
    df.to_csv(saida, index=False, sep=';', decimal=',')
    print(f"Arquivo salvo: {saida}")

print("\nTratamento concluído!")
