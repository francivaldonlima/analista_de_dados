import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURACAO
# ─────────────────────────────────────────────

DATA_INICIO = '2026-02-28 18:00:00'

arquivos = {
    'urt220': 'tratamento_de_dados/urt220.xlsx',
    'urt221': 'tratamento_de_dados/urt221.xlsx',
}

# ─────────────────────────────────────────────
# PROCESSAMENTO
# ─────────────────────────────────────────────

for nome, caminho in arquivos.items():
    print(f"\n{'='*50}")
    print(f"Processando: {caminho}")
    print(f"{'='*50}")

    df = pd.read_excel(caminho)
    print(f"Shape original: {df.shape}")

    # 1. Converter E3TIMESTAMP para datetime (ano, mes, dia, hora)
    df['E3TIMESTAMP'] = pd.to_datetime(
        df['E3TIMESTAMP'],
        format='%d-%b-%y %I.%M.%S.%f000 %p',
        errors='coerce'
    )
    df['E3TIMESTAMP'] = df['E3TIMESTAMP'].dt.floor('s')  # remove microssegundos
    print(f"  E3TIMESTAMP convertido. Exemplo: {df['E3TIMESTAMP'].iloc[0]}")

    # 2. Remover colunas com a palavra QUALITY
    cols_quality = [c for c in df.columns if 'QUALITY' in c]
    df = df.drop(columns=cols_quality)
    print(f"  Colunas QUALITY removidas: {len(cols_quality)}")
    print(f"  Colunas restantes: {df.shape[1]}")

    # 3. Tratar valores nulos com interpolacao linear
    nulos_antes = df.isnull().sum().sum()
    df = df.interpolate(method='linear', limit_direction='both')
    nulos_depois = df.isnull().sum().sum()
    print(f"  Nulos tratados: {nulos_antes} -> {nulos_depois}")

    # 4. Filtrar dados a partir de 28/02/2026 18:00
    df = df[df['E3TIMESTAMP'] >= DATA_INICIO].reset_index(drop=True)
    print(f"  Linhas apos filtro de data: {len(df)}")
    if len(df) > 0:
        print(f"  Periodo: {df['E3TIMESTAMP'].min()} ate {df['E3TIMESTAMP'].max()}")

    # 5. Salvar como .xlsx
    saida = f'tratamento_de_dados/{nome}_filtrado.xlsx'
    df.to_excel(saida, index=False)
    print(f"  Arquivo salvo: {saida}")

print("\nProcessamento concluido!")
