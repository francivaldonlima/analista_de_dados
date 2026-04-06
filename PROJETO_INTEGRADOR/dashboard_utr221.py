# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║                    dashboard_utr221.py                          ║
║       Dashboard Interativo — Simulação de Falha UTR-221         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OBJETIVO:                                                       ║
║    Dashboard web interativo criado com Streamlit para simular    ║
║    o comportamento da pressão UTR-221 durante um período de      ║
║    falha de comunicação — quando o sinal do SCADA é perdido      ║
║    e não há dados reais disponíveis.                             ║
║                                                                  ║
║  O QUE É STREAMLIT?                                              ║
║    Streamlit é uma biblioteca Python que transforma scripts      ║
║    em aplicações web interativas sem precisar de HTML/CSS/JS.    ║
║    Basta escrever Python e o Streamlit cria a interface.         ║
║                                                                  ║
║  COMO FUNCIONA A SIMULAÇÃO?                                      ║
║    1. O usuário define o período da falha (data, hora, duração)  ║
║    2. O sistema carrega os últimos 60 min de dados reais como    ║
║       "semente" (seed) para inicializar os lags e médias         ║
║    3. O modelo Random Forest prevê a pressão minuto a minuto     ║
║       de forma recursiva durante todo o período de falha         ║
║    4. O gráfico mostra o antes (real) e durante (previsto)       ║
║    5. Se houver dados reais do período, exibe comparação         ║
║                                                                  ║
║  PRÉ-REQUISITO:                                                  ║
║    Executar primeiro: python treinar_modelo.py                   ║
║    (gera os arquivos .pkl necessários)                           ║
║                                                                  ║
║  COMO EXECUTAR:                                                  ║
║    python -m streamlit run dashboard_utr221.py                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTAÇÃO DAS BIBLIOTECAS
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
# Streamlit: framework para criar dashboards web com Python puro
# st.title(), st.sidebar, st.button(), st.metric() etc. = componentes da interface

import pandas as pd    # Manipulação de DataFrames (tabelas de dados)
import numpy as np     # Operações matemáticas e arrays

import matplotlib.pyplot as plt    # Criação de gráficos
import matplotlib.dates as mdates  # Formatação de datas nos gráficos

import joblib
# joblib: carrega os arquivos .pkl salvos pelo treinar_modelo.py
# pkl = pickle = formato de serialização de objetos Python

from datetime import datetime, date, time
# datetime: manipulação de datas e horas
# date: apenas data (sem hora)
# time: apenas hora (sem data)


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONFIGURAÇÃO DA PÁGINA DO DASHBOARD
# ═══════════════════════════════════════════════════════════════════
# Esta seção define as configurações globais da página web.
# DEVE ser a PRIMEIRA chamada Streamlit no script (antes de qualquer st.xxx)

st.set_page_config(
    page_title="UTR-221 — Simulação de Falha",  # Título na aba do navegador
    page_icon="🚰",                               # Ícone da aba (emoji ou URL de imagem)
    layout="wide",                                # Layout: "wide" = usar toda a largura da tela
                                                  # alternativa: "centered" = coluna central
)

# Título principal da página (exibido no topo do dashboard)
st.title("🚰 UTR-221 — Simulação de Falha de Comunicação")

# Texto descritivo abaixo do título (suporta Markdown: **negrito**, *itálico*)
st.markdown(
    "Defina o período e os parâmetros da falha. "
    "O modelo de Machine Learning irá prever a **pressão UTR-221** minuto a minuto."
)


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CARREGAMENTO DOS ARTEFATOS DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Carregamos os 3 arquivos .pkl gerados pelo treinar_modelo.py.
# Usamos cache para não recarregar a cada interação do usuário.

@st.cache_resource(show_spinner="Carregando modelo treinado...")
def carregar_artefatos():
    """
    Carrega os artefatos do modelo treinado do disco.

    @st.cache_resource: decorator do Streamlit que armazena o resultado
    em memória após a primeira execução. Nas próximas chamadas, retorna
    o valor cacheado sem re-executar a função.
    POR QUÊ? Carregar o modelo do disco toda vez seria muito lento
    (cada clique do usuário re-carregaria os arquivos).

    show_spinner: exibe uma mensagem de "carregando..." na tela enquanto
    a função executa pela primeira vez.

    RETORNA:
      tuple: (modelo, historico, meta) — os 3 artefatos salvos
    """
    modelo    = joblib.load('modelo_utr221.pkl')     # Modelo Random Forest treinado
    historico = joblib.load('historico_utr221.pkl')  # Dados históricos (seed da simulação)
    meta      = joblib.load('meta_utr221.pkl')       # Metadados: features, métricas, setpoints
    return modelo, historico, meta

# Tentar carregar os artefatos — mostrar erro amigável se não encontrar
try:
    modelo, historico, meta = carregar_artefatos()

    # Extrair informações dos metadados para uso no dashboard
    FEATURES       = meta['features']                     # Lista das 19 features na ordem correta
    HORIZONTE_MIN  = meta.get('horizonte_min', 30)        # Minutos à frente (default: 30)
    SETPOINT_ALTO  = meta.get('setpoint_alto',  1.68)     # BAR: pressão de desligamento
    SETPOINT_BAIXO = meta.get('setpoint_baixo', 1.55)     # BAR: pressão de acionamento
    # .get(chave, valor_default): retorna o valor da chave ou o default se não existir

except FileNotFoundError:
    # Mostrar mensagem de erro vermelha no dashboard se os arquivos não forem encontrados
    st.error(
        "Arquivos do modelo não encontrados. "
        "Execute primeiro: `python treinar_modelo.py`"
    )
    st.stop()  # Para a execução do script (não renderiza o restante da página)


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 3 — PAINEL LATERAL (SIDEBAR) COM OS PARÂMETROS
# ═══════════════════════════════════════════════════════════════════
# A sidebar é o painel lateral esquerdo onde o usuário configura a simulação.
# "with st.sidebar:" indica que todos os componentes dentro do bloco
# serão renderizados na sidebar.

with st.sidebar:
    st.header("Parâmetros da Simulação")
    st.markdown("---")  # Linha horizontal separadora

    # ── Configuração do Período da Falha ──────────────────────────
    st.subheader("Período da Falha")

    # Definir limites do seletor de data
    data_minima = historico.index[0].date()   # Primeiro dia no histórico
    data_maxima = (historico.index[-1] + pd.Timedelta(days=7)).date()
        # Permite simular até 7 dias depois do fim do histórico

    # Seletor de data com calendário interativo
    data_falha = st.date_input(
        "Data de início",               # Rótulo exibido ao usuário
        value=historico.index[-1].date(),  # Valor padrão: último dia do histórico
        min_value=data_minima,          # Mínimo permitido
        max_value=data_maxima,          # Máximo permitido
    )

    # Campo de texto para a hora (permite entrada manual no formato HH:MM)
    hora_falha_str = st.text_input(
        "Hora de início (HH:MM)",
        # Valor padrão: hora e minuto do último registro do histórico
        value=time(historico.index[-1].hour, historico.index[-1].minute).strftime("%H:%M"),
    )

    # Converter o texto "HH:MM" para objeto time do Python
    try:
        # split(":") = dividir "14:30" em ["14", "30"]
        # int(p) = converter cada parte para inteiro
        # time(*[...]) = criar objeto time com hora e minuto
        hora_falha = time(*[int(p) for p in hora_falha_str.split(":")])
    except Exception:
        # Se o usuário digitou um formato inválido (ex: "14h30"), mostrar erro
        st.error("Hora inválida. Use o formato HH:MM (ex: 14:30)")
        st.stop()

    # Slider para escolher a duração da falha em minutos
    duracao_min = st.slider(
        "Duração da falha (minutos)",
        min_value=5,     # Mínimo: 5 minutos
        max_value=420,   # Máximo: 7 horas (420 minutos)
        value=60,        # Padrão: 1 hora (60 minutos)
        step=5,          # Incremento: 5 minutos por clique
    )

    st.markdown("---")  # Separador visual

    # ── Configuração do Controle das Bombas ───────────────────────
    st.subheader("Controle das Bombas (Rodízio)")

    # Inputs numéricos para os setpoints (pressões limites de controle)
    # Estes valores são carregados dos metadados mas podem ser ajustados
    sp_alto  = st.number_input(
        "Setpoint alto — desliga bombas (BAR)",
        value=SETPOINT_ALTO,   # Valor padrão dos metadados
        step=0.01,             # Incremento: 0.01 BAR por clique
        format="%.2f"          # Formato de exibição: 2 casas decimais
    )
    sp_baixo = st.number_input(
        "Setpoint baixo — liga bombas (BAR)",
        value=SETPOINT_BAIXO,
        step=0.01,
        format="%.2f"
    )

    # Texto explicativo dos setpoints (caption = texto pequeno e cinza)
    st.caption(
        f"Pressão >= {sp_alto:.2f} BAR → bombas desligam  |  "
        f"Pressão <= {sp_baixo:.2f} BAR → bombas ligam"
    )

    # Seleção do modo de inicialização das bombas
    # "Automático" = usa o estado real do histórico como ponto de partida
    # "Manual" = o usuário define o estado inicial das bombas
    modo_bomba = st.radio(
        "Estado inicial das bombas",
        options=["Automático (do histórico)", "Manual (definir abaixo)"],
        index=0,  # Selecionar "Automático" por padrão
    )

    if modo_bomba == "Manual (definir abaixo)":
        # Modo Manual: exibir selectboxes para definir o estado de cada bomba
        bomba1_init = float(st.selectbox(
            "Bomba 1 (S2A) — estado inicial",
            options=[0, 1],   # Valores possíveis: 0 ou 1
            # format_func: função para converter o valor em texto exibido
            format_func=lambda x: "Ligada" if x == 1 else "Desligada",
            index=1,          # Selecionar "Ligada" (valor 1) por padrão
        ))
        bomba2_init = float(st.selectbox(
            "Bomba 2 (S2B) — estado inicial",
            options=[0, 1],
            format_func=lambda x: "Ligada" if x == 1 else "Desligada",
            index=0,          # Selecionar "Desligada" (valor 0) por padrão
        ))
    else:
        # Modo Automático: ler o estado das bombas do último registro histórico
        bomba1_init = float(historico['BOMBA_1'].iloc[-1])  # -1 = último valor
        bomba2_init = float(historico['BOMBA_2'].iloc[-1])

    st.markdown("---")

    # ── Configuração do Nível do Reservatório UTR-220 ─────────────
    st.subheader("Nível UTR-220")
    nivel_220 = st.number_input(
        "Nível UTR-220 (metros)",
        min_value=0.0,    # Não pode ser negativo
        max_value=50.0,   # Máximo razoável para um reservatório
        # Valor padrão: último nível real disponível
        value=float(round(historico['NIVEL-UTR-220'].iloc[-1], 2)),
        step=0.01,
        format="%.2f",
    )

    st.markdown("---")

    # Botão principal para iniciar a simulação
    # type="primary" = botão em destaque (cor azul)
    # use_container_width=True = ocupar toda a largura da sidebar
    simular = st.button("Executar Simulação", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 4 — CARDS DE MÉTRICAS DO MODELO
# ═══════════════════════════════════════════════════════════════════
# Exibir informações sobre a qualidade do modelo de forma visual
# usando "cards" (caixas de métrica do Streamlit).

# Extrair métricas salvas nos metadados
metricas_md = meta.get('metricas', {})  # {} = dict vazio se não encontrar

# Criar 4 colunas de igual largura para os 4 cards
c1, c2, c3, c4 = st.columns(4)

# st.metric(rótulo, valor): cria um card com o rótulo em cima e valor em destaque
c1.metric("Acurácia do Modelo",       f"{metricas_md.get('Acuracia', 0):.1f}%")
    # .1f = 1 casa decimal, % no final
c2.metric("MAE (BAR)",                f"{metricas_md.get('MAE', 0):.4f}")
    # .4f = 4 casas decimais
c3.metric("R²",                       f"{metricas_md.get('R2', 0):.4f}")
c4.metric(f"Horizonte de previsão",   f"{HORIZONTE_MIN} min")

st.markdown("---")  # Linha separadora antes do conteúdo principal


# ═══════════════════════════════════════════════════════════════════
# SEÇÃO 5 — EXECUÇÃO DA SIMULAÇÃO
# ═══════════════════════════════════════════════════════════════════
# Esta seção só executa quando o usuário clica no botão "Executar Simulação".
# "if simular:" = bloco condicional: True quando o botão foi clicado.

if simular:

    # Combinar data e hora da falha num único objeto Timestamp do pandas
    # datetime.combine(data, hora) = juntar date + time em datetime
    # pd.Timestamp() = converter para o formato do pandas
    falha_inicio = pd.Timestamp(datetime.combine(data_falha, hora_falha))

    # Calcular o timestamp de fim da falha
    falha_fim = falha_inicio + pd.Timedelta(minutes=duracao_min)

    # Exibir subtítulo com o período simulado
    st.subheader(
        f"Simulação: {falha_inicio.strftime('%d/%m/%Y %H:%M')} "
        f"— {falha_fim.strftime('%H:%M')} ({duracao_min} min)"
    )

    # ── Preparar o histórico inicial (SEED da simulação) ──────────
    # Para calcular os lags e médias móveis do primeiro minuto da simulação,
    # precisamos dos 60 minutos reais ANTES do início da falha.
    # Este histórico é a "memória inicial" do sistema simulado.

    # Selecionar dados reais anteriores ao início da falha
    dados_antes = historico[historico.index < falha_inicio]['PRESSAO-UTR-221']

    if len(dados_antes) == 0:
        # Caso especial: nenhum dado histórico antes da falha
        # (ex: simulação começa antes do início do histórico)
        st.warning("Nenhum dado histórico antes da falha. Usando último valor disponível.")
        seed_nivel    = float(historico['PRESSAO-UTR-221'].iloc[-1])
        historico_sim = [seed_nivel] * 60  # Repetir o último valor 60 vezes

    elif len(dados_antes) < 60:
        # Caso intermediário: menos de 60 minutos de histórico disponível
        # Completamos com o primeiro valor disponível (repetido)
        st.info(f"Histórico disponível: {len(dados_antes)} min (ideal: 60). Completando com último valor.")
        # Criar uma lista de preenchimento para completar até 60 posições
        complemento   = [float(dados_antes.iloc[0])] * (60 - len(dados_antes))
        historico_sim = complemento + list(dados_antes.values)
        # complemento vai na frente + dados reais depois (ordem cronológica)

    else:
        # Caso normal: pelo menos 60 minutos de histórico disponível
        # Usar os 60 minutos mais recentes antes da falha
        historico_sim = list(dados_antes.iloc[-60:].values)

    # ── Inicializar o estado das bombas ───────────────────────────
    # Obter os índices dos timestamps históricos anteriores à falha
    idx_antes = historico.index[historico.index < falha_inicio]

    if len(idx_antes) > 0:
        # Usar o estado real das bombas no último momento antes da falha
        bomba1     = float(historico.loc[idx_antes[-1], 'BOMBA_1'])
        bomba2     = float(historico.loc[idx_antes[-1], 'BOMBA_2'])
        # Estado anterior da Bomba 2 (para calcular transição no 1º passo)
        bomba2_ant = float(historico.loc[idx_antes[-2], 'BOMBA_2']) if len(idx_antes) >= 2 else bomba2
        # Tempo que a Bomba 2 já estava ligada antes da falha
        tempo_b2   = int(historico.loc[idx_antes[-1], 'tempo_ligada_B2'])
    else:
        # Sem dados históricos: usar os valores definidos pelo usuário
        bomba1, bomba2, bomba2_ant, tempo_b2 = bomba1_init, bomba2_init, bomba2_init, 0

    # Se o usuário escolheu modo Manual, sobrescrever o estado automático
    if modo_bomba == "Manual (definir abaixo)":
        bomba1, bomba2 = float(bomba1_init), float(bomba2_init)
        bomba2_ant = bomba2  # Estado anterior = estado inicial (sem transição)
        tempo_b2   = 0       # Zerar contador de tempo

    # Inicializar históricos de 15 minutos das bombas (para B1_media_15m e B2_media_15m)
    if len(idx_antes) >= 15:
        # Usar os 15 minutos reais mais recentes
        hist_b1 = list(historico.loc[idx_antes[-15:], 'BOMBA_1'].values)
        hist_b2 = list(historico.loc[idx_antes[-15:], 'BOMBA_2'].values)
    else:
        # Não há histórico suficiente: usar o estado atual repetido
        hist_b1 = [bomba1] * 15
        hist_b2 = [bomba2] * 15

    # ── Loop Principal: Predição Recursiva Minuto a Minuto ─────────
    # Para cada minuto do período de falha, calculamos as features
    # e o modelo prevê a pressão daquele minuto.
    # A previsão alimenta o próximo passo (recursividade).

    registros = []  # Lista para acumular os resultados

    # Criar todos os timestamps do período de falha (1 minuto de intervalo)
    timestamps_falha = pd.date_range(falha_inicio, falha_fim, freq='1min')

    for ts in timestamps_falha:  # Iterar por cada minuto da falha

        # ── Extrair lags do histórico disponível ──────────────────
        nivel_lag1  = historico_sim[-1]   # Pressão 1 min atrás
        nivel_lag2  = historico_sim[-2]  if len(historico_sim) >= 2  else historico_sim[0]
        nivel_lag6  = historico_sim[-6]  if len(historico_sim) >= 6  else historico_sim[0]
        nivel_lag15 = historico_sim[-15] if len(historico_sim) >= 15 else historico_sim[0]
        nivel_lag60 = historico_sim[-60] if len(historico_sim) >= 60 else historico_sim[0]
            # Se não há 60 valores no histórico, usar o mais antigo disponível

        # Guardar estado anterior da Bomba 2 para calcular transição
        bomba2_ant = bomba2

        # ── Lógica de controle por setpoints ──────────────────────
        # Simula o controlador automático do sistema de distribuição de água.
        # Esta lógica é aplicada INDEPENDENTE do modo de bomba escolhido.
        # No modo Manual, o usuário define o estado INICIAL,
        # mas durante a simulação os setpoints continuam controlando.
        if nivel_lag1 >= sp_alto:
            # Pressão alta: desligar as bombas para evitar sobrepressão
            bomba1, bomba2 = 0.0, 0.0
            tempo_b2 = 0
        elif nivel_lag1 <= sp_baixo:
            # Pressão baixa: ligar as bombas para reestabelecer a pressão
            bomba1, bomba2 = 1.0, 1.0

        # Atualizar o contador de tempo da Bomba 2
        if bomba2 == 1:
            tempo_b2 += 1  # Incrementar minuto a minuto enquanto ligada
        else:
            tempo_b2 = 0   # Reiniciar quando desligada

        # Atualizar históricos de 15 minutos (janela deslizante)
        hist_b1.append(bomba1)
        hist_b1 = hist_b1[-15:]  # Manter apenas os últimos 15 valores
        hist_b2.append(bomba2)
        hist_b2 = hist_b2[-15:]

        # Calcular turno com base na hora atual do timestamp
        hora_val  = ts.hour
        turno_val = 0 if hora_val < 6 else (1 if hora_val < 12 else (2 if hora_val < 18 else 3))

        # ── Montar as 19 features para este minuto ────────────────
        feat_values = {
            'BOMBA_1':         bomba1,
            'BOMBA_2':         bomba2,
            'BOMBA_2_lag1':    bomba2_ant,
            'B1_media_15m':    np.mean(hist_b1),    # Média dos últimos 15 min
            'B2_media_15m':    np.mean(hist_b2),
            'nivel_220_lag1':  nivel_220,            # Nível fixo definido pelo usuário
            'nivel_221_lag1':  nivel_lag1,
            'nivel_221_lag5':  nivel_lag6,           # lag5 usa posição -6 no buffer
            'nivel_221_lag15': nivel_lag15,
            'hora':            hora_val,
            'turno':           turno_val,
            'tendencia_1min':  nivel_lag1 - nivel_lag2,   # Variação 1 minuto
            'tendencia_5min':  nivel_lag1 - nivel_lag6,   # Variação 5 minutos
            'tendencia_1h':    nivel_lag1 - nivel_lag60,  # Variação 1 hora
            'media_movel_1h':  np.mean(historico_sim[-60:]),   # Média 1 hora
            'media_movel_15m': np.mean(historico_sim[-15:]),   # Média 15 min
            'transicao_B2':    bomba2 - bomba2_ant,            # Mudança no estado da bomba
            'tempo_ligada_B2': tempo_b2,
            'rodizio':         1 if bomba1 == 1 else (2 if bomba2 == 1 else 0),
        }

        # Converter para DataFrame de 1 linha (formato esperado pelo modelo)
        # [FEATURES] garante a ordem correta das colunas
        X_pred   = pd.DataFrame([feat_values])[FEATURES]

        # Fazer a previsão com o modelo treinado
        # max(..., 0) = pressão não pode ser negativa fisicamente
        previsao = max(modelo.predict(X_pred)[0], 0)

        # ── Verificar se há dado real para comparação ──────────────
        # Se o período de falha coincide com dados históricos existentes,
        # podemos comparar o previsto com o real (validação da simulação).
        real = None  # Assume que não há dado real disponível
        if ts in historico.index:
            r = historico.loc[ts, 'PRESSAO-UTR-221']
            if r != 0:  # Ignorar zeros (podem ser falhas de sensor)
                real = r

        # Acumular resultado deste minuto
        registros.append({
            'Timestamp': ts,
            'Previsto':  round(previsao, 4),  # Pressão prevista em BAR
            'Real':      real,                 # Pressão real (None se não disponível)
            'Bomba1':    int(bomba1),          # Estado da Bomba 1 neste minuto
            'Bomba2':    int(bomba2),          # Estado da Bomba 2 neste minuto
            'Rodizio':   int(1 if bomba1 == 1 else (2 if bomba2 == 1 else 0)),
        })

        # Adicionar a previsão ao histórico para o PRÓXIMO passo
        # É aqui que a recursividade acontece: previsão → histórico → próxima previsão
        historico_sim.append(previsao)

    # Converter a lista de resultados em um DataFrame indexado por timestamp
    df_resultado = pd.DataFrame(registros).set_index('Timestamp')

    # ── Gráfico Principal da Simulação ────────────────────────────
    # Mostrar: dados reais 1h antes + dados previstos durante a falha
    # Opcionalmente: dados reais durante a falha (para validação)

    # Janela de contexto: 1 hora ANTES do início da falha
    contexto_inicio = falha_inicio - pd.Timedelta(hours=1)

    # Filtrar dados históricos reais para o contexto (sem zeros)
    dados_contexto  = historico.loc[
        (historico.index >= contexto_inicio) & (historico.index < falha_inicio),
        'PRESSAO-UTR-221'
    ]
    dados_contexto = dados_contexto[dados_contexto != 0]  # Remover zeros inválidos

    # Criar figura do matplotlib
    fig, ax = plt.subplots(figsize=(14, 5))

    if len(dados_contexto) > 0:
        # Linha azul: pressão real antes da falha (contexto)
        ax.plot(dados_contexto.index, dados_contexto.values,
                color='steelblue', linewidth=2, label='Pressão Real (antes da falha)')

    # Linha vermelha tracejada: previsão do modelo durante a falha
    ax.plot(df_resultado.index, df_resultado['Previsto'],
            color='tomato', linewidth=2, linestyle='--',
            label=f'Previsão ML ({HORIZONTE_MIN} min horizonte)')

    # Verificar se há dados reais disponíveis durante o período de falha
    # (só acontece se a falha simulada coincide com o período histórico)
    reais_disponiveis = df_resultado['Real'].dropna()
    if len(reais_disponiveis) > 0:
        # Linha verde: dados reais durante a falha (quando disponíveis)
        # Permite comparar visualmente previsto vs. real
        ax.plot(reais_disponiveis.index, reais_disponiveis.values,
                color='seagreen', linewidth=1.5, alpha=0.85,
                label='Pressão Real (disponível para comparação)')

    # Área sombreada vermelha indicando o período de falha
    # axvspan = retângulo vertical sombreado entre dois timestamps
    ax.axvspan(falha_inicio, falha_fim, alpha=0.10, color='red', label='Período de falha')

    # Linha vertical marcando o início da falha
    ax.axvline(falha_inicio, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    # Linhas horizontais dos setpoints (referência visual)
    ax.axhline(sp_alto,  color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(sp_baixo, color='gray', linestyle=':',  alpha=0.4, linewidth=1)

    # Configurações do gráfico
    ax.set_title(
        f"Simulação UTR-221 — {falha_inicio.strftime('%d/%m/%Y')}  "
        f"({hora_falha.strftime('%H:%M')} — {falha_fim.strftime('%H:%M')})",
        fontsize=13
    )
    ax.set_xlabel('Horário')
    ax.set_ylabel('Pressão (BAR)')
    ax.legend(loc='best', fontsize=9)  # Legenda na melhor posição automaticamente
    ax.grid(True, alpha=0.3)           # Grade suave
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Formato HH:MM no eixo X
    plt.xticks(rotation=45)            # Rotacionar rótulos para não sobrepor
    plt.tight_layout()

    # Exibir o gráfico no dashboard Streamlit
    st.pyplot(fig)

    # ── Métricas da Simulação (se houver dados reais para comparar) ──
    # Esta seção só aparece quando os dados históricos cobrem o período simulado
    if len(reais_disponiveis) > 0:
        # Importar métricas do scikit-learn para calcular a qualidade da simulação
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Alinhar os dados previstos com os dados reais disponíveis
        prev_al  = df_resultado.loc[reais_disponiveis.index, 'Previsto']

        # Calcular métricas de comparação previsto vs. real
        sim_mae  = mean_absolute_error(reais_disponiveis, prev_al)
        sim_rmse = np.sqrt(mean_squared_error(reais_disponiveis, prev_al))
        sim_r2   = r2_score(reais_disponiveis, prev_al)

        # MAPE com proteção contra divisão por zero
        mask_nz  = reais_disponiveis != 0
        sim_mape = (np.mean(np.abs(
            (reais_disponiveis[mask_nz] - prev_al[mask_nz]) / reais_disponiveis[mask_nz]
        )) * 100) if mask_nz.sum() > 0 else 0.0  # 0% se não há dados válidos

        # Exibir métricas em 4 cards
        st.markdown("#### Métricas da Simulação (comparação com dados reais)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE (BAR)",  f"{sim_mae:.4f}")
        mc2.metric("RMSE (BAR)", f"{sim_rmse:.4f}")
        mc3.metric("R²",         f"{sim_r2:.4f}")
        mc4.metric("Acurácia",   f"{100 - sim_mape:.1f}%")

    # ── Resumo dos Parâmetros Utilizados ──────────────────────────
    st.markdown("#### Parâmetros utilizados")
    col_p1, col_p2 = st.columns(2)  # 2 colunas de igual largura

    with col_p1:
        # Informações sobre as bombas
        st.info(
            # \n em f-string e dois espaços antes = quebra de linha no Markdown
            f"**Estado inicial das bombas:** {modo_bomba}  \n"
            f"**Bomba 1 (S2A) no final:** {'Ligada' if int(df_resultado['Bomba1'].iloc[-1]) == 1 else 'Desligada'}  \n"
            f"**Bomba 2 (S2B) no final:** {'Ligada' if int(df_resultado['Bomba2'].iloc[-1]) == 1 else 'Desligada'}  \n"
            f"**Nível UTR-220:** {nivel_220:.2f} m"
        )

    with col_p2:
        # Informações sobre o período da falha
        st.info(
            f"**Início da falha:** {falha_inicio.strftime('%d/%m/%Y %H:%M')}  \n"
            f"**Fim da falha:** {falha_fim.strftime('%d/%m/%Y %H:%M')}  \n"
            f"**Duração:** {duracao_min} minutos  \n"
            f"**Setpoints:** desliga >= {sp_alto:.2f} BAR  |  liga <= {sp_baixo:.2f} BAR"
        )

    # ── Tabela Detalhada Minuto a Minuto ──────────────────────────
    st.markdown("#### Previsão minuto a minuto")

    # Preparar DataFrame para exibição
    df_exibir = df_resultado[['Previsto', 'Real', 'Bomba1', 'Bomba2']].copy()

    # Formatar o índice (timestamps) como texto legível
    df_exibir.index = df_exibir.index.strftime('%d/%m/%Y %H:%M')

    # Renomear colunas para nomes mais amigáveis ao usuário
    df_exibir.columns = ['Previsto (BAR)', 'Real (BAR)', 'Bomba 1', 'Bomba 2']

    # Converter coluna Real para numérico (pode conter None = sem dado real)
    # errors='coerce' = converter None/NaN para NaN numérico (não lança erro)
    df_exibir['Real (BAR)'] = pd.to_numeric(df_exibir['Real (BAR)'], errors='coerce')

    # Calcular o erro absoluto quando há dado real disponível
    df_exibir['Erro (BAR)'] = (
        df_exibir['Real (BAR)'] - df_exibir['Previsto (BAR)']
    ).abs().round(4)
    # .abs() = valor absoluto (sempre positivo)
    # .round(4) = arredondar para 4 casas decimais

    # Exibir tabela interativa (permite ordenar e buscar)
    # use_container_width=True = usar toda a largura disponível
    st.dataframe(df_exibir, use_container_width=True)

    # ── Botão de Download dos Resultados ──────────────────────────
    # Converter o DataFrame para CSV codificado em UTF-8
    csv = df_resultado.to_csv(index=True).encode('utf-8')

    # Criar botão de download (o arquivo só é gerado quando clicado)
    st.download_button(
        label="Baixar resultado em CSV",     # Texto do botão
        data=csv,                            # Conteúdo do arquivo
        # Nome do arquivo com data/hora da falha no formato YYYYMMDD_HHMM
        file_name=f"simulacao_falha_{falha_inicio.strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv',                     # Tipo MIME para o navegador reconhecer como CSV
    )

# ── Estado inicial (antes de clicar em Executar Simulação) ────────
else:
    # Quando o botão ainda não foi clicado, exibir informações sobre os dados disponíveis
    st.info("Preencha os parâmetros no painel lateral e clique em **Executar Simulação**.")

    st.markdown("#### Dados históricos disponíveis")

    # Criar 3 cards com informações do histórico
    c1, c2, c3 = st.columns(3)

    # Exibir primeiro e último timestamp dos dados de treino
    c1.metric("Primeiro registro",
              pd.Timestamp(meta['primeiro_ts']).strftime('%d/%m/%Y %H:%M'))
    c2.metric("Último registro",
              pd.Timestamp(meta['ultimo_ts']).strftime('%d/%m/%Y %H:%M'))
    c3.metric("Total de minutos",
              f"{meta['total_minutos']:,}")
              # :, = formatar número com separador de milhar (ex: 43.200)
