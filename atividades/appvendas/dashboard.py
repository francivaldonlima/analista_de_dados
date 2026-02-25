import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard de Vendas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove padding topo */
    .block-container { padding-top: 1.2rem; }

    /* Sidebar azul escuro */
    [data-testid="stSidebar"] { background-color: #1b3a5c; }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* Cards KPI */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.72rem; color: #6b7a99; font-weight: 700;
        text-transform: uppercase; letter-spacing: .08em;
    }
    .kpi-value {
        font-size: 1.85rem; font-weight: 800; color: #1b3a5c; margin-top: 6px;
    }
    .kpi-delta-pos { font-size: 0.88rem; color: #27ae60; margin-top: 4px; }
    .kpi-delta-neg { font-size: 0.88rem; color: #e74c3c; margin-top: 4px; }

    /* Card genérico de seção */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 4px;
    }
    .card-title {
        font-size: 0.72rem; font-weight: 700; color: #6b7a99;
        text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px;
    }

    /* Label filtros */
    .filtros-label {
        font-size: 0.82rem; font-weight: 800; color: #e74c3c;
        text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px;
    }

    /* Placeholder páginas futuras */
    .placeholder {
        background: #f4f6fb; border: 2px dashed #b0bdd6;
        border-radius: 12px; padding: 80px; text-align: center;
        color: #8a99b5; font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cores padrão dos gráficos ───────────────────────────────────────────────
AZUL_PRINCIPAL = "#2b6cb0"
PALETTE = ["#2b6cb0", "#63b3ed", "#f6ad55", "#fc8181", "#68d391", "#b794f4"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
    font=dict(family="sans-serif", size=11, color="#4a5568"),
)


# ── Carregamento dos dados ──────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_csv("vendas_5000_linhas_tratado.csv", index_col=0)
    df["data_venda"] = pd.to_datetime(df["data_venda"])
    for col in ["categoria", "forma_pagamento", "regiao", "loja", "produto"]:
        df[col] = df[col].astype(str).str.strip()
    return df

df_raw = carregar_dados()


# ── Sidebar – Navegação ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Dashboard Vendas")
    st.markdown("---")
    pagina = st.radio(
        "nav",
        options=[
            "🏠  Visão Geral",
            "📋  Tabela de Dados",
            "📦  Análise de Produtos",
            "🏪  Análise por Loja",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Dados: 2023 – 2024  ·  5 000 registros")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 – VISÃO GERAL  (página inicial)
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠  Visão Geral":

    # ── Helper: aplicar filtros ─────────────────────────────────────────────
    def aplicar_filtros(df, data_ini, data_fim, categoria):
        d = df[(df["data_venda"] >= pd.Timestamp(data_ini)) &
               (df["data_venda"] <= pd.Timestamp(data_fim))]
        if categoria != "Todas":
            d = d[d["categoria"] == categoria]
        return d

    # ════════════════════════════════════════════════════════════════════════
    # LINHA 1 – Gráfico de linha  +  3 KPIs
    # ════════════════════════════════════════════════════════════════════════
    col_linha, col_k1, col_k2, col_k3 = st.columns([3.2, 1.6, 1.6, 1.6])

    # -- Gráfico: Venda por Período (mensal) ---------------------------------
    with col_linha:
        vendas_mes = (
            df_raw.set_index("data_venda")
            .resample("ME")["total_venda"]
            .sum()
            .reset_index()
        )
        vendas_mes.columns = ["mes", "faturamento"]

        fig_linha = go.Figure()
        fig_linha.add_trace(go.Scatter(
            x=vendas_mes["mes"],
            y=vendas_mes["faturamento"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=AZUL_PRINCIPAL, width=2.5),
            fillcolor="rgba(43,108,176,0.15)",
            hovertemplate="<b>%{x|%b/%Y}</b><br>R$ %{y:,.0f}<extra></extra>",
        ))
        fig_linha.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="VENDA POR PERÍODO", font=dict(size=11, color="#6b7a99"),
                       x=0, xanchor="left"),
            xaxis=dict(showgrid=False, tickformat="%b/%y", tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#edf2f7", tickprefix="R$ ",
                       tickformat=",.0f"),
            height=200,
        )
        st.plotly_chart(fig_linha, use_container_width=True)

    # -- KPIs ----------------------------------------------------------------
    lucro_liq   = df_raw["lucro_venda"].sum()
    total_vend  = len(df_raw)

    # Lucro total %  (lucro / faturamento)
    lucro_pct   = lucro_liq / df_raw["total_venda"].sum() * 100

    # Comparação ano anterior (2023 vs 2024)
    lucro_2023 = df_raw[df_raw["data_venda"].dt.year == 2023]["lucro_venda"].sum()
    lucro_2024 = df_raw[df_raw["data_venda"].dt.year == 2024]["lucro_venda"].sum()
    delta_lucro = ((lucro_2024 - lucro_2023) / lucro_2023 * 100) if lucro_2023 else 0

    with col_k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Lucro Líquido</div>
            <div class="kpi-value">R$ {lucro_liq/1000:.0f}K</div>
            <div class="{'kpi-delta-pos' if delta_lucro >= 0 else 'kpi-delta-neg'}">
                {'▲' if delta_lucro >= 0 else '▼'} {abs(delta_lucro):.0f}% vs 2023
            </div>
        </div>""", unsafe_allow_html=True)

    with col_k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Lucro Total</div>
            <div class="kpi-value">▲ {lucro_pct:.0f}%</div>
            <div class="kpi-delta-pos">Margem sobre faturamento</div>
        </div>""", unsafe_allow_html=True)

    with col_k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Vendas</div>
            <div class="kpi-value">{total_vend:,}</div>
            <div class="kpi-delta-pos">Transações registradas</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # LINHA 2 – Top 8 Produtos  |  Por Produto (barras)  |  Formas Pagamento
    # ════════════════════════════════════════════════════════════════════════
    col_top8, col_bar, col_donut = st.columns([2.8, 3.6, 2.8])

    # -- Top 8 Produtos por Faturamento -------------------------------------
    with col_top8:
        st.markdown('<div class="card-title">Top 8 Produtos por Faturamento</div>',
                    unsafe_allow_html=True)
        top8 = (
            df_raw.groupby("produto")["total_venda"]
            .sum()
            .nlargest(8)
            .reset_index()
            .rename(columns={"produto": "Produto", "total_venda": "Faturamento"})
        )
        top8["Faturamento"] = top8["Faturamento"].map("R$ {:,.0f}".format)

        st.dataframe(
            top8,
            use_container_width=True,
            hide_index=True,
            height=260,
        )

    # -- Faturamento por Produto (barras) -----------------------------------
    with col_bar:
        fat_prod = (
            df_raw.groupby("produto")["total_venda"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig_bar = px.bar(
            fat_prod,
            x="produto",
            y="total_venda",
            color_discrete_sequence=[AZUL_PRINCIPAL],
            labels={"produto": "", "total_venda": ""},
        )
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="POR PRODUTO", font=dict(size=11, color="#6b7a99"),
                       x=0.5, xanchor="center"),
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#edf2f7"),
            height=290,
        )
        fig_bar.update_traces(
            hovertemplate="<b>%{x}</b><br>R$ %{y:,.0f}<extra></extra>"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # -- Formas de Pagamento (donut) ----------------------------------------
    with col_donut:
        fat_pag = (
            df_raw.groupby("forma_pagamento")["total_venda"]
            .sum()
            .reset_index()
        )
        fig_donut = px.pie(
            fat_pag,
            names="forma_pagamento",
            values="total_venda",
            hole=0.52,
            color_discrete_sequence=PALETTE,
        )
        fig_donut.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="FORMAS DE PAGAMENTO", font=dict(size=11, color="#6b7a99"),
                       x=0.5, xanchor="center"),
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
            height=290,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # LINHA 3 – Filtros + tabela extra  |  Lucro por Loja
    # ════════════════════════════════════════════════════════════════════════
    col_filtros, col_loja = st.columns([4.5, 4.5])

    with col_filtros:
        st.markdown('<p class="filtros-label">FILTROS</p>', unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        with f1:
            data_min = df_raw["data_venda"].min().date()
            data_max = df_raw["data_venda"].max().date()
            data_ini = st.date_input("Start Date", value=data_min,
                                     min_value=data_min, max_value=data_max,
                                     label_visibility="visible")
        with f2:
            cats = ["Todas"] + sorted(df_raw["categoria"].unique().tolist())
            cat_sel = st.selectbox("Category", cats)

        # Tabela resultante do filtro
        df_f = aplicar_filtros(df_raw, data_ini, data_max, cat_sel)

        resumo = (
            df_f.groupby("produto")
            .agg(Qtd=("quantidade", "sum"),
                 Faturamento=("total_venda", "sum"),
                 Lucro=("lucro_venda", "sum"))
            .sort_values("Faturamento", ascending=False)
            .reset_index()
            .rename(columns={"produto": "Produto"})
        )
        resumo["Faturamento"] = resumo["Faturamento"].map("R$ {:,.0f}".format)
        resumo["Lucro"]       = resumo["Lucro"].map("R$ {:,.0f}".format)

        st.dataframe(resumo, use_container_width=True, hide_index=True, height=200)

    with col_loja:
        st.markdown('<p class="card-title">LUCRO POR LOJA</p>',
                    unsafe_allow_html=True)

        lucro_loja = (
            df_raw.groupby("loja")
            .agg(
                Faturamento=("total_venda",  "sum"),
                Custo=("custo_venda",        "sum"),
                Lucro=("lucro_venda",         "sum"),
                Vendas=("total_venda",        "count"),
            )
            .sort_values("Lucro", ascending=False)
            .reset_index()
            .rename(columns={"loja": "Loja"})
        )
        lucro_loja["Margem"] = (
            lucro_loja["Lucro"] / lucro_loja["Faturamento"] * 100
        ).map("{:.1f}%".format)
        lucro_loja["Faturamento"] = lucro_loja["Faturamento"].map("R$ {:,.0f}".format)
        lucro_loja["Lucro"]       = lucro_loja["Lucro"].map("R$ {:,.0f}".format)
        lucro_loja["Custo"]       = lucro_loja["Custo"].map("R$ {:,.0f}".format)

        st.dataframe(
            lucro_loja[["Loja", "Vendas", "Faturamento", "Lucro", "Margem"]],
            use_container_width=True,
            hide_index=True,
            height=230,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 – TABELA DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📋  Tabela de Dados":
    st.title("📋 Tabela de Dados Tratada")
    st.markdown("Visualização completa do dataset após tratamento ETL.")

    with st.expander("🔍 Filtros", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cats = ["Todas"] + sorted(df_raw["categoria"].unique().tolist())
            cat_sel = st.selectbox("Categoria", cats)
        with c2:
            lojas = ["Todas"] + sorted(df_raw["loja"].unique().tolist())
            loja_sel = st.selectbox("Loja", lojas)
        with c3:
            regioes = ["Todas"] + sorted(df_raw["regiao"].unique().tolist())
            reg_sel = st.selectbox("Região", regioes)
        with c4:
            d_min = df_raw["data_venda"].min().date()
            d_max = df_raw["data_venda"].max().date()
            intervalo = st.date_input("Período", value=(d_min, d_max),
                                      min_value=d_min, max_value=d_max)

    df_f = df_raw.copy()
    if cat_sel  != "Todas": df_f = df_f[df_f["categoria"] == cat_sel]
    if loja_sel != "Todas": df_f = df_f[df_f["loja"]     == loja_sel]
    if reg_sel  != "Todas": df_f = df_f[df_f["regiao"]   == reg_sel]
    if len(intervalo) == 2:
        df_f = df_f[(df_f["data_venda"] >= pd.Timestamp(intervalo[0])) &
                    (df_f["data_venda"] <= pd.Timestamp(intervalo[1]))]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros",        f"{len(df_f):,}")
    m2.metric("Faturamento",      f"R$ {df_f['total_venda'].sum():,.0f}")
    m3.metric("Lucro",            f"R$ {df_f['lucro_venda'].sum():,.0f}")
    m4.metric("Ticket Médio",     f"R$ {df_f['total_venda'].mean():,.2f}")

    st.markdown("---")

    cols_show = [
        "data_venda","cliente","loja","produto","categoria",
        "quantidade","preco_unitario","total_venda",
        "custo_venda","lucro_venda","forma_pagamento",
        "cidade","estado","regiao",
    ]
    df_ex = df_f[cols_show].copy()
    df_ex["data_venda"] = df_ex["data_venda"].dt.strftime("%d/%m/%Y")

    st.dataframe(
        df_ex.style.format({
            "preco_unitario": "R$ {:,.2f}",
            "total_venda":    "R$ {:,.2f}",
            "custo_venda":    "R$ {:,.2f}",
            "lucro_venda":    "R$ {:,.2f}",
        }),
        use_container_width=True, height=520,
    )
    st.caption(f"Exibindo {len(df_f):,} de {len(df_raw):,} registros.")

    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar seleção como CSV", data=csv,
                       file_name="vendas_filtrado.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 – ANÁLISE DE PRODUTOS  (placeholder)
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📦  Análise de Produtos":
    st.title("📦 Análise de Produtos")
    st.markdown(
        '<div class="placeholder">🚧 Em construção — '
        'Top produtos, faturamento por categoria e análise de margem.</div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 – ANÁLISE POR LOJA  (placeholder)
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🏪  Análise por Loja":
    st.title("🏪 Análise por Loja")
    st.markdown(
        '<div class="placeholder">🚧 Em construção — '
        'lucro por loja, mapa regional e ranking de lojas.</div>',
        unsafe_allow_html=True)
