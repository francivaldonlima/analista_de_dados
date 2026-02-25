# 📊 AppVendas — Dashboard de Vendas com Streamlit

Projeto de análise de dados de vendas com pipeline ETL em Jupyter Notebook e dashboard interativo construído com Streamlit.

---

## 🗂️ Estrutura do Projeto

```
appvendas/
├── etlVendas.ipynb                  # Notebook ETL — tratamento e análise
├── dashboard.py                     # Dashboard Streamlit (4 páginas)
├── appVendas.py                     # Exercício base do dashboard
├── appSimples.py                    # App Streamlit de exemplo
├── vendas_5000_linhas.csv           # Dataset bruto (5.000 registros)
└── vendas_5000_linhas_tratado.csv   # Dataset tratado (gerado pelo ETL)
```

---

## ⚙️ Pré-requisitos

- Python 3.10 ou superior
- pip

Instale as dependências necessárias:

```bash
pip install pandas streamlit plotly matplotlib jupyter
```

---

## 📓 Passo 1 — Executar o Notebook ETL (`etlVendas.ipynb`)

O notebook realiza todo o tratamento e enriquecimento do dataset bruto, gerando o arquivo `vendas_5000_linhas_tratado.csv` usado pelo dashboard.

### Abrindo no Jupyter Notebook

```bash
# Na pasta do projeto
cd atividades/appvendas

# Inicia o servidor Jupyter
jupyter notebook
```

Após abrir o navegador, clique em **`etlVendas.ipynb`** para abrir o notebook.

### Abrindo no VS Code

1. Abra o arquivo `etlVendas.ipynb` no VS Code
2. Clique em **"Select Kernel"** (canto superior direito)
3. Escolha o interpretador Python instalado
4. Clique em **"Run All"** (▶▶) para executar todas as células

### O que o notebook faz

| Etapa | Descrição |
|-------|-----------|
| 1️⃣ Carregamento | Lê `vendas_5000_linhas.csv` e exibe diagnóstico inicial |
| 2️⃣ Correção de tipos | Converte datas, padroniza strings e corrige a coluna `regiao` (valores com case inconsistente) |
| 3️⃣ Total da Venda | `preco_unitario × quantidade` |
| 4️⃣ Imposto Pago | Acessórios: 10% · Eletrônicos: 7,5% · Móveis: 5% |
| 5️⃣ Taxa de Pagamento | Crédito/Débito: 5% · Boleto: 2,5% · Pix: 0% |
| 6️⃣ Custo da Venda | `custo_produtos + imposto_pago + taxa_pagamento` |
| 7️⃣ Lucro da Venda | `total_venda − custo_venda` |
| 8️⃣ Exportação | Salva `vendas_5000_linhas_tratado.csv` |

> **Importante:** execute o notebook antes de iniciar o dashboard para garantir que o arquivo tratado esteja atualizado.

---

## 🚀 Passo 2 — Executar o Dashboard Streamlit

Certifique-se de estar na pasta do projeto e de que o arquivo `vendas_5000_linhas_tratado.csv` já foi gerado.

```bash
cd atividades/appvendas
python -m streamlit run dashboard.py
```

O dashboard abrirá automaticamente no navegador em `http://localhost:8501`.

> **Dica Windows:** use `python -m streamlit` caso o comando `streamlit` não seja reconhecido pelo terminal.

---

## 📋 Páginas do Dashboard

### 🏠 Visão Geral *(página inicial)*

| Componente | Descrição |
|---|---|
| Venda por Período | Gráfico de linha com faturamento mensal (2023–2024) |
| KPI Lucro Líquido | Total acumulado com variação % ano a ano |
| KPI Lucro Total | Margem percentual sobre o faturamento |
| KPI Total Vendas | Número de transações registradas |
| Top 8 Produtos | Tabela com os 8 produtos de maior faturamento |
| Por Produto | Gráfico de barras com faturamento por produto |
| Formas de Pagamento | Gráfico donut com distribuição por forma de pagamento |
| Filtros | Filtro por data inicial e categoria — atualiza tabela de resumo |
| Lucro por Loja | Tabela com faturamento, lucro e margem por loja |

### 📋 Tabela de Dados

Exibe o dataset completo tratado com filtros por **Categoria**, **Loja**, **Região** e **Período**. Permite baixar a seleção como CSV.

### 📦 Análise de Produtos *(em construção)*

Análise detalhada por produto — em desenvolvimento.

### 🏪 Análise por Loja *(em construção)*

Análise de desempenho por loja e região — em desenvolvimento.

---

## 📊 Sobre o Dataset

| Atributo | Valor |
|---|---|
| Registros | 5.000 |
| Período | Janeiro/2023 – Dezembro/2024 |
| Lojas | Loja Norte, Sul, Leste, Oeste, Centro |
| Categorias | Eletrônicos, Móveis, Acessórios |
| Regiões | Sudeste, Sul, Nordeste, Centro-Oeste |
| Formas de pagamento | Boleto, Pix, Crédito, Débito |

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e tratamento dos dados |
| `streamlit` | Interface do dashboard web |
| `plotly` | Gráficos interativos |
| `matplotlib` | Gráficos no notebook |
| `jupyter` | Ambiente de execução do ETL |
