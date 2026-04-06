# Projeto Integrador — Previsão de Pressão com Machine Learning (UTR-221)

## Visão Geral

Este projeto aplica Machine Learning para prever a pressão da **UTR-221** com 1 minuto de antecedência, usando dados históricos exportados de um sistema **SCADA** (Supervisory Control and Data Acquisition). O objetivo principal é simular o comportamento do sensor durante períodos de **falha de comunicação**, quando não há dados reais disponíveis.

---

## Estrutura da Pasta

```
PROJETO_INTEGRADOR/
├── urt220.xlsx              # Dados brutos do sistema SCADA — UTR-220 (85 variáveis)
├── urt221.xlsx              # Dados brutos do sistema SCADA — UTR-221 (9 variáveis)
├── tratar_dados.py          # Script de limpeza e preparação dos dados
├── treinar_modelo.py        # Script de treinamento do modelo Random Forest
├── UTR221_ML_validacao.py   # Script de validação científica e análise comparativa
├── dashboard_utr221.py      # Dashboard interativo com Streamlit
├── index.html               # Apresentação em slides (abrir no navegador)
├── apresentacao.html        # Apresentação auxiliar
└── LEIA-ME.txt              # Manual resumido do projeto
```

---

## Descrição dos Scripts

### 1. `tratar_dados.py` — Tratamento de Dados
Prepara os dados brutos do SCADA para o treinamento do modelo.

**Etapas:**
- Carrega os arquivos `urt220.xlsx` e `urt221.xlsx`
- Converte a coluna `E3TIMESTAMP` para formato datetime
- Marca como inválidos (`NaN`) os valores com `QUALITY = 0` (sinalizados pelo SCADA como não confiáveis)
- Regulariza a série temporal em intervalos de **1 minuto**
- Imputa valores ausentes:
  - Sensores analógicos (pressão, nível, vazão): **interpolação linear**
  - Status/bombas (valores binários): **forward-fill** (último valor conhecido)
- Gera relatório de diagnóstico (antes x depois)
- Exporta: `urt220_tratado.xlsx` e `urt221_tratado.xlsx`

---

### 2. `treinar_modelo.py` — Treinamento do Modelo
Treina um modelo **Random Forest** para prever a pressão da UTR-221 1 minuto à frente.

**O que é Random Forest?**
Algoritmo que cria 200 "árvores de decisão" e combina suas previsões. É robusto e funciona bem com dados reais e ruidosos de sistemas industriais.

**Artefatos gerados:**
| Arquivo | Conteúdo |
|---|---|
| `modelo_utr221.pkl` | Modelo treinado (Random Forest) |
| `historico_utr221.pkl` | Série histórica completa (seed do dashboard) |
| `meta_utr221.pkl` | Metadados: features, métricas, timestamps |

---

### 3. `UTR221_ML_validacao.py` — Validação Científica
Script de análise aprofundada do modelo. Vai além do treinamento simples.

**O que faz:**
- Treina **dois modelos** para comparação:
  - Modelo de **1 minuto** (curto prazo — referência)
  - Modelo de **30 minutos** (horizonte longo — mais desafiador)
- Compara a importância das features em cada horizonte
- Avalia métricas de acurácia
- Realiza previsão recursiva das próximas **7 horas (420 minutos)**
- Exporta resultados e gráficos

---

### 4. `dashboard_utr221.py` — Dashboard Interativo
Aplicação web criada com **Streamlit** para simular o comportamento da pressão durante falha de comunicação.

**Como funciona:**
1. O usuário define o período de falha (data, hora, duração)
2. O sistema carrega os últimos 60 min de dados reais como "semente"
3. O modelo Random Forest prevê a pressão **minuto a minuto** de forma recursiva
4. O gráfico exibe o período **antes (real)** e **durante (previsto)**
5. Se houver dados reais do período, exibe comparação lado a lado

---

## Métricas do Modelo (1 minuto à frente)

| Métrica | Valor |
|---|---|
| Acurácia | **99,71%** |
| MAE (Erro Médio Absoluto) | 0,0039 BAR |
| RMSE | 0,0255 BAR |
| R² | 0,8966 |
| MAPE | 0,29% |

---

## Como Executar

### Pré-requisitos
```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib openpyxl
```

### Ordem de execução

```bash
# 1. Tratar os dados (opcional — dados já tratados disponíveis)
python tratar_dados.py

# 2. Treinar o modelo (obrigatório antes do dashboard)
python treinar_modelo.py

# 3. Abrir o dashboard interativo
python -m streamlit run dashboard_utr221.py

# 4. Rodar análise científica (opcional)
python UTR221_ML_validacao.py
```

### Apresentação
Abra o arquivo `index.html` diretamente no navegador (duplo clique).

| Tecla | Ação |
|---|---|
| `→` ou `Espaço` | Próximo slide |
| `←` | Slide anterior |
| `F` | Tela cheia |
| `S` | Notas do apresentador |

---

## Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| Python 3.13 | Linguagem principal |
| Pandas | Manipulação de dados |
| NumPy | Operações matemáticas |
| Scikit-Learn | Algoritmo Random Forest |
| Matplotlib | Visualização de gráficos |
| Streamlit | Dashboard web interativo |
| Joblib | Serialização do modelo (.pkl) |
| OpenPyXL | Leitura/escrita de arquivos Excel |

---

## Dados de Entrada

| Arquivo | Variáveis | Período |
|---|---|---|
| `urt220.xlsx` | 85 colunas (pressão, nível, status de bombas, etc.) | Fev–Mar 2026 |
| `urt221.xlsx` | 9 colunas (nível, pressão, vazão) | Fev–Mar 2026 |

Os dados são exportados diretamente do sistema SCADA industrial, com leituras a cada **1 minuto**.
