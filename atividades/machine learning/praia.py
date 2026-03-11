import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1. CRIANDO O DATAFRAME
# ============================================================
dados = [
    [18, 90, 0], [20, 85, 0], [22, 80, 0], [25, 70, 1],
    [27, 65, 1], [30, 50, 1], [15, 95, 0], [28, 55, 1],
    [23, 75, 1], [19, 88, 0], [26, 60, 1], [21, 82, 0]
]

df = pd.DataFrame(dados, columns=["Temperatura", "Umidade", "Alvo"])
display(df)

# ============================================================
# 2. SEPARANDO DADOS PREDITIVOS E ALVO
# ============================================================
dados_preditivos = df[["Temperatura", "Umidade"]]
alvo = df["Alvo"]

# ============================================================
# 3. DIVIDINDO EM TREINO E TESTE
# ============================================================
dp_train, dp_teste, alvo_train, alvo_teste = train_test_split(
    dados_preditivos, alvo, test_size=0.3, random_state=42
)

print(f"Amostras para treino : {len(dp_train)}")
print(f"Amostras para teste  : {len(dp_teste)}")

# ============================================================
# 4. CRIANDO O MODELO KNN
# ============================================================
knn = KNeighborsClassifier(n_neighbors=3)

# ============================================================
# 5. TREINANDO O MODELO
# ============================================================
knn.fit(dp_train, alvo_train)

# ============================================================
# 6. VERIFICANDO A PRECISÃO
# ============================================================
alvo_previsto = knn.predict(dp_teste)

acuracia = accuracy_score(alvo_teste, alvo_previsto)
display(f"Precisão do KNN k=3: {acuracia * 100:.2f}%")

# ============================================================
# BÔNUS — Testando uma previsão manual
# ============================================================
# Temperatura=29°C, Umidade=52% → vai à praia?
nova_pessoa = [[29, 52]]
resultado = knn.predict(nova_pessoa)
display(f"Temperatura=29°C, Umidade=52% → {'Vai à praia 🏖️' if resultado[0] == 1 else 'Não vai à praia 🏠'}")