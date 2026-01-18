import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince

sns.set(style="whitegrid")

# =====================================================
# 1️⃣ Création automatique du dataset
# =====================================================
def create_dataset(file_name="Housing_dataset.csv", n=500, random_seed=42):
    if not os.path.exists(file_name):
        print("⚠️ Dataset non trouvé → création en cours...")
        np.random.seed(random_seed)

        data = {
            "PRICE": np.random.randint(50000, 1000000, n),
            "BEDS": np.random.randint(1, 6, n),
            "BATH": np.random.randint(1, 5, n),
            "PROPERTYSQFT": np.random.randint(400, 5000, n),
            "LATITUDE": np.random.uniform(33.5, 35.5, n),
            "LONGITUDE": np.random.uniform(-8.5, -5.0, n),
            "TYPE": np.random.choice(["Apartment", "House", "Villa", "Studio"], n),
            "LOCALITY": np.random.choice(["Casablanca", "Rabat", "Marrakech", "Tanger", "Agadir"], n),
            "BROKERTITLE": np.random.choice(["Agent A", "Agent B", "Agent C"], n),
            "MAIN_ADDRESS": np.random.choice(["Address 1", "Address 2", "Address 3"], n),
            "FORMATTED_ADDRESS": np.random.choice(["Formatted 1", "Formatted 2"], n)
        }

        pd.DataFrame(data).to_csv(file_name, index=False)
        print("✅ Dataset créé avec succès")

# =====================================================
# 2️⃣ Chargement du dataset
# =====================================================
def load_dataset(file_name="Housing_dataset.csv"):
    return pd.read_csv(file_name)

# =====================================================
# 3️⃣ Exploration initiale
# =====================================================
def explore_data(df):
    print("\n📌 Dimensions :", df.shape)
    df.info()

    print("\n📌 Valeurs manquantes :")
    print(df.isnull().sum())

    print("\n📌 Doublons :", df.duplicated().sum())

    plt.figure(figsize=(10,4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Heatmap des valeurs manquantes")
    plt.show()

# =====================================================
# 4️⃣ Prétraitement
# =====================================================
def prepare_data(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    n_before = df.shape[0]
    df = df.drop_duplicates()
    print(f"✅ {n_before - df.shape[0]} doublons supprimés")

    df = df.drop(columns=["BROKERTITLE", "MAIN_ADDRESS", "FORMATTED_ADDRESS"], errors="ignore")

    return df

# =====================================================
# 5️⃣ Analyse univariée
# =====================================================
def stats_plot(df, variable, bins=30):
    print(df[variable].describe())

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.histplot(df[variable], bins=bins, kde=True, ax=axes[0])
    sns.boxplot(x=df[variable], ax=axes[1])
    plt.show()

def univariate_analysis(df):
    for col in df.select_dtypes(include=["int64","float64"]).columns:
        stats_plot(df, col)

    plt.figure(figsize=(8,4))
    sns.countplot(x="TYPE", data=df)
    plt.title("Répartition des types de logements")
    plt.show()

# =====================================================
# 6️⃣ Analyse multivariée
# =====================================================
def multivariate_analysis(df):
    num_df = df.select_dtypes(include=["int64","float64"])

    plt.figure(figsize=(10,8))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.show()

    sns.pairplot(num_df)
    plt.show()

# =====================================================
# 7️⃣ Analyse LOCALITY (Heatmap + Pairplot)
# =====================================================
def locality_analysis(df):
    housing_locality = pd.get_dummies(df, columns=["LOCALITY"], drop_first=True)

    num_df = housing_locality.select_dtypes(include=["int64","float64","uint8"])

    plt.figure(figsize=(12,10))
    sns.heatmap(num_df.corr(), cmap="coolwarm")
    plt.title("Corrélation après encodage de LOCALITY")
    plt.show()

    sns.pairplot(num_df)
    plt.suptitle("Pairplot après encodage de LOCALITY", y=1.02)
    plt.show()

# =====================================================
# 8️⃣ Encodage général
# =====================================================
def encode_data(df):
    return pd.get_dummies(df, drop_first=True)

# =====================================================
# 9️⃣ FAMD
# =====================================================
def perform_famd(df):
    famd = prince.FAMD(n_components=2, random_state=42)
    famd_result = famd.fit_transform(df)

    plt.figure(figsize=(7,7))
    plt.scatter(famd_result[0], famd_result[1], s=10)
    plt.title("Projection FAMD")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.show()

# =====================================================
# 🔟 Export
# =====================================================
def export_data(df):
    df.to_csv("data_housing_prepared.csv", index=False)
    print("✅ Dataset exporté")

# =====================================================
# MAIN
# =====================================================
def main():
    create_dataset()
    housing = load_dataset()

    explore_data(housing)
    housing = prepare_data(housing)

    univariate_analysis(housing)
    multivariate_analysis(housing)

    locality_analysis(housing)

    housing_encoded = encode_data(housing)
    perform_famd(housing)

    export_data(housing_encoded)

if __name__ == "__main__":
    main()
