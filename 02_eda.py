import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CARGAR DATASET FINAL
# ==========================================================

dataset = pd.read_csv(
    "Dataset_Final_Canal_Panama.csv",
    parse_dates=["Fecha_Inicio", "Fecha_Fin"]
)

# ==========================================================
# INFORMACIÓN GENERAL
# ==========================================================

print("=" * 70)
print("INFORMACIÓN GENERAL")
print("=" * 70)

print(dataset.info())

print("\nPrimeras filas:")
print(dataset.head())

# ==========================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ==========================================================

print("\n" + "=" * 70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 70)

print(dataset.describe())

# ==========================================================
# VALORES FALTANTES
# ==========================================================

print("\n" + "=" * 70)
print("VALORES FALTANTES")
print("=" * 70)

print(dataset.isnull().sum())

# ==========================================================
# HISTOGRAMA DE TODAS LAS VARIABLES NUMÉRICAS
# ==========================================================

dataset.hist(figsize=(15,10))

plt.tight_layout()
plt.show()

# ==========================================================
# EVOLUCIÓN DEL NIVEL DEL LAGO GATÚN
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(
    dataset["Fecha_Inicio"],
    dataset["Nivel_Lago_Gatun"]
)

plt.title("Nivel del Lago Gatún")
plt.xlabel("Fecha")
plt.ylabel("Nivel (m)")

plt.grid(True)

plt.show()

# ==========================================================
# MATRIZ DE CORRELACIÓN
# ==========================================================

correlacion = dataset.select_dtypes(include="number").corr()

print("\n" + "=" * 70)
print("MATRIZ DE CORRELACIÓN")
print("=" * 70)

print(correlacion)