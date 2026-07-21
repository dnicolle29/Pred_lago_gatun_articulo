import pandas as pd

# Leer el archivo de NOAA
oni = pd.read_csv(
    "oni.ascii.txt",
    sep=r"\s+",
    engine="python"
)

print("Primeras filas:")
print(oni.head())

print("\nInformación:")
print(oni.info())

# Guardar como CSV
oni.to_csv("ONI.csv", index=False, encoding="utf-8-sig")

print("\n✅ Archivo ONI.csv creado correctamente.")