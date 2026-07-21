import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CARGAR DATASET DE HIDROLOGÍA
# ==========================================================

hidrologia = pd.read_csv(
    r"valido_aportesmensuales\BulkExport-CHCP-A,GAT-20260709010854.csv",
    sep=";",
    skiprows=4,
    decimal=","
)

hidrologia.columns = [
    "Fecha_Inicio",
    "Fecha_Fin",
    "Aporte_CHCP",
    "Aporte_Gatun",
    "Aporte_Alhajuela",
    "Evaporacion",
    "Nivel_Lago_Gatun"
]

# ==========================================================
# CARGAR DATASET DE PRECIPITACIÓN
# ==========================================================

precipitacion = pd.read_csv(
    r"valido_preci_and_temp\BulkExport-ACL,ESC,GAM,GAT,GTW-20260709011402.csv",
    sep=";",
    skiprows=4,
    decimal=","
)

precipitacion.columns = [
    "Fecha_Inicio",
    "Fecha_Fin",
    "ACL",
    "ESC",
    "GAM",
    "GAT",
    "GTW"
]

# ==========================================================
# CARGAR ONI
# ==========================================================

oni = pd.read_csv("ONI.csv")

# ==========================================================
# CONVERTIR FECHAS
# ==========================================================

hidrologia["Fecha_Inicio"] = pd.to_datetime(hidrologia["Fecha_Inicio"])
precipitacion["Fecha_Inicio"] = pd.to_datetime(precipitacion["Fecha_Inicio"])

# ==========================================================
# PREPARAR ONI
# ==========================================================

meses = {
    "DJF":1,
    "JFM":2,
    "FMA":3,
    "MAM":4,
    "AMJ":5,
    "MJJ":6,
    "JJA":7,
    "JAS":8,
    "ASO":9,
    "SON":10,
    "OND":11,
    "NDJ":12
}

oni["Mes"] = oni["SEAS"].map(meses)

oni["Fecha"] = pd.to_datetime(
    dict(
        year=oni["YR"],
        month=oni["Mes"],
        day=14
    )
)

oni["Año"] = oni["Fecha"].dt.year
oni["Mes"] = oni["Fecha"].dt.month

hidrologia["Año"] = hidrologia["Fecha_Inicio"].dt.year
hidrologia["Mes"] = hidrologia["Fecha_Inicio"].dt.month

precipitacion["Año"] = precipitacion["Fecha_Inicio"].dt.year
precipitacion["Mes"] = precipitacion["Fecha_Inicio"].dt.month

# ==========================================================
# UNIR DATASETS
# ==========================================================

dataset = pd.merge(
    hidrologia,
    precipitacion,
    on=["Fecha_Inicio","Año","Mes"],
    how="inner",
    suffixes=("_hidro","_preci")
)

dataset = pd.merge(
    dataset,
    oni[["Año","Mes","ANOM"]],
    on=["Año","Mes"],
    how="left"
)

dataset.rename(columns={"ANOM":"ONI"}, inplace=True)

# ==========================================================
# ELIMINAR COLUMNAS QUE NO SE ANALIZAN
# ==========================================================

dataset = dataset.drop(columns=[
    "Fecha_Inicio",
    "Fecha_Fin_hidro",
    "Fecha_Fin_preci",
    "Año",
    "Mes"
])

# ==========================================================
# MATRIZ DE CORRELACIÓN
# ==========================================================

correlacion = dataset.corr(numeric_only=True)

print("="*70)
print("MATRIZ DE CORRELACIÓN")
print("="*70)
print(correlacion)

# ==========================================================
# MAPA DE CALOR
# ==========================================================

plt.figure(figsize=(10,8))

plt.imshow(correlacion, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlacion.columns)),
    correlacion.columns,
    rotation=90
)

plt.yticks(
    range(len(correlacion.columns)),
    correlacion.columns
)

plt.title("Matriz de Correlación")

plt.tight_layout()

plt.show()