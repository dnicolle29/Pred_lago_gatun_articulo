# ==========================================================
# PROYECTO:
# Predicción del Nivel del Lago Gatún mediante Machine Learning
# Autor: Daniela Quirós, Ana de Hoyos, M-ilagros Alonzo, Javier Chong y Nobel de Gracia.
# Universidad Tecnológica de Panamá
# ==========================================================

# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# CARGAR DATASET DE HIDROLOGÍA
# ==========================================================

print("="*70)
print("CARGANDO DATASET DE HIDROLOGÍA")
print("="*70)

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

print("Hidrología cargada correctamente.")
print("Registros:", len(hidrologia))


# ==========================================================
# CARGAR DATASET DE PRECIPITACIÓN
# ==========================================================

print("\n" + "="*70)
print("CARGANDO DATASET DE PRECIPITACIÓN")
print("="*70)

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

print("Precipitación cargada correctamente.")
print("Registros:", len(precipitacion))


# ==========================================================
# CARGAR DATASET ONI
# ==========================================================

print("\n" + "="*70)
print("CARGANDO DATASET ONI")
print("="*70)

oni = pd.read_csv("ONI.csv")

print("ONI cargado correctamente.")
print("Registros:", len(oni))


# ==========================================================
# CONVERTIR FECHAS
# ==========================================================

hidrologia["Fecha_Inicio"] = pd.to_datetime(hidrologia["Fecha_Inicio"])
hidrologia["Fecha_Fin"] = pd.to_datetime(hidrologia["Fecha_Fin"])

precipitacion["Fecha_Inicio"] = pd.to_datetime(
    precipitacion["Fecha_Inicio"]
)

precipitacion["Fecha_Fin"] = pd.to_datetime(
    precipitacion["Fecha_Fin"]
)

print("\nFechas convertidas correctamente.")


# ==========================================================
# PREPARAR DATASET ONI
# ==========================================================

meses = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12
}

oni["Mes"] = oni["SEAS"].map(meses)

oni["Fecha"] = pd.to_datetime(
    dict(
        year=oni["YR"],
        month=oni["Mes"],
        day=14
    )
)

oni.rename(
    columns={
        "ANOM": "ONI"
    },
    inplace=True
)

oni = oni[
    [
        "Fecha",
        "YR",
        "Mes",
        "ONI"
    ]
]

oni.rename(
    columns={
        "YR": "Año"
    },
    inplace=True
)

print("\nONI preparado correctamente.")

print("\nPrimeras filas del ONI:")
print(oni.head())

print("\nÚltimas filas del ONI:")
print(oni.tail())


# ==========================================================
# CREAR COLUMNAS DE AÑO Y MES
# ==========================================================

hidrologia["Año"] = hidrologia["Fecha_Inicio"].dt.year
hidrologia["Mes"] = hidrologia["Fecha_Inicio"].dt.month

precipitacion["Año"] = precipitacion["Fecha_Inicio"].dt.year
precipitacion["Mes"] = precipitacion["Fecha_Inicio"].dt.month

print("\nColumnas Año y Mes creadas correctamente.")

# ==========================================================
# UNIR LOS DATASETS
# ==========================================================

print("\n" + "="*70)
print("UNIENDO LOS DATASETS")
print("="*70)

# Unir Hidrología + Precipitación

dataset = pd.merge(
    hidrologia,
    precipitacion,
    on=["Fecha_Inicio", "Año", "Mes"],
    how="inner",
    suffixes=("_hidro", "_preci")
)

# Unir con ONI

dataset = pd.merge(
    dataset,
    oni[["Año", "Mes", "ONI"]],
    on=["Año", "Mes"],
    how="left"
)

# ==========================================================
# ELIMINAR COLUMNAS DUPLICADAS
# ==========================================================

dataset.drop(
    columns=[
        "Fecha_Fin_preci",
        "Año",
        "Mes"
    ],
    inplace=True
)

dataset.rename(
    columns={
        "Fecha_Fin_hidro": "Fecha_Fin"
    },
    inplace=True)

print("\nDatasets unidos correctamente.")

# ==========================================================
# INFORMACIÓN GENERAL
# ==========================================================

print("\nInformación del dataset:")

print(dataset.info())

print("\nPrimeras 10 filas")

print(dataset.head(10))

print("\nÚltimas 10 filas")

print(dataset.tail(10))

print("\nDimensiones del dataset:")

print(dataset.shape)

print("\nCantidad de registros:")

print(len(dataset))

print("\nValores faltantes:")

print(dataset.isnull().sum())
# ==========================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ==========================================================

print("\n" + "="*70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*70)

print(dataset.describe())

print("\nDescripción completa")

print(dataset.describe(include="all"))
# ==========================================================
# EXPORTAR DATASET UNIFICADO
# ==========================================================

dataset.to_csv(
    "Dataset_Unificado.csv",
    index=False
)

print("\nDataset_Unificado.csv guardado correctamente.")

# ==========================================================
# LIMPIEZA DEL DATASET
# ==========================================================

print("\n" + "="*70)
print("LIMPIEZA DEL DATASET")
print("="*70)

print("\nValores faltantes antes de la limpieza:\n")
print(dataset.isnull().sum())


# ==========================================================
# VARIABLES NUMÉRICAS
# ==========================================================

columnas_numericas = [
    "Aporte_CHCP",
    "Aporte_Gatun",
    "Aporte_Alhajuela",
    "Evaporacion",
    "Nivel_Lago_Gatun",
    "ACL",
    "ESC",
    "GAM",
    "GAT",
    "GTW",
    "ONI"
]


# ==========================================================
# IMPUTACIÓN CON LA MEDIA
# ==========================================================

imputador = SimpleImputer(strategy="mean")

dataset[columnas_numericas] = imputador.fit_transform(
    dataset[columnas_numericas]
)

print("\nValores faltantes después de imputar:\n")
print(dataset.isnull().sum())

# ==========================================================
# CREAR VARIABLES TEMPORALES
# ==========================================================

dataset["Año"] = dataset["Fecha_Inicio"].dt.year
dataset["Mes"] = dataset["Fecha_Inicio"].dt.month
dataset["Dia"] = dataset["Fecha_Inicio"].dt.day

print("\nVariables temporales creadas correctamente.")

# ==========================================================
# ELIMINAR COLUMNAS NO NECESARIAS
# ==========================================================

dataset_modelo = dataset.copy()

dataset_modelo.drop(
    columns=[
        "Fecha_Inicio",
        "Fecha_Fin"
    ],
    inplace=True
)

print("\nColumnas eliminadas.")

print(dataset_modelo.head())
# ==========================================================
# GUARDAR DATASET LIMPIO
# ==========================================================

dataset_modelo.to_csv(
    "Dataset_Final_Lago_Gatun.csv",
    index=False
)

print("\nDataset_Final_Lago_Gatun.csv creado correctamente.")


# ==========================================================
# INFORMACIÓN FINAL
# ==========================================================

print("\n" + "="*70)
print("DATASET FINAL")
print("="*70)

print(dataset_modelo.info())

print("\nPrimeras filas")
print(dataset_modelo.head())

print("\nÚltimas filas")
print(dataset_modelo.tail())

print("\nCantidad de registros")
print(len(dataset_modelo))

print("\nCantidad de columnas")
print(dataset_modelo.shape[1])

# ==========================================================
# PREPARACIÓN PARA MACHINE LEARNING
# ==========================================================

print("\n" + "="*70)
print("PREPARANDO EL MODELO")
print("="*70)

# Variable objetivo

y = dataset_modelo["Nivel_Lago_Gatun"]

# Variables predictoras

X = dataset_modelo.drop(
    columns=["Nivel_Lago_Gatun"]
)

print("\nVariables predictoras:")

print(X.columns)

print("\nCantidad de variables:", X.shape[1])

print("\nVariable objetivo:")

print(y.name)

print("\nPrimeras filas de X")

print(X.head())

print("\nPrimeros valores de y")

print(y.head())
# ==========================================================
# DIVISIÓN DEL DATASET
# ==========================================================

print("\n" + "="*70)
print("DIVISIÓN TRAIN / TEST")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Entrenamiento:", X_train.shape)

print("Prueba:", X_test.shape)
# ==========================================================
# RANDOM FOREST REGRESSOR
# ==========================================================

print("\n" + "="*70)
print("ENTRENANDO RANDOM FOREST")
print("="*70)

modelo = RandomForestRegressor(

    n_estimators=300,

    random_state=42,

    max_depth=10

)

modelo.fit(

    X_train,

    y_train

)

print("Modelo entrenado correctamente.")

# ==========================================================
# PREDICCIONES
# ==========================================================

predicciones = modelo.predict(X_test)

print("\nPrimeras predicciones:")

print(predicciones[:10])

# ==========================================================
# EVALUACIÓN DEL MODELO
# ==========================================================

print("\n" + "="*70)
print("RESULTADOS DEL MODELO")
print("="*70)

mae = mean_absolute_error(

    y_test,

    predicciones

)

rmse = np.sqrt(

    mean_squared_error(

        y_test,

        predicciones

    )

)

r2 = r2_score(

    y_test,

    predicciones

)

print("MAE :", mae)

print("RMSE:", rmse)

print("R²  :", r2)

# ==========================================================
# EXPORTAR PREDICCIONES
# ==========================================================

resultado = pd.DataFrame({

    "Valor_Real": y_test,

    "Prediccion": predicciones

})

resultado.to_csv(

    "Predicciones.csv",

    index=False

)

print("\nPredicciones.csv guardado correctamente.")
# ==========================================================
# IMPORTANCIA DE VARIABLES
# ==========================================================

print("\n" + "="*70)
print("IMPORTANCIA DE LAS VARIABLES")
print("="*70)

importancia = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": modelo.feature_importances_
})

importancia = importancia.sort_values(
    by="Importancia",
    ascending=False
)

print(importancia)

plt.figure(figsize=(10,6))

plt.bar(
    importancia["Variable"],
    importancia["Importancia"]
)

plt.xticks(rotation=45)

plt.title("Importancia de Variables")

plt.xlabel("Variables")

plt.ylabel("Importancia")

plt.tight_layout()

plt.show()
# ==========================================================
# REAL VS PREDICCIÓN
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values,
    marker="o",
    label="Valor Real"
)

plt.plot(
    predicciones,
    marker="s",
    label="Predicción"
)

plt.title("Nivel del Lago Gatún")

plt.xlabel("Observaciones")

plt.ylabel("Nivel")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
# ==========================================================
# DISPERSIÓN
# ==========================================================

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    predicciones
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Valor Real")

plt.ylabel("Predicción")

plt.title("Real vs Predicción")

plt.grid(True)

plt.tight_layout()

plt.show()
# ==========================================================
# HISTOGRAMA DE ERRORES
# ==========================================================

errores = y_test - predicciones

plt.figure(figsize=(8,5))

plt.hist(
    errores,
    bins=15
)

plt.title("Distribución del Error")

plt.xlabel("Error")

plt.ylabel("Frecuencia")

plt.grid(True)

plt.tight_layout()

plt.show()
# ==========================================================
# EXPORTAR IMPORTANCIA
# ==========================================================

importancia.to_csv(
    "Importancia_Variables.csv",
    index=False
)

print("\nImportancia_Variables.csv guardado correctamente.")
# ==========================================================
# FIN DEL PROYECTO
# ==========================================================

print("\n" + "="*70)
print("PROYECTO FINALIZADO")
print("="*70)

print("""
Archivos generados:

✔ Dataset_Unificado.csv
✔ Dataset_Final_Lago_Gatun.csv
✔ Predicciones.csv
✔ Importancia_Variables.csv

Gráficas generadas:

✔ Histograma de variables
✔ Matriz de correlación
✔ Importancia de variables
✔ Real vs Predicción
✔ Dispersión
✔ Histograma de errores

Modelo utilizado:

✔ Random Forest Regressor

Evaluación:

✔ MAE
✔ RMSE
✔ R²
""")
# ==========================================================
# PREDICCIÓN DE UN NUEVO CASO
# ==========================================================

print("\n" + "="*70)
print("SISTEMA DE PREDICCIÓN DEL NIVEL DEL LAGO GATÚN")
print("="*70)

print("1. Predicción usando el último registro del dataset")
print("2. Ingresar datos manualmente")
print("3. Salir")

opcion = input("\nSeleccione una opción: ")

# ==========================================================
# OPCIÓN 1
# ==========================================================

if opcion == "1":

    print("\n" + "="*70)
    print("PREDICCIÓN UTILIZANDO EL ÚLTIMO REGISTRO DEL DATASET")
    print("="*70)

    ultimo = dataset.iloc[[-1]]

    X_ultimo = ultimo[[
        "Aporte_CHCP",
        "Aporte_Gatun",
        "Aporte_Alhajuela",
        "Evaporacion",
        "ACL",
        "ESC",
        "GAM",
        "GAT",
        "GTW",
        "ONI",
        "Año",
        "Mes",
        "Dia"
    ]]

    nivel_predicho = modelo.predict(X_ultimo)[0]

    nivel_real = ultimo["Nivel_Lago_Gatun"].values[0]

    print(f"\nFecha: {ultimo['Fecha_Inicio'].values[0]}")
    print(f"Nivel real: {nivel_real:.2f} metros")
    print(f"Nivel predicho: {nivel_predicho:.2f} metros")

    error = abs(nivel_real - nivel_predicho)

    print(f"Error: {error:.2f} metros")

    if nivel_predicho < 25:
        estado = "BAJO"
    elif nivel_predicho < 26:
        estado = "NORMAL"
    elif nivel_predicho < 27:
        estado = "ALTO"
    else:
        estado = "MUY ALTO"

    print(f"Estado del lago: {estado}")

# ==========================================================
# OPCIÓN 2
# ==========================================================

elif opcion == "2":

    print("\n" + "="*70)
    print("PREDICCIÓN MANUAL")
    print("="*70)

    aporte_chcp = float(input("Aporte CHCP: "))
    aporte_gatun = float(input("Aporte Gatún: "))
    aporte_alhajuela = float(input("Aporte Alhajuela: "))
    evaporacion = float(input("Evaporación: "))

    acl = float(input("Precipitación ACL: "))
    esc = float(input("Precipitación ESC: "))
    gam = float(input("Precipitación GAM: "))
    gat = float(input("Precipitación GAT: "))
    gtw = float(input("Precipitación GTW: "))

    oni = float(input("Índice ONI: "))

    año = int(input("Año: "))
    mes = int(input("Mes: "))
    dia = int(input("Día: "))

    nuevo = pd.DataFrame({
        "Aporte_CHCP": [aporte_chcp],
        "Aporte_Gatun": [aporte_gatun],
        "Aporte_Alhajuela": [aporte_alhajuela],
        "Evaporacion": [evaporacion],
        "ACL": [acl],
        "ESC": [esc],
        "GAM": [gam],
        "GAT": [gat],
        "GTW": [gtw],
        "ONI": [oni],
        "Año": [año],
        "Mes": [mes],
        "Dia": [dia]
    })

    nivel = modelo.predict(nuevo)[0]

    print("\n" + "="*70)
    print("RESULTADO DE LA PREDICCIÓN")
    print("="*70)

    print(f"\nNivel estimado del Lago Gatún: {nivel:.2f} metros")

    if nivel < 25:
        estado = "BAJO"
    elif nivel < 26:
        estado = "NORMAL"
    elif nivel < 27:
        estado = "ALTO"
    else:
        estado = "MUY ALTO"

    print(f"Estado del lago: {estado}")

# ==========================================================
# OPCIÓN 3
# ==========================================================

elif opcion == "3":

    print("\nGracias por utilizar el sistema.")

# ==========================================================
# OPCIÓN INVÁLIDA
# ==========================================================

else:

    print("\nOpción no válida.")