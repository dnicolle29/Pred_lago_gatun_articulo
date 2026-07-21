# Predicción del Nivel del Lago Gatún mediante Machine Learning

## Descripción

Este proyecto desarrolla un modelo de Machine Learning para predecir el nivel del Lago Gatún utilizando variables hidrológicas y climáticas, con el objetivo de apoyar la planificación logística del Canal de Panamá durante eventos como el fenómeno de El Niño.

## Objetivo

Construir un modelo predictivo basado en Random Forest que estime el nivel del Lago Gatún a partir de información histórica de:

- Precipitación
- Temperatura
- Evaporación
- Aportes hídricos
- Índice Oceánico de El Niño (ONI)

## Tecnologías utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Estructura del proyecto

```
Pred_lago_gatun_articulo/
│
├── proyecto.py
├── correlacion.py
├── oni.py
├── 02_eda.py
│
├── Dataset_Final_Canal_Panama.csv
├── Dataset_Final_Lago_Gatun.csv
├── Dataset_Unificado.csv
├── ONI.csv
├── Predicciones.csv
├── Importancia_Variables.csv
│
├── valido_aportesmensuales/
├── valido_preci_and_temp/
│
└── README.md
```

## Modelo utilizado

El modelo implementado corresponde a un **Random Forest Regressor**, entrenado para estimar el nivel del Lago Gatún utilizando variables climáticas e hidrológicas previamente procesadas.

## Resultados

El proyecto genera:

- Predicciones del nivel del Lago Gatún.
- Importancia de variables.
- Dataset unificado para entrenamiento.
- Archivos CSV con los resultados obtenidos.

## Autor

Daniela Quirós, Ana De Hoyos, Nobel De Gracia, Milagros Alonzo y Javier Chong.

Universidad Tecnológica de Panamá

Licenciatura en Desarrollo de Software