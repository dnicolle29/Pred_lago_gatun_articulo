# 🌊 Predicción del Nivel del Lago Gatún mediante Machine Learning

> Proyecto desarrollado para la **Un Proyecto Final**, enfocado en la predicción del nivel del Lago Gatún utilizando técnicas de Machine Learning para apoyar la planificación logística del Canal de Panamá.

---

## 📖 Descripción

El Lago Gatún es una de las principales fuentes de agua para el funcionamiento del Canal de Panamá. Durante eventos climáticos como el **Fenómeno de El Niño**, la disminución de las precipitaciones provoca reducciones significativas en el nivel del lago, afectando el tránsito marítimo.

Este proyecto desarrolla un modelo predictivo basado en **Random Forest Regressor** para estimar el nivel del Lago Gatún utilizando información hidrológica y climática.

---

## 🎯 Objetivos

- Predecir el nivel del Lago Gatún.
- Integrar múltiples fuentes de datos climáticos e hidrológicos.
- Analizar la importancia de cada variable.
- Apoyar la toma de decisiones mediante predicciones anticipadas.

---

## 🧠 Variables utilizadas

- 🌧️ Precipitación
- 🌡️ Temperatura
- 💧 Evaporación
- 🌊 Aportes hídricos
- 🌎 Índice Oceánico de El Niño (ONI)

---

## 🤖 Modelo de Machine Learning

Se implementó un modelo de:

**Random Forest Regressor**

para estimar el nivel del Lago Gatún a partir de datos históricos previamente procesados.

---

# 📂 Estructura del proyecto

```text
Pred_lago_gatun_articulo/
│
├── 📄 proyecto.py
├── 📄 correlacion.py
├── 📄 oni.py
├── 📄 02_eda.py
│
├── 📊 Dataset_Final_Canal_Panama.csv
├── 📊 Dataset_Final_Lago_Gatun.csv
├── 📊 Dataset_Unificado.csv
├── 📊 ONI.csv
├── 📊 Predicciones.csv
├── 📊 Importancia_Variables.csv
│
├── 📁 valido_aportesmensuales/
├── 📁 valido_preci_and_temp/
│
└── 📄 README.md
```

---

## ⚙️ Tecnologías utilizadas

| Herramienta | Uso |
|-------------|-----|
| Python | Lenguaje principal |
| Pandas | Manipulación de datos |
| NumPy | Cálculos numéricos |
| Scikit-learn | Machine Learning |
| Matplotlib | Visualización |
| Git | Control de versiones |
| GitHub | Repositorio del proyecto |

---

## 🚀 Ejecución

1. Clonar el repositorio

```bash
git clone https://github.com/dnicolle29/Pred_lago_gatun_articulo.git
```

2. Entrar a la carpeta

```bash
cd Pred_lago_gatun_articulo
```

3. Instalar las dependencias

```bash
pip install pandas numpy matplotlib scikit-learn
```

4. Ejecutar el proyecto

```bash
python proyecto.py
```

---

## 📈 Resultados obtenidos

El proyecto genera:

- ✅ Dataset unificado
- ✅ Predicción del nivel del Lago Gatún
- ✅ Importancia de variables
- ✅ Archivos CSV con los resultados
- ✅ Análisis exploratorio de datos (EDA)

---

## 📚 Contexto

Este proyecto fue desarrollado como parte de una investigación orientada a aplicar técnicas de **Machine Learning** para apoyar la gestión sostenible del recurso hídrico del Canal de Panamá.

---



## 👩‍💻 Autores

Daniela Quirós, Ana De Hoyos, Nobel De Gracia, Milagros Alonzo y Javier Chong.

Universidad Tecnológica de Panamá

Licenciatura en Desarrollo de Software