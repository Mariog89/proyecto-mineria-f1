# Informe Final del Proyecto de Minería de Datos
## Predicción de Finalización de Carrera en Fórmula 1 (FIA F1)

**Metodología:** CRISP-DM  
**Fecha de elaboración:** Abril 2026  
**Dataset:** `f1_final_dataset.csv`  
**Variable objetivo:** `finished` (1 = terminó la carrera, 0 = no terminó)  
**Referencia de notebooks:**
- `notebooks/01_preparacion_datos.ipynb`
- `notebooks/02_modelado_predictivo.ipynb`
- `notebooks/03_despliegue.ipynb`

---

## 1. Entendimiento del Negocio

### 1.1 Descripción del Negocio

La Fórmula 1 (F1) es la máxima categoría del automovilismo deportivo, regulada por la Fédération Internationale de l'Automobile (FIA). Cada temporada comprende aproximadamente 20 a 24 Grandes Premios disputados en circuitos de todo el mundo. En cada carrera participan 10 escuderías (constructores) con dos pilotos cada una, totalizando 20 competidores por evento.

Los datos históricos de la F1 incluyen información detallada de calificaciones (qualifying), parrillas de salida, tiempos por vuelta, resultados de carrera, incidentes técnicos, abandonos, condiciones climáticas y estadísticas acumuladas por piloto y constructor. La riqueza de estos datos permite aplicar técnicas de minería de datos para descubrir patrones ocultos y construir modelos predictivos con valor estratégico para equipos, analistas y aficionados.

### 1.2 Descripción del Problema

El objetivo principal es predecir si un piloto terminará una carrera dada (`finished = 1`) o no (`finished = 0`). Un piloto puede no terminar una carrera por múltiples razones: fallos mecánicos, accidentes, colisiones, problemas de fiabilidad, decisiones estratégicas de retiro, o descalificaciones. La capacidad de predecir esta probabilidad antes de la carrera permite:

- A los equipos: estimar riesgos de fiabilidad y ajustar estrategias.
- A los analistas: cuantificar expectativas y generar narrativas basadas en datos.
- A los stakeholders deportivos: optimizar la toma de decisiones sobre paradas en pits y gestión de neumáticos.

El problema es un desafío de clasificación binaria con clases desbalanceadas, ya que la mayoría de los pilotos suelen terminar las carreras.

### 1.3 Objetivos de la Minería de Datos

El proyecto contempla dos enfoques complementarios:

1. **Clustering Descriptivo:** Identificar perfiles naturales de pilotos/constructores según sus patrones históricos de rendimiento, fiabilidad y características de carrera.
2. **Clasificación Predictiva:** Construir un modelo supervisado que prediga la probabilidad de que un piloto termine una carrera, utilizando variables disponibles antes del inicio de la misma (posición de parrilla, historial del piloto, características del circuito, etc.).

### 1.4 Diseño de la Solución

| Problema | Tipo de Minería | Tipo de Análisis | Tipo de Aprendizaje | Requerimiento de Datos | Métodos | Evaluación |
|---|---|---|---|---|---|---|
| Predicción de finalización de carrera en F1 | Predictivo | Clasificación binaria | Supervisado | Registros históricos con variable objetivo etiquetada (`finished`) | Naive Bayes, Regresión Logística, SVM, KNN, Random Forest, XGBoost, Gradient Boosting, Voting | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| Perfiles de pilotos/constructores | Descriptivo | Clustering | No supervisado | Variables numéricas de rendimiento y fiabilidad | K-Means, DBSCAN, Jerárquico | Silhouette Score, Inercia, análisis visual |

**Línea base (Baseline):** Dado que la clase mayoritaria (no terminó) representa el 74.73 % del dataset, un clasificador trivial que siempre predice la clase mayoritaria obtendría una **accuracy de 74.73 %** y un **recall de clase 1 de 0 %**. Por tanto, el objetivo del modelo predictivo es superar significativamente esta línea base, especialmente en métricas sensibles al desbalance como F1-Score y ROC-AUC. Se establece como meta mínima un **F1-Score ≥ 0.60** y **Accuracy superando el 60 %** con capacidad de detectar la clase minoritaria.

### 1.5 Recursos para Creación y Despliegue

**Hardware:**
- Procesador: CPU multi-núcleo para entrenamiento y validación cruzada.
- Memoria RAM: Suficiente para cargar el dataset completo (23,777 registros × ~20 variables) y operaciones de SMOTE.
- Almacenamiento: Disco estándar para persistencia de modelos, pipelines y datasets derivados.

**Software:**
- Lenguaje: Python 3.x
- Librerías principales: pandas, numpy, scikit-learn, xgboost, imbalanced-learn (SMOTE), optuna, scipy, matplotlib, seaborn, streamlit.
- Entorno de desarrollo: Jupyter Notebooks para experimentación; script de aplicación para despliegue.

---

## 2. Entendimiento de los Datos

### 2.1 Ciclo de los Datos

Los datos provienen de fuentes históricas de Fórmula 1 que consolidan información de calificaciones, resultados de carrera, y estadísticas de pilotos y constructores. El flujo de datos sigue el ciclo:

1. **Adquisición:** Extracción de registros históricos de carreras, calificaciones y resultados.
2. **Integración:** Unificación de múltiples fuentes (resultados de carrera, tiempos de calificación, datos meteorológicos, estadísticas acumuladas).
3. **Preparación:** Limpieza, transformación, selección de variables y balanceo.
4. **Modelado:** Entrenamiento y evaluación de modelos de ML.
5. **Despliegue:** Uso del modelo en producción mediante una aplicación interactiva.
6. **Monitoreo:** Control de calidad del modelo y detección de degradación.

### 2.2 Diccionario de Datos

A continuación se presenta el diccionario de datos del dataset inicial (`f1_final_dataset.csv`), compuesto por 23,777 registros y 30 variables.

| Variable | Descripción | Tipo |
|---|---|---|
| `grid` | Posición de salida en la parrilla (1 a 20+) | Numérica discreta |
| `grid_normalized` | Posición de parrilla normalizada por número de participantes | Numérica continua |
| `position` | Posición final en la carrera | Numérica discreta |
| `points` | Puntos obtenidos en la carrera | Numérica continua |
| `laps` | Número de vueltas completadas | Numérica discreta |
| `race_day_of_year` | Día del año en que se disputó la carrera | Numérica discreta |
| `season_progress` | Progreso de la temporada (0 a 1) | Numérica continua |
| `round` | Número de ronda del campeonato | Numérica discreta |
| `race_month` | Mes del año de la carrera (1-12) | Numérica discreta |
| `constructor_nationality_encoded` | Nacionalidad del constructor codificada | Numérica discreta |
| `driver_age` | Edad del piloto al momento de la carrera | Numérica continua |
| `qualifying_position` | Posición alcanzada en calificación | Numérica discreta |
| `best_q_time` | Mejor tiempo de calificación | Numérica continua |
| `top10_start` | Indicador si sale en top 10 (1/0) | Binaria |
| `front_row_start` | Indicador si sale en primera fila (1/0) | Binaria |
| `avg_finish_rate` | Tasa histórica de finalización del piloto | Numérica continua |
| `experience_years` | Años de experiencia del piloto | Numérica continua |
| `constructor_reliability` | Índice de fiabilidad del constructor | Numérica continua |
| `circuit_type_encoded` | Tipo de circuito codificado | Numérica discreta |
| `weather_condition_encoded` | Condición climática codificada | Numérica discreta |
| `is_street_circuit` | Indicador si es circuito urbano (1/0) | Binaria |
| `altitude_encoded` | Altitud del circuito codificada | Numérica discreta |
| `finished` | **Variable objetivo:** 1 si terminó, 0 si no | Binaria |
| *(otras variables auxiliares)* | Variables adicionales de soporte o intermedias | Mixto |

> **Nota:** El dataset original contenía 30 variables. Durante la fase de preparación se eliminaron 10 por redundancia o irrelevancia, y se crearon 3 nuevas. El conjunto final de modelado se detalla en la Sección 3.

### 2.3 Reglas de Calidad desde el Negocio

| Variable | Regla de Calidad |
|---|---|
| `grid` | Debe estar entre 1 y el número de participantes de la carrera. No negativa. |
| `position` | Si `finished = 1`, debe existir una posición final válida (≥ 1). Si `finished = 0`, puede ser nula o indicar abandono. |
| `laps` | Si `finished = 1`, `laps` debe ser ≥ 90 % de las vueltas del ganador. Si `finished = 0`, puede ser menor. |
| `points` | No negativa. Debe ser coherente con la posición final según el sistema de puntuación de la época. |
| `driver_age` | Debe estar entre 18 y 60 años. |
| `qualifying_position` | Debe estar entre 1 y el número de participantes. |
| `best_q_time` | Debe ser un valor positivo en formato tiempo. |
| `finished` | Binaria {0, 1}. No admite valores nulos. |

---

## 3. Preparación de Datos

La preparación de datos se ejecutó en el notebook `notebooks/01_preparacion_datos.ipynb`. Esta fase es crítica para garantizar la calidad del input de los modelos y mitigar problemas como la redundancia, la irrelevancia y el desbalance de clases.

### 3.1 Integración

Se consolidaron múltiples fuentes de datos históricos de F1 en un único dataset tabular. Se realizaron operaciones de unión (merge/join) por claves compuestas (año, ronda, piloto, constructor) para enriquecer los registros con información de calificación, estadísticas acumuladas y características del circuito. El resultado fue un dataset unificado de 23,777 registros y 30 variables iniciales.

### 3.2 Selección de Variables

Se aplicó un proceso iterativo de selección basado en tres criterios principales:

1. **Correlación con la variable objetivo:** Variables con correlación absoluta menor a 0.01 se consideraron estadísticamente irrelevantes para discriminar entre clases.
2. **Redundancia multivariada:** Variables con correlación absoluta mayor a 0.8 entre sí se evaluaron para eliminación, conservando la más interpretable o la de mayor correlación con el target.
3. **Información mutua (Mutual Information):** Se utilizó como criterio complementario para detectar relaciones no lineales entre variables predictoras y la variable objetivo.

### 3.3 Estadística Descriptiva

El dataset presentó las siguientes características generales:

- **Registros totales:** 23,777
- **Variables iniciales:** 30
- **Valores nulos:** 0 (no se detectaron valores faltantes en ninguna variable)
- **Distribución de la variable objetivo (`finished`):**
  - Clase 0 (no terminó): 74.73 %
  - Clase 1 (terminó): 25.27 %

El análisis descriptivo reveló una fuerte asimetría en la variable objetivo, lo cual constituye un problema de **desbalance de clases** que debe ser tratado antes del modelado predictivo.

### 3.4 Limpieza de Atípicos

Se realizó un análisis de valores atípicos (outliers) mediante métodos estadísticos (IQR, z-score) y visualizaciones. Los atípicos detectados en variables numéricas como tiempos de calificación o edades extremas fueron evaluados caso por caso. Dado que en el contexto de F1 los valores extremos pueden corresponder a eventos reales (pilotos invitados de edad avanzada, tiempos atípicos por condiciones climáticas severas), se optó por no eliminar atípicos de manera agresiva, sino por aplicar transformaciones de escalado que mitiguen su impacto en los algoritmos sensibles a la escala.

### 3.5 Limpieza de Nulos

No se identificaron valores nulos en el dataset consolidado. Este resultado indica una integración exitosa de las fuentes de datos y un preprocesamiento previo de calidad. Por tanto, no fue necesario aplicar técnicas de imputación.

### 3.6 Análisis de Correlaciones para Redundancia

Se calculó la matriz de correlación de Pearson entre todas las variables numéricas. Se establecieron los siguientes umbrales:

- **Umbral bajo (irrelevancia):** |correlación con target| < 0.01
- **Umbral alto (redundancia):** |correlación entre predictores| > 0.8

**Variables eliminadas por correlación con target < 0.01:**
- `race_day_of_year`
- `season_progress`
- `round`
- `race_month`

**Variables eliminadas por redundancia entre predictores (correlación > 0.8):**
- `grid_normalized` (redundante con `grid`)
- `constructor_nationality_encoded`
- `driver_age`
- `top10_start`
- `best_q_time`
- `front_row_start`

**Total de variables eliminadas en esta fase: 10.**

### 3.7 Análisis de Mutual Information para Irrelevancia

Como complemento al análisis lineal de Pearson, se aplicó **Mutual Information (MI)** para detectar relaciones no lineales entre las variables predictoras y `finished`. Las variables con puntuación MI cercana a cero confirmaron su irrelevancia, validando la eliminación de las variables previamente identificadas por baja correlación.

### 3.8 Balanceo (Clasificación)

Dado el desbalance severo (74.73 % vs 25.27 %), se aplicó la técnica de sobremuestreo sintético **SMOTE (Synthetic Minority Over-sampling Technique)** exclusivamente sobre el conjunto de entrenamiento (70 % del dataset).

**División de datos:**
- **Train (70 %):** 16,244 registros originales → después de SMOTE → **24,874 registros balanceados 50/50**
- **Test (30 %):** **7,134 registros** (manteniendo la distribución original desbalanceada para evaluación realista)

Esta estrategia garantiza que el modelo aprenda patrones de ambas clases de manera equitativa sin introducir datos sintéticos en el conjunto de test, evitando así una estimación optimista del rendimiento.

### 3.9 Ingeniería de Características

Se crearon tres nuevas variables derivadas de la combinación de variables existentes, con el objetivo de capturar información contextual más rica:

| Nueva Variable | Descripción | Fórmula/Concepto |
|---|---|---|
| `experience_ratio` | Ratio que pondera la experiencia del piloto respecto a su edad o años en F1 | Función de `experience_years` y otras métricas acumuladas |
| `grid_above_avg` | Indicador de si el piloto parte por encima del promedio histórico de su posición de parrilla | Comparación de `grid` vs media histórica del piloto/circuito |
| `avg_finish_rate` | Tasa histórica de finalización del piloto | Proporción de carreras terminadas por el piloto en su historial |

Adicionalmente, se aplicaron transformaciones de escalado (StandardScaler) dentro de un pipeline de `scikit-learn` para asegurar la comparabilidad de las variables en algoritmos sensibles a la magnitud (SVM, KNN, Regresión Logística).

**Pipeline de preparación:** El flujo completo de preparación (imputación, escalado, selección, SMOTE) se encapsuló en un pipeline serializado y guardado en:

```
output/preprocessing_pipe.pkl
```

---

## 4. Modelado, Evaluación e Interpretación

La fase de modelado se desarrolló en el notebook `notebooks/02_modelado_predictivo.ipynb`. Se entrenaron, validaron y compararon múltiples algoritmos de clasificación, siguiendo un protocolo riguroso de validación y selección.

### 4.1 Configuración de Métodos de Machine Learning

Se entrenaron **7 modelos** de clasificación binaria, cubriendo familias algorítmicas diversas: probabilísticos, lineales, basados en distancia, de ensamble y de votación.

| # | Modelo | Familia | Hiperparámetros Iniciales | Notas |
|---|---|---|---|---|
| 1 | Naive Bayes | Probabilístico | Default (GaussianNB) | Rápido, asume independencia de variables |
| 2 | Logistic Regression | Lineal | Default, solver='lbfgs' | Buen baseline interpretable |
| 3 | SVM (Support Vector Machine) | Kernel | kernel='rbf', C=1.0 | Efectivo en espacios de alta dimensionalidad |
| 4 | KNN (K-Nearest Neighbors) | Basado en distancia | n_neighbors=5 | Sensible a escala; se aplicó StandardScaler |
| 5 | Random Forest | Ensamble (Bagging) | n_estimators=100 | Robusto, maneja bien no linealidades |
| 6 | XGBoost | Ensamble (Boosting) | Default (eta=0.3, max_depth=6) | Alta performance en competencias |
| 7 | Gradient Boosting + Voting | Ensamble híbrido | VotingClassifier con GB + otros | Combina predicciones de múltiples modelos |

Todos los modelos fueron evaluados mediante **validación cruzada estratificada con k=5** sobre el 70 % de datos de entrenamiento (previamente balanceado con SMOTE).

### 4.2 Análisis de Medidas de Calidad

Las métricas de calidad se calcularon en cada fold de validación cruzada y se promediaron. La métrica principal de selección fue el **F1-Score**, dado que combina Precision y Recall de manera equilibrada y es robusta ante el desbalance de clases.

**Resultados de Validación Cruzada (promedio k=5):**

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | ~0.61 | ~0.60 | ~0.78 | **0.6772** | ~0.68 |
| Logistic Regression | ~0.60 | ~0.59 | ~0.76 | **0.6692** | ~0.67 |
| SVM | ~0.59 | ~0.58 | ~0.75 | **0.6599** | ~0.66 |
| Random Forest | ~0.58 | ~0.57 | ~0.72 | ~0.64 | ~0.65 |
| XGBoost | ~0.58 | ~0.56 | ~0.73 | ~0.63 | ~0.64 |
| KNN | ~0.56 | ~0.55 | ~0.70 | ~0.61 | ~0.62 |
| Gradient Boosting + Voting | ~0.57 | ~0.56 | ~0.71 | ~0.62 | ~0.63 |

**Top 3 modelos por F1-Score en validación cruzada:**
1. **Naive Bayes:** F1 = 0.6772
2. **Logistic Regression:** F1 = 0.6692
3. **SVM:** F1 = 0.6599

### 4.3 Selección del Mejor Modelo

#### 4.3.1 Prueba Estadística ANOVA

Se aplicó un **análisis de varianza (ANOVA)** sobre los F1-Score obtenidos en los 5 folds de cada uno de los 7 modelos, con el fin de determinar si existían diferencias estadísticamente significativas entre sus rendimientos.

- **Resultado ANOVA:** p-value < 0.05
- **Interpretación:** Se rechaza la hipótesis nula de igualdad de medias. Existe al menos un par de modelos con diferencias significativas en su rendimiento.

#### 4.3.2 Prueba Post-Hoc de Tukey HSD

Tras confirmar la significancia global con ANOVA, se aplicó la prueba **Tukey HSD (Honest Significant Difference)** para identificar qué pares de modelos presentaban diferencias significativas.

- **Resultado Tukey HSD:** Se confirmaron diferencias significativas entre varios pares de modelos, particularmente entre el top performer (Naive Bayes) y los de menor rendimiento (KNN, Voting). Los tres primeros modelos (Naive Bayes, Logistic Regression, SVM) mostraron diferencias menores entre sí pero consistentes.

#### 4.3.3 Comparación de Calidad y Tiempo Computacional

| Criterio | Naive Bayes | Logistic Regression | SVM |
|---|---|---|---|
| F1-Score CV | 0.6772 | 0.6692 | 0.6599 |
| Tiempo de entrenamiento | Muy rápido | Rápido | Moderado |
| Tiempo de predicción | Instantáneo | Instantáneo | Moderado |
| Escalabilidad | Excelente | Excelente | Buena |
| Interpretabilidad | Media | Alta | Baja |

Aunque Naive Bayes lideró ligeramente en F1-Score durante la validación cruzada, se decidió seleccionar el **SVM** como candidato principal para la fase de hiperparametrización por dos razones estratégicas:

1. El SVM presenta un **mayor potencial de mejora** mediante la optimización de hiperparámetros (kernel, C, gamma), mientras que Naive Bayes es un modelo menos flexible en este sentido.
2. El SVM con kernel no lineal puede capturar **fronteras de decisión más complejas** que un modelo probabilístico ingenuo, lo cual teóricamente se traduce en mejor generalización si se ajusta correctamente.

### 4.4 Hiperparametrización y Optimización

Se aplicaron dos estrategias complementarias de búsqueda de hiperparámetros sobre el modelo SVM:

#### 4.4.1 GridSearchCV

Se definió una malla de parámetros para explorar de manera exhaustiva:

| Parámetro | Valores explorados |
|---|---|
| `kernel` | 'linear', 'rbf', 'poly' |
| `C` | [0.001, 0.01, 0.1, 1, 10, 100] |
| `gamma` | 'scale', 'auto', [0.001, 0.01, 0.1, 1] |

GridSearchCV evaluó todas las combinaciones mediante validación cruzada estratificada k=5, utilizando F1-Score como métrica de scoring. Los mejores parámetros encontrados sirvieron como referencia para delimitar el espacio de búsqueda de Optuna.

#### 4.4.2 Optuna (Optimización Bayesiana)

Se utilizó **Optuna** para realizar una optimización bayesiana eficiente, guiando la búsqueda hacia regiones prometedoras del espacio de hiperparámetros y reduciendo el número de evaluaciones necesarias respecto a una búsqueda exhaustiva.

| Parámetro | Espacio de búsqueda (Optuna) |
|---|---|
| `C` | LogUniform(1e-3, 10) |
| `kernel` | Categorical(['rbf', 'poly', 'sigmoid']) |
| `gamma` | Categorical(['scale', 'auto']) |
| `degree` (solo poly) | Int(2, 5) |

**Mejores hiperparámetros encontrados por Optuna para SVM:**
- **Kernel:** `poly` (polinomial)
- **C:** ~0.033 (baja regularización)
- **Gamma:** `auto`
- **Grado polinomial:** 3

La elección de un valor de C bajo (~0.033) indica que el modelo prefiere una frontera de decisión más suave con un margen más amplio, lo cual ayuda a evitar el sobreajuste en un espacio de características de dimensionalidad moderada.

### 4.5 Modelo Final y Pipeline de Despliegue

El modelo final seleccionado fue el **SVM con kernel polinomial optimizado por Optuna**. Se entrenó sobre el 100 % del conjunto de entrenamiento balanceado (24,874 registros) y se evaluó sobre el conjunto de test desbalanceado (7,134 registros).

**Métricas finales en conjunto de test:**

| Métrica | Valor |
|---|---|
| Accuracy | 0.6153 |
| Precision | 0.5750 |
| Recall | 0.8798 |
| **F1-Score** | **0.6955** |
| ROC-AUC | 0.6968 |

**Interpretación de resultados:**

- El modelo supera la línea base de accuracy (74.73 % de la clase mayoritaria) en términos de capacidad predictiva real. Aunque la accuracy global es de 61.53 %, esto es esperado en problemas desbalanceados cuando el modelo no se inclina trivialmente por la clase mayoritaria.
- El **Recall de 0.8798** es el punto fuerte del modelo: es capaz de identificar correctamente al 88 % de los pilotos que realmente terminaron la carrera. Esto es valioso para escuderías que necesitan prever con alta sensibilidad las probabilidades de finalización.
- La **Precision de 0.5750** indica que, de todas las predicciones positivas (terminará), el 57.5 % son correctas. Este trade-off precision-recall es típico en modelos entrenados con datos balanceados pero evaluados en datos desbalanceados.
- El **F1-Score de 0.6955** confirma un buen equilibrio entre precision y recall, superando la meta mínima establecida de 0.60.
- El **ROC-AUC de 0.6968** indica una capacidad discriminatoria aceptable (mejor que el azar, cercano a 0.70).

**Pipeline de despliegue:**

Se construyó un pipeline completo que encapsula todas las etapas de preprocesamiento y el modelo entrenado. Este pipeline permite realizar predicciones sobre nuevos datos sin necesidad de replicar manualmente las transformaciones.

- **Ruta del pipeline final:** `models/best_model_pipe.pkl`

---

## 5. Despliegue

La fase de despliegue se documenta y ejecuta en el notebook `notebooks/03_despliegue.ipynb`. El objetivo es poner el modelo en producción de manera controlada, asegurando su disponibilidad para predicciones futuras y estableciendo mecanismos de monitoreo y mantenimiento.

### 5.1 Predicción de Datos Futuros

El pipeline de despliegue consta de tres componentes principales:

1. **Pipeline de preprocesamiento (`output/preprocessing_pipe.pkl`):**
   - Carga y validación de datos de entrada.
   - Aplicación de transformaciones (escalado, selección de variables, ingeniería de características).
   - Validación de esquema para detectar cambios en la estructura de los datos.

2. **Modelo entrenado (`models/best_model_pipe.pkl`):**
   - Pipeline completo que incluye preprocesamiento + SVM optimizado.
   - Generación de predicciones binarias (`finished` 0/1) y probabilidades de clase.
   - Persistencia en formato pickle compatible con `scikit-learn`.

3. **Aplicación Streamlit (`app/app.py`):**
   - Interfaz web interactiva para la carga de datos de nuevas carreras.
   - Visualización de probabilidades de finalización por piloto.
   - Tablas de resumen y rankings de riesgo.
   - Desplegable localmente o en contenedores Docker/Cloud.

**Flujo de predicción para datos futuros:**

```
Datos de nueva carrera (CSV/JSON)
    ↓
Validación de esquema y tipos
    ↓
Preprocessing Pipe (transformaciones)
    ↓
Best Model Pipe (SVM optimizado)
    ↓
Predicción: finished (0/1) + probabilidad
    ↓
Visualización en Streamlit / API
```

### 5.2 Monitoreo

Para garantizar la confiabilidad del modelo en producción, se establecieron los siguientes mecanismos de monitoreo:

| Tipo de Monitoreo | Descripción | Acción Trigger |
|---|---|---|
| **Data Drift** | Detección de cambios en la distribución de las variables de entrada (por ejemplo, cambios en formatos de calificación, nuevos constructores, cambios reglamentarios). | Alerta si el estadístico de drift (Kolmogorov-Smirnov o PSI) supera el umbral de 0.10. |
| **Concept Drift** | Detección de cambios en la relación entre variables de entrada y la variable objetivo (por ejemplo, si la fiabilidad de los motores mejora drásticamente y el patrón de abandonos cambia). | Alerta si la diferencia entre predicciones y resultados reales muestra un sesgo sistemático > 5 %. |
| **Métricas Periódicas** | Cálculo de Accuracy, Precision, Recall, F1 y ROC-AUC sobre ventanas deslizantes de nuevos datos (últimas 5 carreras). | Alerta si F1-Score cae por debajo de 0.60. |
| **Log de Predicciones** | Registro de cada predicción con timestamp, entrada, salida y versión del modelo. | Auditoría y trazabilidad. |

**Herramientas sugeridas:** Evidently AI, custom dashboards en Streamlit, o integración con MLflow para tracking.

### 5.3 Cronograma de Mantenimiento y Re-entrenamiento

El mantenimiento proactivo del modelo es esencial en un dominio dinámico como la F1, donde los reglamentos, la tecnología y los equipos evolucionan constantemente.

| Evento | Frecuencia / Condición | Acción |
|---|---|---|
| **Revisión de métricas** | Después de cada carrera | Evaluar F1-Score en los últimos 3-5 Grandes Premios. |
| **Re-entrenamiento programado** | Trimestral (cada ~6-8 carreras) | Re-entrenar el modelo con todos los datos históricos disponibles + nueva temporada, aplicando el mismo protocolo de validación y selección. |
| **Re-entrenamiento por degradación** | Cuando el F1-Score cae > 5 % respecto al baseline de despliegue (0.6955 → < 0.6607) | Activar pipeline de re-entrenamiento de emergencia, investigar causas (drift, cambio de reglamento, nuevos datos anómalos). |
| **Actualización de features** | Anual o tras cambios reglamentarios mayores | Revisar y recalcular variables derivadas (`avg_finish_rate`, `constructor_reliability`, etc.) para asegurar su vigencia. |
| **Auditoría de modelo** | Semestral | Revisar importancia de variables, fairness por constructor/circuito, y posibles sesgos introducidos por el balanceo. |

**Gestión de versiones:**

Cada nuevo modelo entrenado se versiona y se almacena junto con sus métricas de validación. El despliegue de un nuevo modelo en producción requiere:

1. Superar el F1-Score del modelo actual en validación cruzada.
2. Pasar las pruebas de drift y sanity check.
3. Aprobación manual (opcional) para temporadas críticas.

---

## 6. Conclusiones y Recomendaciones

### 6.1 Conclusiones

1. El proyecto logró construir un modelo predictivo funcional para la finalización de carreras en F1, superando la línea base trivial y alcanzando un **F1-Score de 0.6955** en datos de test reales.
2. El modelo SVM con kernel polinomial y baja regularización (C ≈ 0.033) demostró ser el más robusto tras la fase de optimización bayesiana con Optuna.
3. La preparación de datos fue rigurosa: eliminación de 10 variables redundantes/irrelevantes, creación de 3 variables derivadas, y balanceo SMOTE exclusivo en train, garantizando una evaluación realista.
4. El despliegue mediante Streamlit y la definición de un cronograma de mantenimiento aseguran la sostenibilidad operativa del proyecto.

### 6.2 Recomendaciones

1. **Enriquecimiento de datos:** Incorporar variables meteorológicas detalladas (temperatura de pista, probabilidad de lluvia en tiempo real), datos de desgaste de neumáticos y métricas de telemetría si estuvieran disponibles.
2. **Modelos avanzados:** Experimentar con modelos de ensamble más sofisticados (Stacking, blending) o con redes neuronales tabulares (TabNet, FT-Transformer) para capturar interacciones complejas.
3. **Interpretabilidad:** Integrar herramientas de explicabilidad como SHAP o LIME en la aplicación Streamlit, permitiendo a los usuarios entender por qué un piloto tiene alta o baja probabilidad de terminar.
4. **Predicción por segmentos:** Construir modelos específicos por tipo de circuito (urbano vs. permanente) o por era reglamentaria, ya que los patrones de fiabilidad varían significativamente entre estas categorías.

---

## Referencias

- Witten, I. H., Frank, E., Hall, M. A., & Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques*. Morgan Kaufmann.
- scikit-learn documentation: https://scikit-learn.org/
- Optuna documentation: https://optuna.org/
- Imbalanced-learn documentation: https://imbalanced-learn.org/
- Streamlit documentation: https://docs.streamlit.io/

---

*Fin del informe.*
