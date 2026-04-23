# Exploración Exhaustiva del Dataset F1

## 1. Dimensiones
Filas: 23777 | Columnas: 30

## 2. Primeras 10 Filas
|   grid |   laps |   driver_age |   driver_race_count |   driver_prev_finish_rate |   driver_last5_finish_rate |   constructor_race_count |   constructor_prev_finish_rate |   constructor_prev_avg_grid |   constructor_last5_finish_rate |   circuit_finish_rate |   circuit_avg_grid |   race_month |   race_day_of_year |   season_progress |   q1_seconds |   q2_seconds |   q3_seconds |   best_q_time |   grid_normalized |   has_qualifying |   front_row_start |   top10_start |   year |   round |   driver_nationality_encoded |   constructor_nationality_encoded |   circuit_country_encoded |   circuitRef_encoded |   finished |
|-------:|-------:|-------------:|--------------------:|--------------------------:|---------------------------:|-------------------------:|-------------------------------:|----------------------------:|--------------------------------:|----------------------:|-------------------:|-------------:|-------------------:|------------------:|-------------:|-------------:|-------------:|--------------:|------------------:|-----------------:|------------------:|--------------:|-------:|--------:|-----------------------------:|----------------------------------:|--------------------------:|---------------------:|-----------:|
|     16 |     13 |      30.9295 |                  85 |                 0.285714  |                   0        |                        1 |                       0        |                     0       |                             0   |              0.3159   |           11.7479  |            8 |                217 |         0.666667  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.8      |                0 |                 0 |             0 |   1968 |       8 |                           29 |                                 5 |                        10 |                   45 |          0 |
|      5 |     92 |      32.3012 |                  37 |                 0.361111  |                   0        |                        2 |                       0        |                    16       |                             0   |              0.165631 |           11.6853  |           10 |                280 |         0.916667  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.25     |                0 |                 0 |             1 |   1968 |      11 |                           29 |                                 5 |                        31 |                   66 |          0 |
|      7 |     78 |      34.7132 |                  61 |                 0.383333  |                   0        |                        3 |                       0        |                    10.5     |                             0   |              0.173913 |           12.1759  |            3 |                 65 |         0.0909091 |       87.609 |       89.824 |      88.7805 |        87.264 |          0.28     |                0 |                 0 |             1 |   1971 |       1 |                           29 |                                 5 |                        24 |                   33 |          0 |
|     11 |      7 |      31.0363 |                   8 |                 0         |                   0        |                        4 |                       0        |                     9.33333 |                             0   |              0.173913 |           12.1759  |            3 |                 65 |         0.0909091 |       87.609 |       89.824 |      88.7805 |        87.264 |          0.44     |                0 |                 0 |             0 |   1971 |       1 |                            8 |                                 5 |                        24 |                   33 |          0 |
|     23 |      5 |      41.0924 |                 105 |                 0.0769231 |                   0        |                        5 |                       0        |                     9.75    |                             0   |              0.173913 |           12.1759  |            3 |                 65 |         0.0909091 |       87.609 |       89.824 |      88.7805 |        87.264 |          0.92     |                0 |                 0 |             0 |   1971 |       1 |                           36 |                                 5 |                        24 |                   33 |          0 |
|      9 |     75 |      34.8309 |                  62 |                 0.377049  |                   0        |                        6 |                       0        |                    12.4     |                             0   |              0.13253  |           11.3735  |            4 |                108 |         0.181818  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.409091 |                0 |                 0 |             1 |   1971 |       2 |                           29 |                                 5 |                        25 |                   41 |          1 |
|      7 |     73 |      31.154  |                   9 |                 0         |                   0        |                        7 |                       0.166667 |                    11.8333  |                             0.2 |              0.13253  |           11.3735  |            4 |                108 |         0.181818  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.318182 |                0 |                 0 |             1 |   1971 |       2 |                            8 |                                 5 |                        25 |                   41 |          0 |
|      6 |     80 |      34.9268 |                  63 |                 0.387097  |                   0.333333 |                        8 |                       0.142857 |                    11.1429  |                             0.2 |              0.185881 |            9.39767 |            5 |                143 |         0.272727  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.333333 |                0 |                 0 |             1 |   1971 |       3 |                           29 |                                 5 |                        18 |                   39 |          1 |
|     14 |     22 |      31.2498 |                  10 |                 0         |                   0        |                        9 |                       0.25     |                    10.5     |                             0.4 |              0.185881 |            9.39767 |            5 |                143 |         0.272727  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.777778 |                0 |                 0 |             0 |   1971 |       3 |                            8 |                                 5 |                        18 |                   39 |          0 |
|     14 |     63 |      35.0034 |                  64 |                 0.396825  |                   0.5      |                       10 |                       0.222222 |                    10.8889  |                             0.4 |              0.163028 |           10.4585  |            6 |                171 |         0.363636  |       87.609 |       89.824 |      88.7805 |        87.264 |          0.583333 |                0 |                 0 |             0 |   1971 |       4 |                           29 |                                 5 |                        20 |                   69 |          0 |

## 3. Información General (info)
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 23777 entries, 0 to 23776
Data columns (total 30 columns):
 #   Column                           Non-Null Count  Dtype  
---  ------                           --------------  -----  
 0   grid                             23777 non-null  int64  
 1   laps                             23777 non-null  int64  
 2   driver_age                       23777 non-null  float64
 3   driver_race_count                23777 non-null  int64  
 4   driver_prev_finish_rate          23777 non-null  float64
 5   driver_last5_finish_rate         23777 non-null  float64
 6   constructor_race_count           23777 non-null  int64  
 7   constructor_prev_finish_rate     23777 non-null  float64
 8   constructor_prev_avg_grid        23777 non-null  float64
 9   constructor_last5_finish_rate    23777 non-null  float64
 10  circuit_finish_rate              23777 non-null  float64
 11  circuit_avg_grid                 23777 non-null  float64
 12  race_month                       23777 non-null  int64  
 13  race_day_of_year                 23777 non-null  int64  
 14  season_progress                  23777 non-null  float64
 15  q1_seconds                       23777 non-null  float64
 16  q2_seconds                       23777 non-null  float64
 17  q3_seconds                       23777 non-null  float64
 18  best_q_time                      23777 non-null  float64
 19  grid_normalized                  23777 non-null  float64
 20  has_qualifying                   23777 non-null  int64  
 21  front_row_start                  23777 non-null  int64  
 22  top10_start                      23777 non-null  int64  
 23  year                             23777 non-null  int64  
 24  round                            23777 non-null  int64  
 25  driver_nationality_encoded       23777 non-null  int64  
 26  constructor_nationality_encoded  23777 non-null  int64  
 27  circuit_country_encoded          23777 non-null  int64  
 28  circuitRef_encoded               23777 non-null  int64  
 29  finished                         23777 non-null  int64  
dtypes: float64(14), int64(16)
memory usage: 5.4 MB

```

## 4. Tipos de Datos
| Variable                        | Tipo_Dato   |
|:--------------------------------|:------------|
| grid                            | int64       |
| laps                            | int64       |
| driver_age                      | float64     |
| driver_race_count               | int64       |
| driver_prev_finish_rate         | float64     |
| driver_last5_finish_rate        | float64     |
| constructor_race_count          | int64       |
| constructor_prev_finish_rate    | float64     |
| constructor_prev_avg_grid       | float64     |
| constructor_last5_finish_rate   | float64     |
| circuit_finish_rate             | float64     |
| circuit_avg_grid                | float64     |
| race_month                      | int64       |
| race_day_of_year                | int64       |
| season_progress                 | float64     |
| q1_seconds                      | float64     |
| q2_seconds                      | float64     |
| q3_seconds                      | float64     |
| best_q_time                     | float64     |
| grid_normalized                 | float64     |
| has_qualifying                  | int64       |
| front_row_start                 | int64       |
| top10_start                     | int64       |
| year                            | int64       |
| round                           | int64       |
| driver_nationality_encoded      | int64       |
| constructor_nationality_encoded | int64       |
| circuit_country_encoded         | int64       |
| circuitRef_encoded              | int64       |
| finished                        | int64       |

## 5. Estadísticas Descriptivas (Variables Numéricas)
|                                 |   count |        mean |        std |         min |         25% |         50% |         75% |         max |
|:--------------------------------|--------:|------------:|-----------:|------------:|------------:|------------:|------------:|------------:|
| grid                            |   23777 |   11.2703   |   7.34644  |    0        |    5        |   11        |   17        |   34        |
| laps                            |   23777 |   45.2706   |  30.5254   |    0        |   20        |   52        |   66        |  200        |
| driver_age                      |   23777 |   30.1748   |   5.0984   |   17.4538   |   26.4641   |   29.6454   |   33.3771   |   53.2621   |
| driver_race_count               |   23777 |   59.0434   |  59.4337   |    1        |   13        |   39        |   86        |  326        |
| driver_prev_finish_rate         |   23777 |    0.215685 |   0.21727  |    0        |    0        |    0.157895 |    0.356322 |    1        |
| driver_last5_finish_rate        |   23777 |    0.250472 |   0.305645 |    0        |    0        |    0.2      |    0.4      |    1        |
| constructor_race_count          |   23777 |  347.535    | 417.188    |    1        |   60        |  187        |  469        | 2140        |
| constructor_prev_finish_rate    |   23777 |    0.220794 |   0.177653 |    0        |    0.0625   |    0.205882 |    0.354839 |    1        |
| constructor_prev_avg_grid       |   23777 |   10.9735   |   4.10202  |    0        |    7.64809  |   10.5018   |   13.6674   |   32        |
| constructor_last5_finish_rate   |   23777 |    0.251555 |   0.298946 |    0        |    0        |    0.2      |    0.4      |    1        |
| circuit_finish_rate             |   23777 |    0.252723 |   0.100193 |    0.0625   |    0.181102 |    0.240546 |    0.294939 |    0.595745 |
| circuit_avg_grid                |   23777 |   11.2703   |   0.971286 |    7.64     |   11.0238   |   11.3246   |   11.6795   |   15.1222   |
| race_month                      |   23777 |    6.75977  |   2.37055  |    1        |    5        |    7        |    9        |   12        |
| race_day_of_year                |   23777 |  190.079    |  71.9059   |    1        |  139        |  192        |  250        |  363        |
| season_progress                 |   23777 |    0.535213 |   0.287078 |    0.047619 |    0.285714 |    0.533333 |    0.785714 |    1        |
| q1_seconds                      |   23777 |   88.2726   |   9.11875  |   65.064    |   87.609    |   87.609    |   87.609    | 1002.64     |
| q2_seconds                      |   23777 |   89.7959   |   4.62198  |   64.316    |   89.824    |   89.824    |   89.824    |  132.47     |
| q3_seconds                      |   23777 |   88.8376   |   3.61317  |   64.251    |   88.7805   |   88.7805   |   88.7805   |  129.776    |
| best_q_time                     |   23777 |   87.8706   |   9.03338  |   64.251    |   87.264    |   87.264    |   87.264    | 1002.64     |
| grid_normalized                 |   23777 |    0.486186 |   0.307046 |    0        |    0.222222 |    0.5      |    0.75     |    1        |
| has_qualifying                  |   23777 |    0.311099 |   0.462953 |    0        |    0        |    0        |    1        |    1        |
| front_row_start                 |   23777 |    0.148421 |   0.355524 |    0        |    0        |    0        |    0        |    1        |
| top10_start                     |   23777 |    0.478866 |   0.499564 |    0        |    0        |    0        |    1        |    1        |
| year                            |   23777 | 1987.68     |  17.9722   | 1950        | 1975        | 1989        | 2003        | 2017        |
| round                           |   23777 |    8.16987  |   4.78899  |    1        |    4        |    8        |   12        |   21        |
| driver_nationality_encoded      |   23777 |   15.485    |   9.81117  |    0        |    8        |   17        |   23        |   40        |
| constructor_nationality_encoded |   23777 |    8.74572  |   5.1044   |    0        |    5        |    5        |   14        |   23        |
| circuit_country_encoded         |   23777 |   15.1091   |   9.36593  |    0        |    7        |   13        |   24        |   31        |
| circuitRef_encoded              |   23777 |   39.7683   |  18.857    |    0        |   25        |   39        |   59        |   71        |
| finished                        |   23777 |    0.252723 |   0.434583 |    0        |    0        |    0        |    1        |    1        |

## 6. Distribución de la Variable Objetivo: `finished`
|   Clase |   Conteo |   Porcentaje |
|--------:|---------:|-------------:|
|       0 |    17768 |        74.73 |
|       1 |     6009 |        25.27 |

## 7. Valores Nulos por Columna
| Variable                        |   Nulos |   Porcentaje_Nulos |
|:--------------------------------|--------:|-------------------:|
| grid                            |       0 |                  0 |
| laps                            |       0 |                  0 |
| driver_age                      |       0 |                  0 |
| driver_race_count               |       0 |                  0 |
| driver_prev_finish_rate         |       0 |                  0 |
| driver_last5_finish_rate        |       0 |                  0 |
| constructor_race_count          |       0 |                  0 |
| constructor_prev_finish_rate    |       0 |                  0 |
| constructor_prev_avg_grid       |       0 |                  0 |
| constructor_last5_finish_rate   |       0 |                  0 |
| circuit_finish_rate             |       0 |                  0 |
| circuit_avg_grid                |       0 |                  0 |
| race_month                      |       0 |                  0 |
| race_day_of_year                |       0 |                  0 |
| season_progress                 |       0 |                  0 |
| q1_seconds                      |       0 |                  0 |
| q2_seconds                      |       0 |                  0 |
| q3_seconds                      |       0 |                  0 |
| best_q_time                     |       0 |                  0 |
| grid_normalized                 |       0 |                  0 |
| has_qualifying                  |       0 |                  0 |
| front_row_start                 |       0 |                  0 |
| top10_start                     |       0 |                  0 |
| year                            |       0 |                  0 |
| round                           |       0 |                  0 |
| driver_nationality_encoded      |       0 |                  0 |
| constructor_nationality_encoded |       0 |                  0 |
| circuit_country_encoded         |       0 |                  0 |
| circuitRef_encoded              |       0 |                  0 |
| finished                        |       0 |                  0 |

## 8. Diccionario de Datos Inferido
| Variable                        | Descripcion                                                    | Tipo_Dato   |   Unicos |     Ejemplo |
|:--------------------------------|:---------------------------------------------------------------|:------------|---------:|------------:|
| grid                            | Posición de salida en la parrilla (grid position)              | int64       |       35 |   16        |
| laps                            | Número de vueltas completadas en la carrera                    | int64       |      172 |   13        |
| driver_age                      | Edad del piloto al momento de la carrera                       | float64     |     7251 |   30.9295   |
| driver_race_count               | Número total de carreras previas del piloto                    | int64       |      326 |   85        |
| driver_prev_finish_rate         | Tasa histórica de finalización del piloto                      | float64     |     4517 |    0.285714 |
| driver_last5_finish_rate        | Tasa de finalización del piloto en las últimas 5 carreras      | float64     |       11 |    0        |
| constructor_race_count          | Número total de carreras previas del constructor               | int64       |     2140 |    1        |
| constructor_prev_finish_rate    | Tasa histórica de finalización del constructor                 | float64     |    12187 |    0        |
| constructor_prev_avg_grid       | Posición media de salida histórica del constructor             | float64     |    19761 |    0        |
| constructor_last5_finish_rate   | Tasa de finalización del constructor en las últimas 5 carreras | float64     |       11 |    0        |
| circuit_finish_rate             | Tasa histórica de finalización en el circuito                  | float64     |       67 |    0.3159   |
| circuit_avg_grid                | Posición media de salida histórica en el circuito              | float64     |       66 |   11.7479   |
| race_month                      | Mes en que se disputa la carrera                               | int64       |       12 |    8        |
| race_day_of_year                | Día del año en que se disputa la carrera                       | int64       |      288 |  217        |
| season_progress                 | Progreso de la temporada (fracción de carreras completadas)    | float64     |      140 |    0.666667 |
| q1_seconds                      | Tiempo de clasificación Q1 en segundos                         | float64     |     6736 |   87.609    |
| q2_seconds                      | Tiempo de clasificación Q2 en segundos                         | float64     |     3480 |   89.824    |
| q3_seconds                      | Tiempo de clasificación Q3 en segundos                         | float64     |     2122 |   88.7805   |
| best_q_time                     | Mejor tiempo de clasificación (menor de Q1, Q2, Q3)            | float64     |     6730 |   87.264    |
| grid_normalized                 | Posición de salida normalizada (escala 0-1 aprox)              | float64     |      361 |    0.8      |
| has_qualifying                  | Indicador binario de si hubo sesión de clasificación           | int64       |        2 |    0        |
| front_row_start                 | Indicador binario de si salió en la primera fila               | int64       |        2 |    0        |
| top10_start                     | Indicador binario de si salió en el top 10                     | int64       |        2 |    0        |
| year                            | Año de la carrera                                              | int64       |       68 | 1968        |
| round                           | Número de ronda dentro de la temporada                         | int64       |       21 |    8        |
| driver_nationality_encoded      | Nacionalidad del piloto (codificada numéricamente)             | int64       |       41 |   29        |
| constructor_nationality_encoded | Nacionalidad del constructor (codificada numéricamente)        | int64       |       24 |    5        |
| circuit_country_encoded         | País del circuito (codificado numéricamente)                   | int64       |       32 |   10        |
| circuitRef_encoded              | Referencia del circuito (codificada numéricamente)             | int64       |       72 |   45        |
| finished                        | Variable objetivo: 1 si el piloto finalizó la carrera, 0 si no | int64       |        2 |    0        |

## 9. Correlación Absoluta con `finished` (Top 15)
| Variable                      |   Correlacion_Abs_con_finished |
|:------------------------------|-------------------------------:|
| driver_last5_finish_rate      |                         0.4849 |
| constructor_last5_finish_rate |                         0.4782 |
| driver_prev_finish_rate       |                         0.4653 |
| constructor_prev_finish_rate  |                         0.4565 |
| laps                          |                         0.3467 |
| top10_start                   |                         0.3312 |
| grid                          |                         0.3256 |
| grid_normalized               |                         0.3213 |
| constructor_race_count        |                         0.2886 |
| constructor_prev_avg_grid     |                         0.2805 |
| driver_race_count             |                         0.2761 |
| has_qualifying                |                         0.2395 |
| circuit_finish_rate           |                         0.2305 |
| year                          |                         0.2202 |
| front_row_start               |                         0.1048 |

## 10. Observaciones Clave
- El dataset contiene **23777** registros y **30** variables.
- No se detectaron valores nulos en ninguna columna (porcentaje 0.0%% en todas).
- La variable objetivo `finished` está desbalanceada: la clase 1 (finalizó) representa aproximadamente el 25.27% de los registros.
- Existen múltiples variables derivadas de historiales (tasas de finalización, promedios de grid) que capturan comportamiento pasado.
- Las variables codificadas (`*_encoded`) son numéricas discretas resultado de transformaciones de variables categóricas.
- Los tiempos de clasificación (`q1_seconds`, `q2_seconds`, `q3_seconds`, `best_q_time`) presentan el mismo valor en las primeras filas, lo que podría indicar imputación o datos faltantes para épocas sin sesiones de clasificación modernas.

---
*Generado automáticamente para el proyecto CRISP-DM - F1 Classification*
