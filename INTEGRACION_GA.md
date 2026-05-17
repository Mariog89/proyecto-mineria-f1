# Guía de Integración: Algoritmos Genéticos en 02_modelado_predictivo.ipynb

## 📋 Resumen de Cambios

Se ha implementado **optimización con Algoritmos Genéticos (GA)** usando DEAP como complemento a GridSearchCV y Optuna. Esto permite:

1. **Explorar el espacio de hiperparámetros de forma más inteligente**
2. **Comparar 3 métodos de optimización**: GridSearch, Optuna y GA
3. **Mejor detección de overfitting** con métricas en train/test

---

## 🔧 Instalación

```bash
pip install deap
```

O ejecuta los requirements actualizados:

```bash
pip install -r requirements.txt
```

---

## 📝 Código de Integración en el Notebook

### **PASO 1**: Importar módulos (al inicio del notebook)

```python
# ============================================================
# IMPORTAR MÓDULOS DE OPTIMIZACIÓN GA
# ============================================================
import sys
from pathlib import Path

# Agregar scripts al path
sys.path.insert(0, str(ROOT / "scripts"))

from genetic_optimization import (
    create_ga_optimizer_for_model,
    plot_ga_evolution,
    GeneticOptimizer
)
from verify_overfitting_and_ga import (
    analyze_overfitting_cv,
    run_genetic_optimization_on_top_models,
    compare_optimization_methods,
    print_overfitting_report
)
```

---

### **PASO 2**: Análisis de Overfitting (NUEVO - Sección 4.5)

Agregar DESPUÉS de la sección 4 (Selección del mejor modelo):

```python
# ============================================================
# 4.5 ANÁLISIS DETALLADO DE OVERFITTING EN VALIDACIÓN CRUZADA
# ============================================================
print("\n" + "="*80)
print("ANÁLISIS DE OVERFITTING CON SCORES EN TRAIN/TEST".center(80))
print("="*80)

# Analizar overfitting para cada uno de los top 3 modelos
overfitting_analyses = {}

for name in top3:
    model = best_models[name]
    analysis = analyze_overfitting_cv(
        model, X_train, y_train, cv=5, model_name=name
    )
    overfitting_analyses[name] = analysis
    
    print(f"\n📊 {name}")
    print(f"   {analysis['diagnosis']}")
    for metric, values in analysis['analysis'].items():
        status = "✓" if not values['is_overfitting'] else "⚠️"
        print(f"   {metric}: Train={values['train_mean']:.4f}, "
              f"Test={values['test_mean']:.4f}, Gap={values['gap']:.4f} {status}")

# Guardar análisis
overfitting_df = pd.DataFrame([
    {
        'Modelo': name,
        'Diagnosis': analysis['diagnosis'],
        'F1_Gap': analysis['analysis']['f1']['gap'],
        'Acc_Gap': analysis['analysis']['accuracy']['gap'],
        'ROC_Gap': analysis['analysis']['roc_auc']['gap']
    }
    for name, analysis in overfitting_analyses.items()
])

print("\nTabla Resumen:")
display(overfitting_df)

# Guardar
overfitting_df.to_csv(OUTPUT_DIR / 'overfitting_analysis.csv', index=False)
print(f"✓ Análisis guardado en: {OUTPUT_DIR / 'overfitting_analysis.csv'}")
```

---

### **PASO 3**: Optimización con GA (NUEVO - Sección 5.2.5)

Insertar DESPUÉS de GridSearchCV (sección 5.2) y ANTES de Optuna (sección 5.3):

```python
# ============================================================
# 5.2.5 OPTIMIZACIÓN CON ALGORITMOS GENÉTICOS (GA)
# ============================================================
print("\n" + "="*80)
print("OPTIMIZACIÓN CON ALGORITMOS GENÉTICOS (DEAP)".center(80))
print("="*80)

# Definir espacios de búsqueda GA para cada modelo
# (Igual que GridSearch pero con tipos DEAP)
ga_param_spaces = {
    'LogisticRegression': {
        'clf__C': ('float', 0.01, 100, 0.1),
        'clf__penalty': ('categorical', None, ['l2', 'elasticnet']),
        'clf__solver': ('categorical', None, ['lbfgs', 'liblinear', 'saga'])
    },
    'RandomForest': {
        'clf__n_estimators': ('int', 50, 300, 10),
        'clf__max_depth': ('int', 5, 30, 1),
        'clf__min_samples_split': ('int', 2, 20, 1),
        'clf__min_samples_leaf': ('int', 1, 10, 1)
    },
    'GradientBoosting': {
        'clf__n_estimators': ('int', 50, 300, 10),
        'clf__learning_rate': ('float', 0.001, 0.3, 0.01),
        'clf__max_depth': ('int', 3, 10, 1),
        'clf__subsample': ('float', 0.5, 1.0, 0.1)
    }
}

# Ejecutar GA para los 3 mejores modelos
ga_results = {}

for name in top3:
    if name not in ga_param_spaces:
        print(f'⏭️  GA saltado para {name} (no soportado en esta versión)')
        continue
    
    print(f'\n🧬 GA: {name} ...', end=' ')
    start = time.time()
    
    try:
        model_for_ga = best_models[name]
        param_space = ga_param_spaces[name]
        
        # Crear y ejecutar optimizador GA
        optimizer = create_ga_optimizer_for_model(
            model_for_ga,
            X_train, y_train,
            param_space,
            cv=5,
            generations=15,        # ← Ajustar si necesitas más generaciones
            population_size=20,    # ← Ajustar si necesitas más exploración
            verbose=False
        )
        
        result = optimizer.optimize()
        elapsed = time.time() - start
        
        ga_results[name] = {
            'best_params': result['best_params'],
            'best_cv_f1': result['best_fitness'],
            'time_s': elapsed,
            'history': result['history']
        }
        
        print(f'✓ (F1={result["best_fitness"]:.4f}, {elapsed:.1f}s)')
        
    except Exception as e:
        print(f'✗ Error: {e}')

print('\n✓ Optimización GA completada')
```

---

### **PASO 4**: Comparación de Métodos (Modificar Sección 5.4)

Reemplazar la sección 5.4 (Comparación GridSearch vs Optuna) con:

```python
# ============================================================
# 5.4 COMPARACIÓN GRIDSEARCH VS OPTUNA VS GA (ACTUALIZADO)
# ============================================================

comparison_opt = []

for name in top3:
    row = {'Modelo': name}
    
    # GridSearchCV
    if name in grid_results:
        row['Grid_F1'] = grid_results[name]['best_cv_f1']
        row['Grid_Time'] = grid_results[name]['time_s']
    else:
        row['Grid_F1'] = np.nan
        row['Grid_Time'] = np.nan
    
    # Optuna
    if name in optuna_results:
        row['Optuna_F1'] = optuna_results[name]['best_cv_f1']
        row['Optuna_Time'] = optuna_results[name]['time_s']
    else:
        row['Optuna_F1'] = np.nan
        row['Optuna_Time'] = np.nan
    
    # GA (NUEVO)
    if name in ga_results:
        row['GA_F1'] = ga_results[name]['best_cv_f1']
        row['GA_Time'] = ga_results[name]['time_s']
    else:
        row['GA_F1'] = np.nan
        row['GA_Time'] = np.nan
    
    comparison_opt.append(row)

comparison_df = pd.DataFrame(comparison_opt)

print("\n📊 COMPARACIÓN DE MÉTODOS DE OPTIMIZACIÓN")
print("="*80)
display(comparison_df)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1 Scores
x = np.arange(len(comparison_df))
width = 0.25
axes[0].bar(x - width, comparison_df['Grid_F1'], width, label='GridSearch', alpha=0.8)
axes[0].bar(x, comparison_df['Optuna_F1'], width, label='Optuna', alpha=0.8)
axes[0].bar(x + width, comparison_df['GA_F1'], width, label='GA', alpha=0.8)
axes[0].set_ylabel('F1 Score')
axes[0].set_title('Calidad de Optimización')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Modelo'], rotation=45)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Tiempo
axes[1].bar(x - width, comparison_df['Grid_Time'], width, label='GridSearch', alpha=0.8)
axes[1].bar(x, comparison_df['Optuna_Time'], width, label='Optuna', alpha=0.8)
axes[1].bar(x + width, comparison_df['GA_Time'], width, label='GA', alpha=0.8)
axes[1].set_ylabel('Tiempo (segundos)')
axes[1].set_title('Eficiencia Computacional')
axes[1].set_xticks(x)
axes[1].set_xticklabels(comparison_df['Modelo'], rotation=45)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'optimization_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# Guardar
comparison_df.to_csv(OUTPUT_DIR / 'optimization_methods_comparison.csv', index=False)
print(f"\n✓ Comparación guardada en: {OUTPUT_DIR / 'optimization_methods_comparison.csv'}")
```

---

### **PASO 5**: Selección del Mejor Modelo (Modificar Sección 5.5)

Actualizar para incluir GA en la selección final:

```python
# ============================================================
# 5.5 SELECCIÓN DEL MEJOR MODELO FINAL (ACTUALIZADO CON GA)
# ============================================================

best_by_model = {}

for name in top3:
    candidates = []
    
    # GridSearch
    if name in grid_results:
        candidates.append(('Grid', grid_results[name]['best_cv_f1'], 
                          grid_results[name]['time_s'], 
                          grid_results[name]['best_estimator']))
    
    # Optuna
    if name in optuna_results:
        candidates.append(('Optuna', optuna_results[name]['best_cv_f1'], 
                          optuna_results[name]['time_s'], 
                          optuna_results[name]['best_estimator']))
    
    # GA (NUEVO)
    if name in ga_results:
        # Entrenar modelo con parámetros GA optimizados
        model_copy = best_models[name]
        model_copy.set_params(**ga_results[name]['best_params'])
        candidates.append(('GA', ga_results[name]['best_cv_f1'], 
                          ga_results[name]['time_s'],
                          model_copy))
    
    # Seleccionar mejor entre Grid, Optuna y GA
    best = max(candidates, key=lambda x: x[1])
    best_by_model[name] = {
        'method': best[0],
        'f1': best[1],
        'time_s': best[2],
        'model': best[3]
    }

# Seleccionar mejor modelo global
final_model_name = max(best_by_model.keys(), 
                       key=lambda k: best_by_model[k]['f1'])
final_model_info = best_by_model[final_model_name]

print("\n" + "="*80)
print("🏆 MODELO FINAL SELECCIONADO".center(80))
print("="*80)
print(f"Modelo: {final_model_name}")
print(f"Método de optimización: {final_model_info['method']}")
print(f"F1 Score: {final_model_info['f1']:.4f}")
print(f"Tiempo: {final_model_info['time_s']:.2f}s")
print("="*80)

final_model = final_model_info['model']
```

---

## 📊 Nuevas Métricas Disponibles

### Overfitting Analysis
- **Gap**: Diferencia entre train y test score
- **% Gap**: Porcentaje relativo de overfitting
- **Diagnóstico**: Clasificación automática (Sin/Leve/Moderado/Severo)

### GA Results
- **Best Fitness**: F1 promedio en validación cruzada
- **Generation Tracking**: Evolución del mejor fitness por generación
- **Convergence Rate**: Velocidad de mejora por generación

---

## 🎯 Parámetros GA para Ajustar

En el código de integración, puedes modificar:

```python
optimizer = create_ga_optimizer_for_model(
    model_for_ga,
    X_train, y_train,
    param_space,
    cv=5,                    # ← Número de folds CV
    generations=15,          # ← MÁS GENERACIONES = MÁS EXPLORACIÓN (pero +tiempo)
    population_size=20,      # ← MÁS POBLACIÓN = MÁS DIVERSIDAD (pero +tiempo)
    verbose=False            # ← True para ver progreso
)
```

**Recomendaciones:**
- Para exploración rápida: `generations=10, population_size=15`
- Para exploración completa: `generations=20, population_size=25`
- Para producción: `generations=25, population_size=30`

---

## 📈 Salida Esperada

```
════════════════════════════════════════════════════════════════════════════════
ANÁLISIS DE OVERFITTING CON SCORES EN TRAIN/TEST
════════════════════════════════════════════════════════════════════════════════

📊 LogisticRegression
   ✅ SIN OVERFITTING
   f1: Train=0.8500, Test=0.8420, Gap=0.0080 ✓
   accuracy: Train=0.8600, Test=0.8530, Gap=0.0070 ✓
   roc_auc: Train=0.9100, Test=0.9050, Gap=0.0050 ✓

════════════════════════════════════════════════════════════════════════════════
OPTIMIZACIÓN CON ALGORITMOS GENÉTICOS (DEAP)
════════════════════════════════════════════════════════════════════════════════

🧬 GA: LogisticRegression ...
============================================================
Algoritmo Genético: 15 generaciones
============================================================
Gen  0 | Best: 0.7911 | Avg: 0.7822 ± 0.0089
Gen  5 | Best: 0.7945 | Avg: 0.7889 ± 0.0056
Gen 14 | Best: 0.7958 | Avg: 0.7912 ± 0.0046
✓ LogisticRegression completado en 45.2s

════════════════════════════════════════════════════════════════════════════════
COMPARACIÓN DE MÉTODOS DE OPTIMIZACIÓN
════════════════════════════════════════════════════════════════════════════════

     Modelo             Grid_F1  Grid_Time  Optuna_F1  Optuna_Time  GA_F1  GA_Time
0  LogisticRegression   0.7911       3.5     0.7924         12.1  0.7958    45.2
1  RandomForest         0.8840     953.9     0.8845        120.4  0.8862    65.3
2  GradientBoosting     0.8755      45.2     0.8760         35.8  0.8768    52.1
```

---

## ✅ Verificación

Para verificar que todo está instalado correctamente:

```python
python scripts/verify_overfitting_and_ga.py
```

Deberías ver:
```
✓ Módulo cargado exitosamente
✓ DEAP instalado (Algoritmos Genéticos)
✓ Funciones disponibles para integración en notebook
```

---

## 📝 Archivos Generados

- `output/overfitting_analysis.csv` - Análisis de overfitting
- `output/optimization_methods_comparison.csv` - Comparación Grid/Optuna/GA
- `output/optimization_comparison.png` - Gráfico de comparación
- `models/final_model_ga_optimized.pkl` - Modelo final optimizado

---

## 🔍 Interpretación de Resultados

### GA vs GridSearch vs Optuna

| Aspecto | GridSearch | Optuna | GA |
|---------|-----------|--------|-----|
| **Velocidad** | Lento (exhaustivo) | Rápido | Medio |
| **Calidad** | Buena | Muy Buena | Muy Buena |
| **Robustez** | Determinístico | Probabilístico | Probabilístico |
| **Escalabilidad** | Baja | Alta | Media |

**Recomendación**: Usar GA cuando GridSearch es muy lento y quieras mejor exploración que Optuna.

---

## 🚀 Próximos Pasos

1. ✅ Instalar DEAP: `pip install deap`
2. ✅ Copiar los módulos GA al notebook
3. ✅ Ejecutar análisis de overfitting
4. ✅ Ejecutar GA para los 3 mejores modelos
5. ✅ Comparar resultados
6. ✅ Guardar modelo final optimizado
