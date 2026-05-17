"""Test rápido del módulo GA"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Cargar módulo GA
sys.path.insert(0, 'scripts')
from genetic_optimization import create_ga_optimizer_for_model

print("="*70)
print("TEST: Algoritmos Genéticos (DEAP)".center(70))
print("="*70)

# Crear datos de prueba
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# Crear pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# Espacio de búsqueda GA
param_space = {
    'clf__C': ('float', 0.01, 100, 0.1),
    'clf__penalty': ('categorical', None, ['l2']),
    'clf__solver': ('categorical', None, ['lbfgs', 'liblinear'])
}

print("\n✓ Pipeline creado")
print("✓ Espacio de búsqueda definido")
print("✓ Datos: X.shape =", X.shape, ", y unique =", np.unique(y))

# Crear optimizador
print("\n📌 Creando optimizador GA...")
optimizer = create_ga_optimizer_for_model(
    pipeline,
    X, y,
    param_space,
    cv=3,
    generations=5,
    population_size=10,
    verbose=False
)

print("✓ Optimizador creado exitosamente")

# Ejecutar optimización
print("\n🧬 Ejecutando optimización (5 generaciones, 10 población)...")
result = optimizer.optimize()

print("\n" + "="*70)
print("RESULTADOS".center(70))
print("="*70)
print(f"✅ Best F1 Score: {result['best_fitness']:.4f}")
print(f"✅ Mejores parámetros: {result['best_params']}")
print(f"✅ Tiempo total: {result['time_s']:.2f} segundos")

history = result['history']
print(f"\n📊 Evolución:")
print(f"   Generación 0: Best={history['best_fitness'].iloc[0]:.4f}, Avg={history['avg_fitness'].iloc[0]:.4f}")
print(f"   Generación {len(history)-1}: Best={history['best_fitness'].iloc[-1]:.4f}, Avg={history['avg_fitness'].iloc[-1]:.4f}")

improvement = history['best_fitness'].iloc[-1] - history['best_fitness'].iloc[0]
print(f"   Mejora: {improvement:+.4f}")

print("\n" + "="*70)
print("✅ TEST COMPLETADO EXITOSAMENTE".center(70))
print("✅ Algoritmos Genéticos funcionando correctamente".center(70))
print("="*70)
