"""
Script para verificar Overfitting en Validación Cruzada y ejecutar 
Optimización con Algoritmos Genéticos (GA)

Uso:
    python verify_overfitting_and_ga.py <modelo_path> <X_train> <y_train>
"""

import sys
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Importar el módulo GA
sys.path.insert(0, str(Path(__file__).parent))
from genetic_optimization import create_ga_optimizer_for_model


def analyze_overfitting_cv(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    model_name: str = "Modelo"
) -> dict:
    """
    Analiza overfitting en validación cruzada.
    
    Retorna métricas en train y test para detectar overfitting.
    """
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Scorers para múltiples métricas
    scoring = {
        'f1': 'f1',
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }
    
    # Cross-validate con train scores
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv_split,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calcular gaps de overfitting
    overfitting_analysis = {}
    
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        std_test = np.std(test_scores)
        gap = mean_train - mean_test
        
        overfitting_analysis[metric] = {
            'train_mean': round(mean_train, 4),
            'test_mean': round(mean_test, 4),
            'test_std': round(std_test, 4),
            'gap': round(gap, 4),
            'gap_ratio': round(gap / mean_train * 100 if mean_train > 0 else 0, 2),
            'is_overfitting': gap > 0.05  # Umbral: gap > 0.05 sugiere overfitting
        }
    
    # Diagnóstico
    f1_gap = overfitting_analysis['f1']['gap']
    if f1_gap < 0.02:
        diagnosis = "✅ SIN OVERFITTING"
    elif f1_gap < 0.05:
        diagnosis = "⚠️  OVERFITTING LEVE"
    elif f1_gap < 0.10:
        diagnosis = "⚠️⚠️ OVERFITTING MODERADO"
    else:
        diagnosis = "❌ OVERFITTING SEVERO"
    
    return {
        'model_name': model_name,
        'analysis': overfitting_analysis,
        'diagnosis': diagnosis,
        'cv_results': cv_results
    }


def run_genetic_optimization_on_top_models(
    models_dict: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_spaces: dict,
    top_n: int = 3,
    generations: int = 20,
    population_size: int = 25
) -> dict:
    """
    Ejecuta optimización con GA en los top N mejores modelos.
    
    Parameters
    ----------
    models_dict : dict
        {nombre_modelo: (pipeline, param_grid)}
    X_train, y_train : arrays
        Datos de entrenamiento
    param_spaces : dict
        {nombre_modelo: param_space_para_ga}
    top_n : int
        Número de mejores modelos a optimizar
    generations : int
        Generaciones del GA
    population_size : int
        Tamaño de población GA
    
    Returns
    -------
    dict
        Resultados GA para cada modelo
    """
    
    ga_results = {}
    
    for model_name, param_space in param_spaces.items():
        if model_name not in models_dict:
            print(f"⚠️  Skipping {model_name}: no en models_dict")
            continue
        
        model_pipeline = models_dict[model_name]
        
        print(f"\n{'='*60}")
        print(f"GA: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Crear optimizador
            optimizer = create_ga_optimizer_for_model(
                model_pipeline,
                X_train, y_train,
                param_space,
                cv=5,
                generations=generations,
                population_size=population_size,
                verbose=True
            )
            
            # Ejecutar optimización
            start = time.time()
            result = optimizer.optimize()
            elapsed = time.time() - start
            
            ga_results[model_name] = {
                'best_params': result['best_params'],
                'best_fitness': result['best_fitness'],
                'history': result['history'],
                'time_s': elapsed
            }
            
            print(f"✓ {model_name} completado en {elapsed:.1f}s")
            print(f"  Best F1: {result['best_fitness']:.4f}")
            
        except Exception as e:
            print(f"✗ Error en {model_name}: {e}")
            ga_results[model_name] = {'error': str(e)}
    
    return ga_results


def compare_optimization_methods(
    grid_results: dict,
    optuna_results: dict,
    ga_results: dict
) -> pd.DataFrame:
    """
    Compara GridSearch vs Optuna vs GA.
    """
    comparison = []
    
    all_models = set(
        list(grid_results.keys()) + 
        list(optuna_results.keys()) + 
        list(ga_results.keys())
    )
    
    for model_name in all_models:
        row = {'Modelo': model_name}
        
        # GridSearch
        if model_name in grid_results:
            row['Grid_F1'] = round(grid_results[model_name].get('best_cv_f1', np.nan), 4)
            row['Grid_Time'] = round(grid_results[model_name].get('time_s', np.nan), 2)
        else:
            row['Grid_F1'] = np.nan
            row['Grid_Time'] = np.nan
        
        # Optuna
        if model_name in optuna_results:
            row['Optuna_F1'] = round(optuna_results[model_name].get('best_cv_f1', np.nan), 4)
            row['Optuna_Time'] = round(optuna_results[model_name].get('time_s', np.nan), 2)
        else:
            row['Optuna_F1'] = np.nan
            row['Optuna_Time'] = np.nan
        
        # GA
        if model_name in ga_results and 'error' not in ga_results[model_name]:
            row['GA_F1'] = round(ga_results[model_name].get('best_fitness', np.nan), 4)
            row['GA_Time'] = round(ga_results[model_name].get('time_s', np.nan), 2)
        else:
            row['GA_F1'] = np.nan
            row['GA_Time'] = np.nan
        
        comparison.append(row)
    
    comparison_df = pd.DataFrame(comparison)
    
    # Determinar mejor método por modelo
    comparison_df['Best_Method'] = comparison_df.apply(
        lambda row: max(
            [(col.split('_')[0], row[col]) for col in ['Grid_F1', 'Optuna_F1', 'GA_F1'] if pd.notna(row[col])],
            key=lambda x: x[1]
        )[0] if any(pd.notna(row[col]) for col in ['Grid_F1', 'Optuna_F1', 'GA_F1']) else 'N/A',
        axis=1
    )
    
    return comparison_df


def print_overfitting_report(analysis_list: list) -> None:
    """Imprime reporte de overfitting."""
    print("\n" + "="*80)
    print("ANÁLISIS DE OVERFITTING EN VALIDACIÓN CRUZADA".center(80))
    print("="*80)
    
    for analysis in analysis_list:
        print(f"\n📊 {analysis['model_name']}")
        print(f"   Diagnóstico: {analysis['diagnosis']}")
        print(f"\n   Métrica      | Train Mean | Test Mean | Gap   | % Gap  | Overfitting")
        print(f"   {'─'*73}")
        
        for metric, values in analysis['analysis'].items():
            status = "✓ No" if not values['is_overfitting'] else "⚠️  Sí"
            print(f"   {metric:12} | {values['train_mean']:10.4f} | {values['test_mean']:9.4f} | "
                  f"{values['gap']:5.4f} | {values['gap_ratio']:5.1f}% | {status}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Verificación de Overfitting y Optimización con GA          ║
    ║  Genera reporte HTML automático en output/                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("✓ Módulo cargado exitosamente")
    print("✓ DEAP instalado (Algoritmos Genéticos)")
    print("✓ Funciones disponibles para integración en notebook")
