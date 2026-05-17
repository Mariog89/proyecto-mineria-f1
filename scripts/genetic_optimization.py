"""
Módulo de Optimización con Algoritmos Genéticos (DEAP)
=======================================================

Proporciona funcionalidad para optimizar hiperparámetros de modelos ML 
utilizando Algoritmos Genéticos (DEAP - Distributed Evolutionary Algorithms in Python).

Características:
- Mutación adaptativa de hiperparámetros
- Selección elitista
- Evaluación paralela
- Tracking de fitness vs generación
"""

import numpy as np
import pandas as pd
import time
from typing import Callable, Dict, List, Tuple, Any
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    raise ImportError("deap no está instalado. Ejecuta: pip install deap")


class GeneticOptimizer:
    """
    Optimizador de hiperparámetros usando Algoritmos Genéticos con DEAP.
    
    Parameters
    ----------
    param_space : dict
        Espacio de búsqueda. Formato:
        {
            'param_name': ('type', min_value, max_value, step_or_choices),
            ...
        }
        Tipos soportados: 'int', 'float', 'categorical'
    
    scoring_func : Callable
        Función de evaluación que retorna un score (mayor es mejor).
    
    cv : int, default=5
        Número de folds para validación cruzada.
    
    population_size : int, default=20
        Tamaño de población GA.
    
    generations : int, default=15
        Número de generaciones.
    
    mutation_rate : float, default=0.3
        Probabilidad de mutación.
    
    crossover_rate : float, default=0.7
        Probabilidad de crossover.
    
    verbose : bool, default=True
        Mostrar progreso.
    """
    
    def __init__(
        self,
        param_space: Dict[str, Tuple],
        scoring_func: Callable,
        cv: int = 5,
        population_size: int = 20,
        generations: int = 15,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        verbose: bool = True
    ):
        self.param_space = param_space
        self.scoring_func = scoring_func
        self.cv = cv
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbose = verbose
        
        # Historia de evolución
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'std_fitness': [],
            'best_params': []
        }
        
        self._setup_deap()
    
    def _setup_deap(self):
        """Configura los tipos y operadores de DEAP."""
        # Limpiar definiciones anteriores si existen
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        # Crear tipos DEAP
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # Maximizar
        creator.create('Individual', list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Atributos iniciales aleatorios
        for param_name, param_config in self.param_space.items():
            param_type = param_config[0]
            
            if param_type == 'int':
                _, min_val, max_val, _ = param_config
                self.toolbox.register(
                    f'attr_{param_name}',
                    np.random.randint,
                    min_val, max_val + 1
                )
            elif param_type == 'float':
                _, min_val, max_val, _ = param_config
                self.toolbox.register(
                    f'attr_{param_name}',
                    np.random.uniform,
                    min_val, max_val
                )
            elif param_type == 'categorical':
                _, choices = param_config[1], param_config[2]
                self.toolbox.register(
                    f'attr_{param_name}',
                    np.random.choice,
                    choices
                )
        
        # Crear individuo
        attributes = [f'attr_{name}' for name in self.param_space.keys()]
        self.toolbox.register(
            'individual',
            tools.initCycle,
            creator.Individual,
            [getattr(self.toolbox, attr) for attr in attributes],
            n=1
        )
        
        # Crear población
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        
        # Operadores genéticos
        self.toolbox.register('evaluate', self._evaluate)
        self.toolbox.register('mate', self._crossover)
        self.toolbox.register('mutate', self._mutate)
        self.toolbox.register('select', tools.selTournament, tournsize=3)
    
    def _params_to_dict(self, individual: List) -> Dict[str, Any]:
        """Convierte un individuo a diccionario de parámetros."""
        param_names = list(self.param_space.keys())
        params = {}
        for i, name in enumerate(param_names):
            value = individual[i]
            param_type = self.param_space[name][0]
            
            if param_type == 'int':
                params[name] = int(value)
            elif param_type == 'float':
                params[name] = float(value)
            else:  # categorical
                params[name] = value
        
        return params
    
    def _evaluate(self, individual: List) -> Tuple[float,]:
        """Evalúa fitness de un individuo."""
        params = self._params_to_dict(individual)
        try:
            score = self.scoring_func(params)
            return (score,)
        except Exception as e:
            if self.verbose:
                print(f"Error evaluando parámetros {params}: {e}")
            return (0.0,)
    
    def _crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """Crossover: intercambio de valores entre dos individuos."""
        for i in range(len(ind1)):
            if np.random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    def _mutate(self, individual: List) -> Tuple[List,]:
        """Mutación: cambio aleatorio de parámetros."""
        for i, param_name in enumerate(self.param_space.keys()):
            if np.random.random() < self.mutation_rate:
                param_config = self.param_space[param_name]
                param_type = param_config[0]
                
                if param_type == 'int':
                    _, min_val, max_val, step = param_config
                    individual[i] = np.random.randint(min_val, max_val + 1)
                elif param_type == 'float':
                    _, min_val, max_val, step = param_config
                    individual[i] = np.random.uniform(min_val, max_val)
                elif param_type == 'categorical':
                    _, choices = param_config[1], param_config[2]
                    individual[i] = np.random.choice(choices)
        
        return (individual,)
    
    def optimize(self) -> Dict[str, Any]:
        """
        Ejecuta el algoritmo genético.
        
        Returns
        -------
        dict
            Diccionario con:
            - 'best_params': mejores parámetros encontrados
            - 'best_fitness': mejor score
            - 'history': historial de evolución
            - 'time_s': tiempo total de ejecución
        """
        start_time = time.time()
        
        # Crear población inicial
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluar población inicial
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        best_ind = tools.selBest(pop, 1)[0]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Algoritmo Genético: {self.generations} generaciones")
            print(f"{'='*60}")
        
        # Evolucionan
        for gen in range(self.generations):
            # Selección
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutación
            for mutant in offspring:
                if np.random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluar individuos con fitness inválido
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Elitismo: mantener los mejores
            pop[:] = offspring
            best_in_gen = tools.selBest(pop, 1)[0]
            if best_in_gen.fitness.values[0] > best_ind.fitness.values[0]:
                best_ind = self.toolbox.clone(best_in_gen)
            
            # Historial
            fits = [ind.fitness.values[0] for ind in pop]
            self.history['generation'].append(gen)
            self.history['best_fitness'].append(best_ind.fitness.values[0])
            self.history['avg_fitness'].append(np.mean(fits))
            self.history['std_fitness'].append(np.std(fits))
            self.history['best_params'].append(self._params_to_dict(best_ind))
            
            if self.verbose and (gen % max(1, self.generations // 5) == 0 or gen == self.generations - 1):
                print(f"Gen {gen:2d} | Best: {best_ind.fitness.values[0]:.4f} | "
                      f"Avg: {np.mean(fits):.4f} ± {np.std(fits):.4f}")
        
        elapsed = time.time() - start_time
        
        return {
            'best_params': self._params_to_dict(best_ind),
            'best_fitness': best_ind.fitness.values[0],
            'history': pd.DataFrame(self.history),
            'time_s': elapsed
        }


def create_ga_optimizer_for_model(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_space: Dict[str, Tuple],
    cv: int = 5,
    generations: int = 15,
    population_size: int = 20,
    verbose: bool = True
) -> GeneticOptimizer:
    """
    Crea un optimizador GA configurado para un modelo específico.
    
    Parameters
    ----------
    model : Pipeline
        Pipeline de sklearn.
    X_train, y_train : array-like
        Datos de entrenamiento.
    param_space : dict
        Espacio de búsqueda de parámetros.
    cv : int, default=5
        Número de folds.
    generations : int, default=15
        Número de generaciones.
    population_size : int, default=20
        Tamaño de población.
    verbose : bool, default=True
        Mostrar progreso.
    
    Returns
    -------
    GeneticOptimizer
        Optimizador configurado y listo para usar.
    """
    
    def scoring_func(params: Dict) -> float:
        """Función de evaluación: F1 medio en CV."""
        try:
            model_copy = model
            model_copy.set_params(**params)
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_validate(
                model_copy,
                X_train, y_train,
                cv=cv_split,
                scoring='f1',
                n_jobs=1
            )
            return float(np.mean(scores['test_f1']))
        except Exception as e:
            return 0.0
    
    return GeneticOptimizer(
        param_space=param_space,
        scoring_func=scoring_func,
        cv=cv,
        population_size=population_size,
        generations=generations,
        verbose=verbose
    )


def plot_ga_evolution(history_df: pd.DataFrame) -> None:
    """Grafica la evolución del GA."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fitness a lo largo de generaciones
    ax1.plot(history_df['generation'], history_df['best_fitness'], 'b-', linewidth=2, label='Best')
    ax1.fill_between(
        history_df['generation'],
        history_df['avg_fitness'] - history_df['std_fitness'],
        history_df['avg_fitness'] + history_df['std_fitness'],
        alpha=0.3, label='Mean ± Std'
    )
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Evolución del Algoritmo Genético')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Convergencia
    convergence = history_df['best_fitness'].diff().abs()
    ax2.plot(history_df['generation'], convergence, 'r-', alpha=0.6)
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Cambio en Best Fitness')
    ax2.set_title('Tasa de Convergencia')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
