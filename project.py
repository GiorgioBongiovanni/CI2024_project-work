import numpy as np
import math
import random
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

# Initialize the global executor
executor = ProcessPoolExecutor(max_workers=24)  # Usa i 24 core fisici


# Dizionario delle operazioni consentite (nodi interni dell'albero)
OPERATIONS = {
    "add": (lambda x, y: x + y, 2),
    "mul": (lambda x, y: x * y, 2),
    "sub": (lambda x, y: x - y, 2),
    "div": (lambda x, y: x / y if abs(y) > 1e-6 else 1, 2),
    "neg": (lambda x: -x, 1),
    "log": (lambda x: math.log(max(1e-6, abs(x))), 1),
    "exp": (lambda x: math.exp(min(50, max(-50, x))), 1),
    "sqrt": (lambda x: math.sqrt(max(0, x)), 1),
    "abs": (abs, 1),
    "max": (max, 2),
    "min": (min, 2),
    "sin": (lambda x: math.sin(min(1e6, max(-1e6, x))), 1),
    "square": (lambda x: min(1e6, x * x), 1)
}


MUTATION_PROBABILITIES = {
    "subtree": 0.9,  
    "hoist": 0.05,  
    "point": 0.05    
}

class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"{self.value}({', '.join(str(c) for c in self.children)})"

def weighted_choice(items, weights):
    total_weight = sum(weights)
    rand_val = random.uniform(0, total_weight)
    for item, weight in zip(items, weights):
        if rand_val < weight:
            return item
        rand_val -= weight
    return items[-1]

def calculate_feature_importance(X, y):
    mi_scores = mutual_info_regression(X.T, y)
    importance = np.abs(mi_scores) / np.sum(np.abs(mi_scores))
    return importance

def generate_terminals_with_importance(num_variables, importance):
    terminals = [f"x{i+1}" for i in range(num_variables)]
    weights = importance.tolist()
    return terminals, weights

def generate_tree(depth, terminals, terminal_weights):
    if depth == 0 or random.random() < 0.3:
        terminal = weighted_choice(terminals + [round(random.uniform(-10, 10), 2)], terminal_weights + [0.1])
        return Node(terminal if isinstance(terminal, str) and terminal.startswith("x") else float(terminal))
    op_name, (_, arity) = random.choice(list(OPERATIONS.items()))
    children = [generate_tree(depth - 1, terminals, terminal_weights) for _ in range(arity)]
    return Node(op_name, children)

def trim_tree(node, max_depth, terminals):
    if max_depth == 0:
        return Node(random.choice(terminals + [round(random.uniform(-10, 10), 2)]))
    if node.is_leaf():
        return node
    return Node(node.value, [trim_tree(child, max_depth - 1, terminals) for child in node.children])

def generate_full_tree(depth, terminals, terminal_weights):
    if depth == 0:
        terminal = weighted_choice(terminals + [round(random.uniform(-10, 10), 2)], terminal_weights + [0.1])
        return Node(terminal if isinstance(terminal, str) and terminal.startswith("x") else float(terminal))
    op_name, (_, arity) = random.choice(list(OPERATIONS.items()))
    children = [generate_full_tree(depth - 1, terminals, terminal_weights) for _ in range(arity)]
    return Node(op_name, children)

def generate_grow_tree(min_depth, max_depth, terminals, terminal_weights, current_depth=0):
    if current_depth < min_depth:
        op_name, (_, arity) = random.choice(list(OPERATIONS.items()))
        children = [generate_grow_tree(min_depth, max_depth, terminals, terminal_weights, current_depth + 1) 
                    for _ in range(arity)]
        return Node(op_name, children)

    if current_depth == max_depth or random.random() < 0.3:
        terminal = weighted_choice(terminals + [round(random.uniform(-10, 10), 2)], terminal_weights + [0.1])
        return Node(terminal if isinstance(terminal, str) and terminal.startswith("x") else float(terminal))

    op_name, (_, arity) = random.choice(list(OPERATIONS.items()))
    children = [generate_grow_tree(min_depth, max_depth, terminals, terminal_weights, current_depth + 1) 
                for _ in range(arity)]
    return Node(op_name, children)

def generate_population(pop_size, min_depth, max_depth, terminals, terminal_weights):
    population = []
    for i in range(pop_size):
        if i < pop_size // 2:
            population.append(generate_full_tree(max_depth, terminals, terminal_weights))
        else:
            population.append(generate_grow_tree(min_depth, max_depth, terminals, terminal_weights))
    return population

def tournament_selection(population, fitnesses, k=150):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

def random_subtree(node):
    all_nodes = get_all_nodes(node)
    return random.choice(all_nodes)

def get_all_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes

def replace_subtree(tree, target, replacement):
    if tree == target:
        tree.value, tree.children = replacement.value, replacement.children
        return
    for i, child in enumerate(tree.children):
        if child == target:
            tree.children[i] = replacement
            return
        replace_subtree(child, target, replacement)

def crossover(parent1, parent2, max_depth, terminals):
    p1, p2 = deepcopy(parent1), deepcopy(parent2)
    node1 = random_subtree(p1)
    node2 = random_subtree(p2)
    replace_subtree(p1, node1, trim_tree(deepcopy(node2), max_depth - get_tree_depth(p1, node1), terminals))
    return p1

def get_tree_depth(node, target, depth=0):
    if node == target:
        return depth
    for child in node.children:
        d = get_tree_depth(child, target, depth + 1)
        if d is not None:
            return d
    return None

def subtree_mutation(tree, terminals, terminal_weights, max_depth):
    subtree = random_subtree(tree)
    residual_depth = max_depth - get_tree_depth(tree, subtree)
    if residual_depth <= 1:
        new_subtree = Node(weighted_choice(terminals + [round(random.uniform(-10, 10), 2)], terminal_weights))
    else:
        new_subtree = generate_tree(random.randint(1, residual_depth), terminals, terminal_weights)
    replace_subtree(tree, subtree, new_subtree)
    return tree

def hoist_mutation(tree):
    subtree = random_subtree(tree)
    if subtree.is_leaf():
        return tree
    replacement = random.choice(subtree.children)
    replace_subtree(tree, subtree, replacement)
    return tree

def point_mutation(tree, terminals, terminal_weights):
    mutation_target = random_subtree(tree)
    if mutation_target.is_leaf():
        mutation_target.value = weighted_choice(terminals + [round(random.uniform(-10, 10), 2)], terminal_weights)
    else:
        current_arity = len(mutation_target.children)
        compatible_operations = [op for op, (_, arity) in OPERATIONS.items() if arity == current_arity]
        if compatible_operations:
            mutation_target.value = random.choice(compatible_operations)
    return tree

def select_mutation():
    mutation_types = list(MUTATION_PROBABILITIES.keys())
    probabilities = list(MUTATION_PROBABILITIES.values())
    return random.choices(mutation_types, probabilities)[0]

def apply_mutation(child, mutation_type, terminals, terminal_weights, max_depth):
    if mutation_type == "subtree":
        return subtree_mutation(child, terminals, terminal_weights, max_depth)
    elif mutation_type == "hoist":
        return hoist_mutation(child)
    else:  # point
        return point_mutation(child, terminals, terminal_weights)

def evaluate_node(node, **kwargs):
    if node.is_leaf():
        return float(kwargs.get(node.value, node.value))
    func, _ = OPERATIONS[node.value]
    results = [evaluate_node(child, **kwargs) for child in node.children]
    return func(*results)


def evaluate_tree(tree, X, y_true, parsimony_coefficient=0.1):
    try:
        y_pred = []
        for j in range(X.shape[1]):
            variables = {f"x{i+1}": X[i, j] for i in range(X.shape[0])}
            pred = evaluate_node(tree, **variables)
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        mse = np.mean((y_pred - y_true) ** 2)

        # Penalizzazione della complessità
        complexity_penalty = parsimony_coefficient * len(get_all_nodes(tree))
        return mse + complexity_penalty
    except:
        return float('inf')


def evaluate_tree_wrapper(args):
    tree, X, y_true = args
    return evaluate_tree(tree, X, y_true)

# Funzione globale per valutare un batch di individui
def evaluate_fitness_batch(args):
    population, X, y_true, batch_start, batch_size = args
    batch_end = min(batch_start + batch_size, len(population))  # Calcola il limite del batch
    fitnesses = []
    for i in range(batch_start, batch_end):
        fitnesses.append(evaluate_tree(population[i], X, y_true))
    return fitnesses

# Funzione per valutare la popolazione in parallelo
def evaluate_population(population, X, y_true, num_cores=24):
    pop_size = len(population)
    # Calcola quanti elementi deve processare ogni batch
    items_per_core = pop_size // num_cores
    remaining_items = pop_size % num_cores
    
    # Prepara gli argomenti per ogni batch
    args = []
    start_idx = 0
    
    # Distribuisci gli elementi tra i core
    for i in range(num_cores):
        # Aggiungi un elemento extra ai primi remaining_items core
        current_batch_size = items_per_core + (1 if i < remaining_items else 0)
        if current_batch_size > 0:  # Crea il batch solo se ha elementi da processare
            args.append((population, X, y_true, start_idx, current_batch_size))
            start_idx += current_batch_size

    # Parallelizza la valutazione
    all_fitness_batches = list(executor.map(evaluate_fitness_batch, args))

    # Unisci tutti i risultati dei batch in una singola lista
    fitnesses = [fitness for batch in all_fitness_batches for fitness in batch]
    
    return fitnesses

def create_offspring_batch(args):
    population, fitnesses, terminals, terminal_weights, max_depth, batch_size = args
    offspring = []
    for _ in range(batch_size):
        parent1 = tournament_selection(population, fitnesses)
        child = deepcopy(parent1)
        
        if random.random() <= 0.2:  # Mutation
            mutation_type = select_mutation()
            child = apply_mutation(child, mutation_type, terminals, terminal_weights, max_depth)
        else:  # Crossover
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2, max_depth, terminals)
        
        offspring.append(child)
    return offspring


def generate_next_population(population, fitnesses, terminals, terminal_weights, max_depth, pop_size, elite_fraction=0.05, num_cores=24):
    """
    Genera la prossima generazione mantenendo una frazione di elites.
    """
    # Numero di elites da mantenere
    num_elites = max(1, int(pop_size * elite_fraction))
    
    # Seleziona le elites (individui con fitness migliore)
    elite_indices = np.argsort(fitnesses)[:num_elites]
    elites = [population[i] for i in elite_indices]
    
    # Calcola quanti nuovi individui dobbiamo generare
    num_offspring_needed = pop_size - num_elites
    
    # Calcola quanti individui deve generare ogni batch
    items_per_core = num_offspring_needed // num_cores
    remaining_items = num_offspring_needed % num_cores
    
    # Prepara gli argomenti per ogni batch
    args = []
    total_offspring = 0
    
    # Distribuisci gli elementi tra i core
    for i in range(num_cores):
        # Aggiungi un elemento extra ai primi remaining_items core
        current_batch_size = items_per_core + (1 if i < remaining_items else 0)
        if current_batch_size > 0:  # Crea il batch solo se ha elementi da generare
            args.append((population, fitnesses, terminals, terminal_weights, max_depth, current_batch_size))
            total_offspring += current_batch_size
    
    # Genera gli offspring in parallelo
    all_offspring_batches = list(executor.map(create_offspring_batch, args))
    next_population = [individual for batch in all_offspring_batches for individual in batch]
    
    # Aggiungi le elites
    next_population = elites + next_population
    
    assert len(next_population) == pop_size, f"Population size mismatch: {len(next_population)} != {pop_size}"
    return next_population


def extract_constants(tree):
    """
    Raccoglie le costanti numeriche dai nodi foglia dell'albero.

    Args:
        tree: L'albero da analizzare.

    Returns:
        Lista di valori costanti nei nodi foglia.
    """
    if tree.is_leaf():
        if isinstance(tree.value, (int, float)):
            return [tree.value]
        return []
    constants = []
    for child in tree.children:
        constants.extend(extract_constants(child))
    return constants


def replace_constants(tree, constants, index=0):
    """
    Sostituisce i valori numerici nei nodi foglia con nuovi valori.

    Args:
        tree: L'albero in cui sostituire i valori.
        constants: Lista di costanti con cui sostituire i valori esistenti.
        index: Indice corrente nella lista delle costanti.

    Returns:
        L'albero aggiornato e il prossimo indice.
    """
    if tree.is_leaf():
        if isinstance(tree.value, (int, float)):
            tree.value = constants[index]
            index += 1
        return tree, index
    for i, child in enumerate(tree.children):
        tree.children[i], index = replace_constants(child, constants, index)
    return tree, index


def optimize_constants(tree, X, y_true):
    """
    Ottimizza le costanti numeriche nei nodi foglia dell'albero per minimizzare l'errore.

    Args:
        tree: L'albero da ottimizzare.
        X: Input del dataset.
        y_true: Output atteso del dataset.

    Returns:
        L'albero con costanti ottimizzate.
    """
    constants = extract_constants(tree)
    if not constants:
        return tree  # Nessuna costante da ottimizzare

    def loss_function(new_constants):
        """
        Funzione obiettivo per minimizzare l'errore tra output predetto e vero.
        """
        temp_tree, _ = replace_constants(deepcopy(tree), new_constants)
        y_pred = []
        for j in range(X.shape[1]):
            variables = {f"x{i+1}": X[i, j] for i in range(X.shape[0])}
            y_pred.append(evaluate_node(temp_tree, **variables))
        y_pred = np.array(y_pred)
        return np.mean((y_pred - y_true) ** 2)

    # Ottimizzazione con scipy.optimize.minimize
    result = minimize(loss_function, constants, method='L-BFGS-B')
    optimized_constants = result.x

    # Aggiorna l'albero con le costanti ottimizzate
    tree, _ = replace_constants(tree, optimized_constants)
    return tree


def genetic_algorithm(X, y_true, pop_size=1500, generations=400, min_depth=2, max_depth=6, 
                     elite_fraction=0.05, stagnation_limit=50, grow_interval=20):
    """
    Algoritmo genetico modificato con:
    1. Aggiunta di individui grow ogni 20 generazioni
    2. Sostituzione di metà popolazione dopo 50 generazioni senza miglioramenti
    3. Mantenimento del 5% di elite nella generazione della nuova popolazione
    """
    num_variables = X.shape[0]
    importance = calculate_feature_importance(X, y_true)
    terminals, terminal_weights = generate_terminals_with_importance(num_variables, importance)
    population = generate_population(pop_size, min_depth, max_depth, terminals, terminal_weights)
    
    # Variabili per monitorare la stagnazione
    best_fitness = float('inf')
    stagnation_counter = 0
    original_pop_size = pop_size  # Memorizza la dimensione originale della popolazione

    for gen in range(1, generations + 1):
        # Valuta la popolazione corrente
        fitnesses = evaluate_population(population, X, y_true)
        current_best_fitness = min(fitnesses)

        # Controlla se la fitness è migliorata
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        print(f"Generation {gen}: Best Fitness = {current_best_fitness:.5f}, Population Size = {len(population)}")

        # 1. Aggiunta di individui grow ogni 20 generazioni
        if gen % grow_interval == 0:
            grow_size = len(population) // 2  # Metà della popolazione corrente
            new_individuals = [
                generate_grow_tree(min_depth, max_depth, terminals, terminal_weights)
                for _ in range(grow_size)
            ]
            population.extend(new_individuals)
            
            # Rivaluta le fitness includendo i nuovi individui
            fitnesses = evaluate_population(population, X, y_true)
            
            print(f"Generation {gen}: Added {grow_size} grow individuals. New population size: {len(population)}")

        # 2. Gestione della stagnazione (50 generazioni senza miglioramenti)
        if stagnation_counter >= stagnation_limit:
            print(f"Generation {gen}: Stagnation detected. Replacing random half of population...")
            # Seleziona casualmente metà degli indici della popolazione
            indices_to_replace = random.sample(range(len(population)), len(population) // 2)
            
            # Crea nuovi individui grow per sostituire quelli selezionati
            for idx in indices_to_replace:
                population[idx] = generate_grow_tree(min_depth, max_depth, terminals, terminal_weights)
            
            stagnation_counter = 0

        # 3. Genera la prossima generazione mantenendo il 5% di elite
        population = generate_next_population(
            population, 
            fitnesses, 
            terminals, 
            terminal_weights, 
            max_depth, 
            original_pop_size,  # Mantiene la dimensione originale
            elite_fraction=elite_fraction  # 5% di elite
        )

    # Valutazione finale
    fitnesses = evaluate_population(population, X, y_true)
    best_tree = population[np.argmin(fitnesses)]
    print(f"\nBest solution before constant optimization: {best_tree}")

    # Ottimizza le costanti della soluzione migliore
    best_tree = optimize_constants(best_tree, X, y_true)

    # Valuta la soluzione ottimizzata
    final_fitness = evaluate_tree(best_tree, X, y_true)
    print(f"\nBest solution after constant optimization: {best_tree}")
    print(f"Final Fitness: {final_fitness:.5f}")

    return best_tree


if __name__ == "__main__":
    problem = np.load("data/problem_7.npz")
    X = problem["x"]
    y = problem["y"]
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    best_tree = genetic_algorithm(X, y)

    print("\nBest solution:", best_tree)
    print("Size:", len(get_all_nodes(best_tree)))

    executor.shutdown()
