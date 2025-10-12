# Assignment 1 - Traveling Salesman Problem (TSP) using Genetic Algorithm
import random
import math
import tsplib95
import pandas as pd
from prim import solve_tsp_prim
from datetime import datetime
import sys

from tsp.optimal import bays29opt
from tsp.optimal import berlin52opt
from tsp.optimal import eil76opt
from tsp.optimal import ulysses16opt

log_filename = f"tsp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, "w")

class DualOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = DualOutput(sys.__stdout__, log_file)

# 1 - Generate random paths (initial solutions)
def random_paths(Destinations, num_paths): 
    paths = []
    for _ in range(num_paths): # Generate a random path
        path = list(range(1, Destinations)) # Exclude the starting point (0)
        random.shuffle(path) # Shuffle the destinations
        path = [0] + path # Add the starting point at the beginning
        paths.append(path) # Add the path to the list of paths
    return paths

# Example usage
# print(random_paths(5, 10))  # Output: List of 10 random paths

# 2 - Calculate the total distance of a path
def calculate_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1): 
        total_distance += distance_matrix[path[i]][path[i + 1]] # Sum the distances between consecutive points
    total_distance += distance_matrix[path[-1]][path[0]] # Return to the starting point
    return total_distance

# Example distance matrix for 5 destinations
# distance_matrix = [
#     [0, 2, 10, 12, 5], # Using math.inf to represent no direct path
#     [2, 0, 4, 8, 10],
#     [10, 4, 0, 3, 3],
#     [12, 8, 3, 0, 10],
#     [5, 10, 3, 10, 0],
# ]
# Example usage
# path = [0, 1, 2, 3, 4]
# print(calculate_distance(path, distance_matrix))  # Output: Total distance of the path

# 3 - Select the best paths based on distance
def select_best_paths(paths, distance_matrix, num_best):
    paths_with_distances = [(path, calculate_distance(path, distance_matrix)) for path in paths]
    paths_with_distances.sort(key=lambda x: x[1]) # Sort by distance
    best_paths = [path for path, distance in paths_with_distances[:num_best]] # Select the best paths
    return best_paths

# Example usage
# paths = random_paths(5, 10)
# best_paths = select_best_paths(paths, distance_matrix, 5)
# print(best_paths)  # Output: List of best paths

# 4 - Crossover between two paths to create a new path
def crossover(path1, path2):
    size = len(path1)
    start, end = sorted(random.sample(range(1, size), 2)) # Select two crossover points
    new_path = [None] * size
    new_path[start:end] = path1[start:end] # Copy a segment from the first parent

    pointer = 0
    for i in range(size):
        if new_path[i] is None: # Fill in the remaining positions with genes from the second parent
            while path2[pointer] in new_path:
                pointer += 1
            new_path[i] = path2[pointer]
            pointer += 1

    return new_path 

# Example usage
# path1 = [0, 1, 2, 3, 4]
# path2 = [0, 4, 3, 2, 1]
# new_path = crossover(path1, path2)
# print(new_path)  # Output: New path created by crossover

# 5 - Mutate a path by swapping two destinations
def mutate(path, mutation_rate):
    for i in range(1, len(path)): # Avoid mutating the starting point (0)
        if random.random() < mutation_rate: # Mutate with a certain probability
            j = random.randint(1, len(path) - 1) # Select another position to swap with
            path[i], path[j] = path[j], path[i] # Swap the two destinations
    return path

# Tournament selection to choose a parent path
def tournament_select(paths, distance_matrix, k):   
    competitors = random.sample(paths, k if k <= len(paths) else len(paths)) # Select k random paths
    competitors.sort(key=lambda p: calculate_distance(p, distance_matrix)) # Sort by distance (lower is better)
    return competitors[0]

# 6 - Main genetic algorithm function to solve TSP
def genetic_algorithm_tsp(distance_matrix, num_paths, num_generations, mutation_rate, num_best, k_tournament): # Parameters
    Destinations = len(distance_matrix)
    paths = random_paths(Destinations, num_paths) # Step 1: Generate initial random paths
    
    print(f"Starting genetic algorithm with {num_paths} paths for {num_generations} generations...")
    print(f"Problem size: {Destinations} cities")
    print("-" * 50)

    for generation in range(num_generations):
        best_paths = select_best_paths(paths, distance_matrix, num_best,) # Step 3: Select the best paths
        new_paths = best_paths.copy()

        while len(new_paths) < num_paths:
            
            parent1 = tournament_select(paths, distance_matrix, k_tournament) # Select first parent via tournament
            parent2 = tournament_select(paths, distance_matrix, k_tournament) # Select second parent via tournament
           
            attempts = 0
            while parent2 is parent1 and attempts < 3:
                parent2 = tournament_select(paths, distance_matrix, k_tournament)
                attempts += 1
            # ---------

            child = crossover(parent1, parent2) # Step 4: Crossover to create a new path
            child = mutate(child, mutation_rate) # Step 5: Mutate the new path
            new_paths.append(child)

        paths = new_paths
        
        # Print progress every 100 generations or at the end
        if generation % 100 == 0 or generation == num_generations - 1:
            current_best = select_best_paths(paths, distance_matrix, 1)[0]
            current_distance = calculate_distance(current_best, distance_matrix)
            print(f"Generation {generation:4d}: Best distance = {current_distance:.2f}")

    best_path = select_best_paths(paths, distance_matrix, 1)[0] # Get the best path from the final generation
    best_distance = calculate_distance(best_path, distance_matrix)
    return best_path, best_distance

def parse_tsplib_full_matrix(text, n=None): # n x n full distance matrix from TSPLIB EXPLICIT format
    tokens = text.split()
    weights = list(map(int, tokens))
    if n is None:
        r = int(len(weights) ** 0.5)
        if r * r != len(weights):
            raise ValueError("number of values is not a perfect square; pass n=")
        n = r
    if len(weights) != n * n:
        raise ValueError(f"expected {n*n} numbers, received {len(weights)}")
    m = [weights[i*n:(i+1)*n] for i in range(n)]
    for i in range(n):
        m[i][i] = 0
    return m


def get_matrix(filename: str) -> list[list[int]]:
    """Returns TSPLIB95 adjacency matrix"""
    problem = tsplib95.load(filename) # Load the wanted TSP problem
    nodes = list(problem.get_nodes()) # Extract the list of nodes
    n = len(nodes)
    matrix = [[problem.get_weight(i, j) for j in nodes] for i in nodes] # Build the matrix out of the nodes
    df = pd.DataFrame(matrix, index=nodes, columns=nodes) # Convert the distance matrix into a pandas DataFrame
    return df.values.tolist()


# Example usage
# best_path, best_distance = genetic_algorithm_tsp(distance_matrix, num_paths=20, num_generations=100, mutation_rate=0.1, num_best=5)
# print("Best path:", best_path)
# print("Best distance:", best_distance)

# def generate_distance_matrix(size, max_distance=100):
#     """Generate a random distance matrix for testing."""
#     matrix = [[0 if i == j else random.randint(1, max_distance) for j in range(size)] for i in range(size)]
#     # Make the matrix symmetric
#     for i in range(size):
#         for j in range(i + 1, size):
#             matrix[j][i] = matrix[i][j]
#     return matrix

# -------------------------
# EXECUTION
# -------------------------

if __name__ == "__main__":

    distance_matrix = get_matrix('tsp/berlin52.tsp')
        # --- Prim baseline ---
    prim_path, prim_distance = solve_tsp_prim(distance_matrix, start=0)
    print("\n" + "=" * 50)
    print("Running Prim's algorithm as a baseline")
    print(f"Prim distance: {prim_distance}")
    print("=" * 50)
  
    start_time = datetime.now()
    best_path, best_distance = genetic_algorithm_tsp(
        distance_matrix,
        num_paths=400,
        num_generations=1000,
        mutation_rate=0.01,
        num_best=4,
        k_tournament=5
    )
    end_time = datetime.now()
    ga_duration = (end_time - start_time).total_seconds()
    difference = prim_distance - best_distance

    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    print("Best path:", best_path)
    print("GA Best distance:", best_distance)
    print("Prim baseline:", prim_distance)
    print("Optimal distance:", calculate_distance(berlin52opt, distance_matrix))
    print(f"Our GA was: {difference}", "closer to the optima")
    print(f"GA time: {ga_duration} seconds" )
    print(f"Number of cities visited: {len(best_path)}")
    print("=" * 50)
    print("Done!")