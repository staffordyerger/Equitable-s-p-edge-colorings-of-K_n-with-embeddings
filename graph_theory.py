import itertools
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import sys
import multiprocessing
import time

# =======================
# Adjustable Parameters
# =======================

# Define minimum b(n) thresholds as a list of tuples.
# Each tuple contains (condition_function, min_b).
# The list should be ordered from most specific to general.
MIN_B_THRESHOLDS = [
    (lambda x: x == 27, 7),
    (lambda x: x == 25, 7),
    (lambda x: x == 21, 7),             # Specific case for n=21
    (lambda x: 27 <= x <= 31, 5),       # Specific range for 27 <= n <= 31
    (lambda x: x >= 13, 3)              # General case for n >= 13
]

# Default minimum b(n) for n < 13
DEFAULT_MIN_B = 1

# Directory to save the colorings
OUTPUT_DIR = 'Equitable_Complete_Graphs'  # Directory to save images and txt files

# Maximum number of solutions per (n, t, b) configuration
MAX_SOLUTIONS_PER_CONFIGURATION = 1  # Adjust as needed

# Timeout in seconds (30 minutes)
TIMEOUT_PER_CONFIGURATION = 30 * 60  # 1800 seconds

# =======================
# Helper Functions
# =======================

def get_min_b(n, thresholds, default_min_b):
    """
    Determine the minimum b(n) for a given n based on predefined thresholds.

    Parameters:
    - n (int): The number of vertices in the complete graph.
    - thresholds (list of tuples): Each tuple contains (condition_function, min_b).
    - default_min_b (int): The default minimum b(n) if no threshold is met.

    Returns:
    - int: The minimum b(n) for the given n.
    """
    for condition, min_b in thresholds:
        if condition(n):
            return min_b
    return default_min_b

def find_t_b_for_n(n, min_b):
    """
    For a given n, find all possible (t, b(n)) pairs such that:
    n = 2t * b(n) + 1,
    where t >= 1 and b(n) is an odd positive integer >= min_b.

    Parameters:
    - n (int): Number of vertices in the complete graph.
    - min_b (int): Minimum value for b(n).

    Returns:
    - list of tuples: Each tuple is (t, b(n)).
    """
    t_b_pairs = []
    for t in range(1, n):
        # Calculate b(n)
        b = (n - 1) / (2 * t)
        if b.is_integer() and b % 2 == 1 and b >= min_b:
            b = int(b)
            t_b_pairs.append((t, b))
    return t_b_pairs

def backtrack_coloring(edges, edge_index, edge_colors, color_usage, distinct_color_count, b, p, s, solutions, max_solutions):
    """
    Recursive backtracking function to assign colors to edges with constraints:
    - Each vertex uses exactly p distinct colors.
    - Each color appears exactly b times per vertex.

    Parameters:
    - edges (list of tuples): List of edges in the graph.
    - edge_index (int): Current index in the edges list.
    - edge_colors (dict): Current edge-color assignments.
    - color_usage (dict): Color usage per vertex.
    - distinct_color_count (dict): Number of distinct colors used per vertex.
    - b (int): Number of times each color should appear per vertex.
    - p (int): Number of distinct colors per vertex.
    - s (int): Total number of colors available.
    - solutions (list): List to store valid colorings.
    - max_solutions (int): Maximum number of solutions to find.
    """
    if len(solutions) >= max_solutions:
        return  # Stop if we've found the desired number of solutions
    if edge_index == len(edges):
        # All edges have been colored
        # Verify that each vertex uses exactly p colors, each appearing b times
        for node in color_usage:
            if distinct_color_count[node] != p:
                return
            for color_count in color_usage[node].values():
                if color_count != b:
                    return
        # Found a valid coloring
        solutions.append(edge_colors.copy())
        return

    u, v = edges[edge_index]

    # Try existing colors first to maximize reuse
    available_colors = sorted(range(s), key=lambda c: (c not in color_usage[u], c not in color_usage[v]))

    for color in available_colors:
        # Check if assigning this color is valid for both vertices
        can_assign = True
        # For vertex u
        if color in color_usage[u]:
            if color_usage[u][color] >= b:
                can_assign = False
        else:
            if distinct_color_count[u] >= p:
                can_assign = False
        # For vertex v
        if color in color_usage[v]:
            if color_usage[v][color] >= b:
                can_assign = False
        else:
            if distinct_color_count[v] >= p:
                can_assign = False

        if not can_assign:
            continue  # Skip this color

        # Assign color to the edge
        edge_colors[(u, v)] = color
        # Update color_usage for u
        if color not in color_usage[u]:
            distinct_color_count[u] += 1
        color_usage[u][color] += 1
        # Update color_usage for v
        if color not in color_usage[v]:
            distinct_color_count[v] += 1
        color_usage[v][color] += 1

        # Proceed to the next edge
        backtrack_coloring(edges, edge_index + 1, edge_colors, color_usage, distinct_color_count, b, p, s, solutions, max_solutions)

        # Undo the assignment (backtrack)
        del edge_colors[(u, v)]
        # Update color_usage for u
        color_usage[u][color] -= 1
        if color_usage[u][color] == 0:
            del color_usage[u][color]
            distinct_color_count[u] -= 1
        # Update color_usage for v
        color_usage[v][color] -= 1
        if color_usage[v][color] == 0:
            del color_usage[v][color]
            distinct_color_count[v] -= 1

        # Early exit if we've found enough solutions
        if len(solutions) >= max_solutions:
            return

def visualize_and_save_graph(n, s, p, b, edge_colors, output_dir, count):
    """
    Visualize the colored graph and save it to a .png file.
    Also, save the edge-color assignments to a corresponding .txt file.

    Parameters:
    - n (int): Number of vertices in the graph.
    - s (int): Total number of colors.
    - p (int): Number of distinct colors per vertex.
    - b (int): Number of times each color appears per vertex.
    - edge_colors (dict): Edge-color assignments.
    - output_dir (str): Directory to save the outputs.
    - count (int): Coloring count/index.
    """
    G = nx.complete_graph(n)
    pos = nx.circular_layout(G)  # Circular layout without seed

    # Create a color map
    cmap = plt.get_cmap('tab20')
    edge_color_map = [cmap(edge_colors[(u, v)] % 20) for u, v in G.edges()]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color_map, width=2)

    plt.title(f'K_{n} with s={s}, p={p}, b(n)={b}, Coloring #{count}')
    plt.axis('off')

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    filename_png = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}.png'
    filepath_png = os.path.join(output_dir, filename_png)
    plt.savefig(filepath_png, format='PNG')
    plt.close()

    # Save the edge-color assignments to a .txt file
    filename_txt = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}.txt'
    filepath_txt = os.path.join(output_dir, filename_txt)
    save_edge_colors_to_txt(G, edge_colors, filepath_txt)

def save_edge_colors_to_txt(G, edge_colors, filepath):
    """
    Save the edge-color assignments to a .txt file.
    Each line contains: Edge (u, v) - Color c

    Parameters:
    - G (networkx.Graph): The graph.
    - edge_colors (dict): Edge-color assignments.
    - filepath (str): Path to the output .txt file.
    """
    with open(filepath, 'w') as file:
        file.write("Edge-Color Assignments:\n")
        file.write("=======================\n")
        for u, v in G.edges():
            color = edge_colors.get((u, v), edge_colors.get((v, u), 'Unassigned'))
            file.write(f"Edge ({u}, {v}) - Color {color}\n")

def backtrack_coloring_wrapper(edges, b, p, s, max_solutions, queue):
    """
    Wrapper function to run backtrack_coloring and put the solutions into a queue.

    Parameters:
    - edges (list of tuples): List of edges in the graph.
    - b (int): Number of times each color should appear per vertex.
    - p (int): Number of distinct colors per vertex.
    - s (int): Total number of colors available.
    - max_solutions (int): Maximum number of solutions to find.
    - queue (multiprocessing.Queue): Queue to put the solutions into.
    """
    edge_colors = {}
    color_usage = {node: defaultdict(int) for node in set(itertools.chain(*edges))}
    distinct_color_count = {node: 0 for node in set(itertools.chain(*edges))}
    solutions = []

    backtrack_coloring(edges, 0, edge_colors, color_usage, distinct_color_count, b, p, s, solutions, max_solutions)
    queue.put(solutions)

def process_t_b_pair(n, t, b, output_dir):
    """
    Process a single (t, b(n)) pair by attempting to find equitable colorings with a timeout.

    Parameters:
    - n (int): Number of vertices in the complete graph.
    - t (int): Parameter t.
    - b (int): Parameter b(n).
    - output_dir (str): Directory to save the outputs.
    """
    s = 2 * t + 1
    p = 2 * t
    print(f"  Trying t={t}, b(n)={b}: s={s}, p={p}")

    # Check if n-1 equals p * b(n)
    if (n - 1) != p * b:
        print(f"    Skipping: n-1 != p * b(n) ({n - 1} != {p} * {b})")
        return

    G = nx.complete_graph(n)
    edges = list(G.edges())

    # Initialize a multiprocessing queue
    queue = multiprocessing.Queue()

    # Create a separate process for backtracking
    process = multiprocessing.Process(target=backtrack_coloring_wrapper, args=(edges, b, p, s, MAX_SOLUTIONS_PER_CONFIGURATION, queue))
    process.start()

    # Wait for the process to finish with a timeout
    process.join(TIMEOUT_PER_CONFIGURATION)

    if process.is_alive():
        print(f"    Timeout: Could not find colorings for t={t}, b(n)={b} within 30 minutes.")
        process.terminate()
        process.join()
        return

    try:
        solutions = queue.get_nowait()
    except:
        solutions = []

    if not solutions:
        print(f"    No valid colorings found for K_{n} with t={t}, b(n)={b}")
        return

    print(f"    Found {len(solutions)} valid coloring(s) for K_{n} with t={t}, b(n)={b}")

    # Visualize and save the colorings
    for count, edge_colors_solution in enumerate(solutions, 1):
        visualize_and_save_graph(n, s, p, b, edge_colors_solution, output_dir, count)

    print(f"    Saved all colorings for K_{n} with t={t}, b(n)={b}")

# =======================
# Main Function
# =======================

def main():
    max_n = 101  # Adjust as needed to include higher n
    output_dir = OUTPUT_DIR  # Directory to save images and txt files
    max_solutions_per_configuration = MAX_SOLUTIONS_PER_CONFIGURATION  # Limit the number of solutions per configuration

    for n in range(3, max_n + 1, 2):  # n is an odd integer
        min_b = get_min_b(n, MIN_B_THRESHOLDS, DEFAULT_MIN_B)
        t_b_pairs = find_t_b_for_n(n, min_b)
        if not t_b_pairs:
            print(f"\nProcessing K_{n}: No (t, b(n)) pairs found with b(n) >= {min_b}")
            continue
        print(f"\nProcessing K_{n} with possible (t, b(n)) pairs (b(n) >= {min_b}): {t_b_pairs}")

        for t, b in t_b_pairs:
            process_t_b_pair(n, t, b, output_dir)

    print(f"\nAll possible combinations have been processed. Check the '{output_dir}' directory for images and text files.")

if __name__ == "__main__":
    main()

