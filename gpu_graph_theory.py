import itertools
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing
import time
from multiprocessing import Process, Queue
import sys
import warnings

# Increase recursion limit
sys.setrecursionlimit(1000000)

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =======================
# Adjustable Parameters
# =======================

MIN_B_THRESHOLDS = [
    (lambda x: x == 57, 3),
]
DEFAULT_MIN_B = 1
OUTPUT_DIR = 'Equitable_Complete_Graphs'  
MAX_SOLUTIONS_PER_CONFIGURATION = 1  
TIMEOUT_PER_CONFIGURATION = 10 * 60  # 10 minutes

# =======================
# Helper Functions
# =======================

def get_min_b(n, thresholds, default_min_b):
    for condition, min_b in thresholds:
        if condition(n):
            return min_b
    return default_min_b

def find_t_b_for_n(n, min_b):
    t_b_pairs = []
    for t in range(1, n):
        b = (n - 1) / (2 * t)
        if b.is_integer() and b % 2 == 1 and b >= min_b:
            b = int(b)
            t_b_pairs.append((t, b))
    return t_b_pairs

def backtrack_coloring(edges, edge_index, edge_colors, color_usage, distinct_color_count, b, p, s, solutions, max_solutions):
    if len(solutions) >= max_solutions:
        return  
    if edge_index == len(edges):
        for node in color_usage:
            if distinct_color_count[node] != p:
                return
            for color_count in color_usage[node].values():
                if color_count != b:
                    return
        solutions.append(edge_colors.copy())
        return

    u, v = edges[edge_index]
    available_colors = sorted(range(s), key=lambda c: (c not in color_usage[u], c not in color_usage[v]))

    for color in available_colors:
        can_assign = True
        if color in color_usage[u]:
            if color_usage[u][color] >= b:
                can_assign = False
        else:
            if distinct_color_count[u] >= p:
                can_assign = False
        if color in color_usage[v]:
            if color_usage[v][color] >= b:
                can_assign = False
        else:
            if distinct_color_count[v] >= p:
                can_assign = False

        if not can_assign:
            continue  

        edge_colors[(u, v)] = color
        if color not in color_usage[u]:
            distinct_color_count[u] += 1
        color_usage[u][color] += 1
        if color not in color_usage[v]:
            distinct_color_count[v] += 1
        color_usage[v][color] += 1

        backtrack_coloring(edges, edge_index + 1, edge_colors, color_usage, distinct_color_count, b, p, s, solutions, max_solutions)

        del edge_colors[(u, v)]
        color_usage[u][color] -= 1
        if color_usage[u][color] == 0:
            del color_usage[u][color]
            distinct_color_count[u] -= 1
        color_usage[v][color] -= 1
        if color_usage[v][color] == 0:
            del color_usage[v][color]
            distinct_color_count[v] -= 1

        if len(solutions) >= max_solutions:
            return

def backtrack_coloring_wrapper(edges, b, p, s, max_solutions, queue):
    edge_colors = {}
    color_usage = {node: defaultdict(int) for node in set(itertools.chain(*edges))}
    distinct_color_count = {node: 0 for node in set(itertools.chain(*edges))}
    solutions = []

    try:
        backtrack_coloring(edges, 0, edge_colors, color_usage, distinct_color_count, b, p, s, solutions, max_solutions)
        queue.put(solutions)
    except RecursionError:
        queue.put(None)

def visualize_and_save_graph(n, s, p, b, edge_colors, output_dir, count):
    G = nx.complete_graph(n)
    pos = nx.circular_layout(G)  

    cmap = plt.get_cmap('tab20')
    edge_color_map = [cmap(edge_colors[(u, v)] % 20) for u, v in G.edges()]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color_map, width=2)

    plt.title(f'K_{n} with s={s}, p={p}, b(n)={b}, Coloring #{count}')
    plt.axis('off')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename_png = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}.png'
    filepath_png = os.path.join(output_dir, filename_png)
    plt.savefig(filepath_png, format='PNG')
    plt.close()

def process_t_b_pair(n, t, b, output_dir):
    s = 2 * t + 1
    p = 2 * t
    print(f"  Trying t={t}, b(n)={b}: s={s}, p={p}")

    if (n - 1) != p * b:
        print(f"    Skipping: n-1 != p * b(n) ({n - 1} != {p} * {b})")
        return

    G = nx.complete_graph(n)
    edges = list(G.edges())

    queue = Queue()
    proc = Process(target=backtrack_coloring_wrapper, args=(edges, b, p, s, MAX_SOLUTIONS_PER_CONFIGURATION, queue))
    proc.start()
    proc.join(timeout=TIMEOUT_PER_CONFIGURATION)

    if proc.is_alive():
        print(f"    Timeout: Could not find colorings for t={t}, b(n)={b} within 10 minutes.")
        proc.terminate()
        proc.join()
        return

    try:
        solutions = queue.get_nowait()
    except Exception:
        solutions = None

    if not solutions:
        print(f"    No valid colorings found for K_{n} with t={t}, b(n)={b}")
        return
    else:
        print(f"    Found {len(solutions)} valid coloring(s) for K_{n} with t={t}, b(n)={b}")
        for count, edge_colors_solution in enumerate(solutions, 1):
            visualize_and_save_graph(n, s, p, b, edge_colors_solution, output_dir, count)
        print(f"    Saved all colorings for K_{n} with t={t}, b(n)={b}")

def main():
    max_n = 101  
    output_dir = OUTPUT_DIR  

    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPU cores.")

    processes = []
    semaphore = multiprocessing.Semaphore(num_cpus)

    for n in range(3, max_n + 1, 2):  
        min_b = get_min_b(n, MIN_B_THRESHOLDS, DEFAULT_MIN_B)
        t_b_pairs = find_t_b_for_n(n, min_b)
        if not t_b_pairs:
            print(f"\nProcessing K_{n}: No (t, b(n)) pairs found with b(n) >= {min_b}")
            continue
        print(f"\nProcessing K_{n} with possible (t, b(n)) pairs (b(n) >= {min_b}): {t_b_pairs}")

        for t, b in t_b_pairs:
            semaphore.acquire()
            p = Process(target=process_t_b_pair_wrapper, args=(n, t, b, output_dir, semaphore))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    print(f"\nAll possible combinations have been processed. Check the '{output_dir}' directory for images and text files.")

def process_t_b_pair_wrapper(n, t, b, output_dir, semaphore):
    try:
        process_t_b_pair(n, t, b, output_dir)
    finally:
        semaphore.release()

if __name__ == "__main__":
    main()

