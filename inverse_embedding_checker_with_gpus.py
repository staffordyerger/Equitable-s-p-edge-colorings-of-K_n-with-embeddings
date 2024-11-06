import os
import networkx as nx
import matplotlib
# Use the 'Agg' backend to prevent tkinter-related errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import re
import concurrent.futures
import time
from functools import partial

# =======================
# Adjustable Parameters
# =======================

# Directory containing the original .txt files with edge-color assignments
INPUT_DIR = 'Equitable_Complete_Graphs'

# Directory to save the outputs (embedded graphs and their colorings)
OUTPUT_DIR = 'Smaller_Embedded_Complete_Graphs'

# Maximum number of subsets to check per (n, m) combination
MAX_SUBSETS_PER_COMBINATION = 9000000000000000  # Adjust as needed

# Range of m values to consider (must be odd integers less than n)
M_VALUES = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 
39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 
79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]  # Adjust as needed

# Maximum number of embedded graphs to find per (t, b) pair
MAX_EMBEDDINGS_PER_TB = 1  # Adjust as needed

# Time limit in seconds per (t, b(n)) pair
TIME_LIMIT_PER_TB_PAIR = 30  # 30 seconds

# Number of worker processes for parallelization (to be set dynamically)
NUM_WORKERS = os.cpu_count() or 30  # Fallback to 30 if os.cpu_count() is None

# Size of batch of subsets to process in each worker (not used in refactored version)
SUBSET_BATCH_SIZE = 100000  # Adjust based on memory and performance

# =======================
# Helper Functions
# =======================

def find_t_b_for_n(m):
    """
    For a given m, find all possible (t, b(n)) pairs such that m = 2t * b(n) + 1,
    where t >= 1 and b(n) is an odd positive integer.
    """
    t_b_pairs = []
    for t in range(1, m):
        # Calculate b(n)
        b = (m - 1) / (2 * t)
        if b.is_integer() and b % 2 == 1 and b >= 1:
            b = int(b)
            t_b_pairs.append((t, b))
    return t_b_pairs

def read_edge_colors_from_txt(filepath):
    """
    Read edge-color assignments from a .txt file and reconstruct the edge_colors dictionary.
    """
    edge_colors = {}
    # Regular expression pattern to match lines like: "Edge (u, v) - Color c"
    pattern = re.compile(r'Edge \((\d+), (\d+)\) - Color (\d+)')
    try:
        with open(filepath, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                match = pattern.match(line)
                if match:
                    u = int(match.group(1))
                    v = int(match.group(2))
                    color = int(match.group(3))
                    edge_colors[(u, v)] = color
                else:
                    # Optionally, log skipped lines
                    if line and not line.startswith("Edge-Color Assignments"):
                        print(f"      Skipping line {line_num}: {line}")
                    continue  # Skip lines that do not match the pattern
    except Exception as e:
        print(f"    Error reading {filepath}: {e}")
    return edge_colors

def check_equitable_coloring(G_sub_edges, edge_colors_sub, s, p, b, nodes):
    """
    Check if the subgraph defined by G_sub_edges with edge_colors_sub forms an equitable (s, p) edge coloring.
    """
    color_usage = {node: defaultdict(int) for node in nodes}
    distinct_color_count = {node: 0 for node in nodes}

    for (u, v), color in edge_colors_sub.items():
        # Update color_usage for u
        if color not in color_usage[u]:
            distinct_color_count[u] += 1
        color_usage[u][color] += 1
        # Update color_usage for v
        if color not in color_usage[v]:
            distinct_color_count[v] += 1
        color_usage[v][color] += 1

    # Check if each vertex has exactly p distinct colors, each appearing b times
    for node in nodes:
        if distinct_color_count[node] != p:
            return False
        for color_count in color_usage[node].values():
            if color_count != b:
                return False
    return True

def visualize_and_save_graph(G, edge_colors, title, filepath_png):
    """
    Visualize the colored graph and save it to a .png file.
    """
    pos = nx.circular_layout(G)
    # Create a color map
    cmap = plt.get_cmap('tab20')
    edge_color_map = [cmap(edge_colors.get((u, v), edge_colors.get((v, u))) % 20) for u, v in G.edges()]
    
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color_map, width=2)

    plt.title(title)
    plt.axis('off')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath_png), exist_ok=True)

    # Save the figure
    plt.savefig(filepath_png, format='PNG')
    plt.close()
    print(f"      Saved graph as {filepath_png}")

def save_edge_colors_to_txt(G, edge_colors, filepath):
    """
    Save the edge-color assignments to a .txt file.
    Each line contains: Edge (u, v) - Color c
    """
    try:
        with open(filepath, 'w') as file:
            file.write("Edge-Color Assignments:\n")
            file.write("=======================\n")
            for u, v in G.edges():
                color = edge_colors.get((u, v), edge_colors.get((v, u), 'Unassigned'))
                file.write(f"Edge ({u}, {v}) - Color {color}\n")
        print(f"      Saved edge colors to {filepath}")
    except Exception as e:
        print(f"    Error writing to {filepath}: {e}")

def visualize_and_save_combined_graphs(G_original, edge_colors_original, G_embedded, edge_colors_embedded, title_original, title_embedded, filepath_png):
    """
    Visualize both the original and embedded graphs side by side and save to a single .png file.
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original Graph
    pos_original = nx.circular_layout(G_original)
    cmap = plt.get_cmap('tab20')
    edge_color_map_original = [cmap(edge_colors_original.get((u, v), edge_colors_original.get((v, u))) % 20) for u, v in G_original.edges()]
    nx.draw_networkx_nodes(G_original, pos_original, node_size=300, node_color='lightblue', ax=axes[0])
    nx.draw_networkx_labels(G_original, pos_original, font_size=10, ax=axes[0])
    nx.draw_networkx_edges(G_original, pos_original, edge_color=edge_color_map_original, width=2, ax=axes[0])
    axes[0].set_title(title_original, fontsize=14)
    axes[0].axis('off')

    # Embedded Graph
    pos_embedded = nx.circular_layout(G_embedded)
    edge_color_map_embedded = [cmap(edge_colors_embedded.get((u, v), edge_colors_embedded.get((v, u))) % 20) for u, v in G_embedded.edges()]
    nx.draw_networkx_nodes(G_embedded, pos_embedded, node_size=300, node_color='lightgreen', ax=axes[1])
    nx.draw_networkx_labels(G_embedded, pos_embedded, font_size=10, ax=axes[1])
    nx.draw_networkx_edges(G_embedded, pos_embedded, edge_color=edge_color_map_embedded, width=2, ax=axes[1])
    axes[1].set_title(title_embedded, fontsize=14)
    axes[1].axis('off')

    # Overall Title
    plt.suptitle('Original and Embedded Graphs', fontsize=20)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath_png, format='PNG')
    plt.close()
    print(f"      Saved combined graph as {filepath_png}")

def process_t_b_pair(args):
    """
    Process a single (filepath, m, t, b) pair to find embedded equitable (s, p) edge colorings.
    This function is designed to be called in parallel.
    Returns:
        successes: List of tuples (original_title, embedded_title)
        failures: List of tuples (original_title, attempted_embedding_title, reason)
    """
    filepath, filename, n, s, p, b, m, t_b_pair, count = args
    t_sub, b_sub = t_b_pair

    print(f"\nProcessing {filename} for m={m} with (t={t_sub}, b(n)={b_sub})")

    # Read the edge-color assignments
    edge_colors = read_edge_colors_from_txt(filepath)

    if not edge_colors:
        print(f"    No edge colors found in {filename}. Skipping.")
        # Even if no edge colors, it's considered a failure with no embedding
        original_title = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}'
        attempted_embedding_title = f'K_{m}_s_{2 * t_sub}_p_{2 * t_sub}_b_{b_sub}'
        return ([], [(original_title, attempted_embedding_title, 'no embedding existed')])

    # Reconstruct the original graph
    G = nx.complete_graph(n)

    # Calculate s and p for equitable coloring
    s_sub = 2 * t_sub + 1
    p_sub = 2 * t_sub

    embedding_count = 0  # Initialize embedding count for this (t, b(n)) pair
    successes = []
    failures = []

    # Start timer
    start_time = time.time()
    timed_out = False

    # Generate combinations of m vertices
    vertex_subsets = itertools.combinations(G.nodes(), m)

    # To keep track of embeddings found
    embeddings_found = []

    try:
        for subset in vertex_subsets:
            # Check if time limit exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT_PER_TB_PAIR:
                print(f"      Time limit exceeded ({TIME_LIMIT_PER_TB_PAIR} seconds) for (t={t_sub}, b(n)={b_sub}). Moving to next.")
                timed_out = True
                break

            nodes = subset
            edges = []
            edge_colors_sub = {}
            missing_edge = False
            for u, v in itertools.combinations(subset, 2):
                if (u, v) in edge_colors:
                    edges.append((u, v))
                    edge_colors_sub[(u, v)] = edge_colors[(u, v)]
                elif (v, u) in edge_colors:
                    edges.append((u, v))
                    edge_colors_sub[(u, v)] = edge_colors[(v, u)]
                else:
                    missing_edge = True
                    break
            if missing_edge:
                continue
            if check_equitable_coloring(edges, edge_colors_sub, s_sub, p_sub, b_sub, nodes):
                embedding_count += 1
                print(f"      Found embedded K_{m} with t={t_sub}, b(n)={b_sub} (Count: {embedding_count})")

                # Record the characteristics
                original_title = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}'
                embedded_title = f'K_{m}_s_{s_sub}_p_{p_sub}_b_{b_sub}_c_{embedding_count}'
                successes.append((original_title, embedded_title))

                # Save the original and embedded graphs
                output_subdir = os.path.join(
                    OUTPUT_DIR,
                    f"{filename}_embedded_K_{m}_s_{s_sub}_p_{p_sub}_b_{b_sub}_c_{embedding_count}"
                )
                os.makedirs(output_subdir, exist_ok=True)

                # Original graph
                original_png = os.path.join(output_subdir, f'original_{filename}.png')
                visualize_and_save_graph(G, edge_colors, original_title, original_png)
                original_txt = os.path.join(output_subdir, f'original_{filename}.txt')
                save_edge_colors_to_txt(G, edge_colors, original_txt)

                # Embedded graph
                G_sub = G.subgraph(subset).copy()
                embedded_png = os.path.join(output_subdir, f'embedded_K_{m}_s_{s_sub}_p_{p_sub}_b_{b_sub}.png')
                visualize_and_save_graph(G_sub, edge_colors_sub, embedded_title, embedded_png)
                embedded_txt = os.path.join(output_subdir, f'embedded_K_{m}_s_{s_sub}_p_{p_sub}_b_{b_sub}.txt')
                save_edge_colors_to_txt(G_sub, edge_colors_sub, embedded_txt)

                # Combined graph visualization
                combined_title_original = original_title
                combined_title_embedded = embedded_title
                combined_png = os.path.join(
                    output_subdir,
                    f'combined_original_embedded_K_{m}_t_{t_sub}_b_{b_sub}_c_{embedding_count}.png'
                )
                visualize_and_save_combined_graphs(
                    G,
                    edge_colors,
                    G_sub,
                    edge_colors_sub,
                    combined_title_original,
                    combined_title_embedded,
                    combined_png
                )

                # Record the characteristics
                info_filepath = os.path.join(output_subdir, 'embedded_graph_info.txt')
                try:
                    with open(info_filepath, 'w') as info_file:
                        info_file.write(f"Original Graph: {original_title}\n")
                        info_file.write(f"Embedded Graph: {embedded_title}\n")
                    print(f"      Recorded embedded graph characteristics in {info_filepath}")
                except Exception as e:
                    print(f"    Error writing info file: {e}")

                if embedding_count >= MAX_EMBEDDINGS_PER_TB:
                    print(f"      Reached maximum embeddings ({MAX_EMBEDDINGS_PER_TB}) for (t={t_sub}, b(n)={b_sub}).")
                    break

    except Exception as e:
        print(f"      Exception during processing subsets: {e}")

    if embedding_count == 0:
        original_title = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}'
        attempted_embedding_title = f'K_{m}_s_{s_sub}_p_{p_sub}_b_{b_sub}'
        reason = 'timed out' if timed_out else 'no embedding existed'
        failures.append((original_title, attempted_embedding_title, reason))

    print(f"    Completed processing for (t={t_sub}, b(n)={b_sub}) with {embedding_count} embeddings found.")

    return (successes, failures)

def main():
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List all .txt files in the input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory '{INPUT_DIR}' does not exist. Exiting.")
        return

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in '{INPUT_DIR}'. Exiting.")
        return

    print(f"Found {len(txt_files)} .txt files in '{INPUT_DIR}'.")

    # Prepare a list of tasks
    tasks = []

    for txt_file in txt_files:
        filepath = os.path.join(INPUT_DIR, txt_file)
        filename = os.path.basename(filepath)

        # Extract n, s, p, b from the filename
        parts = filename.split('_')
        try:
            n = int(parts[1])
            s = int(parts[3])
            p = int(parts[5])
            b = int(parts[7])
            count_part = parts[9]
            count = re.findall(r'\d+', count_part)[0] if re.findall(r'\d+', count_part) else '1'
        except (IndexError, ValueError) as e:
            print(f"    Error parsing filename {filename}: {e}")
            continue

        # Iterate through each m
        for m in M_VALUES:
            if m >= n or m < 3 or m % 2 == 0:
                print(f"  Skipping m={m} as it is invalid (must be odd and less than n).")
                continue  # Skip invalid m values

            t_b_pairs = find_t_b_for_n(m)
            if not t_b_pairs:
                print(f"  No (t, b(n)) pairs found for K_{m}. Skipping m={m}.")
                continue

            print(f"  Preparing tasks for embedded K_{m} graphs with possible (t, b(n)) pairs: {t_b_pairs}")

            for tb in t_b_pairs:
                # Each task is a tuple of arguments to be passed to process_t_b_pair
                task_args = (filepath, filename, n, s, p, b, m, tb, count)
                tasks.append(task_args)

    print(f"Total tasks to process: {len(tasks)}")

    # Use ProcessPoolExecutor to handle multiple (file, m, t, b) pairs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks to the executor and collect results
        results = executor.map(process_t_b_pair, tasks)

    # Collect all successes and failures
    all_successes = []
    all_failures = []
    for res in results:
        successes, failures = res
        all_successes.extend(successes)
        all_failures.extend(failures)

    # Write the overview file
    overview_path = os.path.join(OUTPUT_DIR, 'overview.txt')
    try:
        with open(overview_path, 'w') as overview_file:
            overview_file.write("Successes:\n")
            overview_file.write("=======================\n")
            for orig, embed in all_successes:
                overview_file.write(f"{orig} -> {embed}\n")
            
            overview_file.write("\nFailures:\n")
            overview_file.write("=======================\n")
            for orig, embed, reason in all_failures:
                overview_file.write(f"{orig} -> {embed} | Reason: {reason}\n")
        print(f"\nOverview file created at {overview_path}")
    except Exception as e:
        print(f"Error writing overview file: {e}")

    print(f"\nAll colorings have been processed. Check the '{OUTPUT_DIR}' directory for results.")

if __name__ == "__main__":
    main()

