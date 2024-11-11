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
import numpy as np
from numba import cuda, njit, prange, int32
import math

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
           39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 
           73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]  # Adjust as needed

# Maximum number of embedded graphs to find per (t, b) pair
MAX_EMBEDDINGS_PER_TB = 1  # Adjust as needed

# Time limit in seconds per (t, b(n)) pair
TIME_LIMIT_PER_TB_PAIR = 30  # 30 seconds

# Number of CPU worker processes for parallelization
NUM_CPU_WORKERS = 2  # Set to the number of available GPUs

# Number of GPUs available
NUM_GPUS = 2  # Adjust based on available GPUs

# Size of batch of subsets to process in each worker
SUBSET_BATCH_SIZE = 1000  # Adjust based on memory and performance

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

def create_edge_color_matrix(n, edge_colors):
    """
    Create an adjacency matrix representation of the edge colors.
    """
    matrix = np.full((n, n), -1, dtype=np.int32)
    for (u, v), color in edge_colors.items():
        matrix[u, v] = color
        matrix[v, u] = color  # Since the graph is undirected
    return matrix

@cuda.jit
def check_equitable_coloring_gpu(edge_indices, edge_colors_matrix, s_sub, p_sub, b_sub, result_array):
    """
    GPU-accelerated function to check equitable coloring for a batch of subsets.
    """
    idx = cuda.grid(1)
    if idx >= edge_indices.shape[0]:
        return

    subset = edge_indices[idx]
    n = subset.shape[0]

    # Assuming maximum 100 colors; adjust as needed
    MAX_COLORS = 100

    # Initialize color usage arrays
    color_usage = cuda.local.array((100, 100), dtype=int32)  # [node][color]
    color_count = cuda.local.array(100, dtype=int32)
    distinct_color_count = cuda.local.array(100, dtype=int32)

    for i in range(n):
        color_count[i] = 0
        distinct_color_count[i] = 0
        for j in range(MAX_COLORS):
            color_usage[i][j] = 0

    # Initialize total colors
    total_colors = cuda.local.array(1000, dtype=int32)  # Adjust as needed
    total_color_count = 0

    valid = 1  # Assume valid unless proven otherwise

    for i in range(n):
        node_u = subset[i]
        for j in range(i + 1, n):
            node_v = subset[j]
            color = edge_colors_matrix[node_u, node_v]
            if color == -1:
                valid = 0
                break
            # Update total colors
            found = False
            for k in range(total_color_count):
                if total_colors[k] == color:
                    found = True
                    break
            if not found:
                if total_color_count < 1000:
                    total_colors[total_color_count] = color
                    total_color_count += 1
                else:
                    valid = 0
                    break

            # Update color usage for node_u
            if color_usage[i][color] == 0:
                distinct_color_count[i] += 1
            color_usage[i][color] += 1

            # Update color usage for node_v
            if color_usage[j][color] == 0:
                distinct_color_count[j] += 1
            color_usage[j][color] += 1

        if valid == 0:
            break

    if valid == 0:
        result_array[idx] = 0
        return

    # Check total number of distinct colors
    if total_color_count != s_sub:
        result_array[idx] = 0
        return

    # Check color counts per node
    for i in range(n):
        if distinct_color_count[i] != p_sub:
            valid = 0
            break
        for j in range(total_color_count):
            color = total_colors[j]
            if color_usage[i][color] != b_sub:
                valid = 0
                break
        if valid == 0:
            break

    result_array[idx] = valid

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
    Process a single (filepath, m, t, b) pair using CPU and GPU to find embedded equitable (s, p) edge colorings.
    """
    # Unpack arguments
    filepath, filename, n, s, p, b, m, t_b_pair, count, gpu_id = args
    t_sub, b_sub = t_b_pair

    try:
        # Assign the GPU device
        cuda.select_device(gpu_id)

        print(f"\nProcessing {filename} for m={m} with (t={t_sub}, b(n)={b_sub}) on CPU using GPU {gpu_id}")

        # Read the edge-color assignments
        edge_colors = read_edge_colors_from_txt(filepath)

        if not edge_colors:
            print(f"    No edge colors found in {filename}. Skipping.")
            # Even if no edge colors, it's considered a failure with no embedding
            original_title = f'K_{n}_s_{s}_p_{p}_b_{b}_c_{count}'
            attempted_embedding_title = f'K_{m}_s_{2 * t_sub +1}_p_{2 * t_sub}_b_{b_sub}'
            return ([], [(original_title, attempted_embedding_title, 'no embedding existed')])

        # Reconstruct the original graph
        G = nx.complete_graph(n)

        # Create edge color matrix
        edge_colors_matrix = create_edge_color_matrix(n, edge_colors)
        edge_colors_matrix_gpu = cuda.to_device(edge_colors_matrix)

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
        vertex_subsets = list(itertools.combinations(range(n), m))
        total_subsets = len(vertex_subsets)

        batch_size = SUBSET_BATCH_SIZE
        num_batches = math.ceil(total_subsets / batch_size)

        for batch_idx in range(num_batches):
            # Check if time limit exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT_PER_TB_PAIR:
                print(f"      Time limit exceeded ({TIME_LIMIT_PER_TB_PAIR} seconds) for (t={t_sub}, b(n)={b_sub}). Moving to next.")
                timed_out = True
                break

            batch_subsets = vertex_subsets[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if len(batch_subsets) == 0:
                continue

            subsets_array = np.array(batch_subsets, dtype=np.int32)
            subsets_gpu = cuda.to_device(subsets_array)

            result_array = np.zeros(len(batch_subsets), dtype=np.int32)
            result_gpu = cuda.to_device(result_array)

            threads_per_block = 256
            blocks_per_grid = (len(batch_subsets) + (threads_per_block - 1)) // threads_per_block

            try:
                # Launch kernel
                check_equitable_coloring_gpu[blocks_per_grid, threads_per_block](
                    subsets_gpu, edge_colors_matrix_gpu, s_sub, p_sub, b_sub, result_gpu
                )
            except Exception as e:
                print(f"    Error during CUDA kernel launch: {e}")
                return (successes, failures)

            result_array = result_gpu.copy_to_host()

            for idx, is_valid in enumerate(result_array):
                if is_valid:
                    subset = batch_subsets[idx]
                    nodes = subset
                    edges = []
                    edge_colors_sub = {}
                    for i in range(len(subset)):
                        u = subset[i]
                        for j in range(i + 1, len(subset)):
                            v = subset[j]
                            color = edge_colors.get((u, v), edge_colors.get((v, u)))
                            edges.append((u, v))
                            edge_colors_sub[(u, v)] = color

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
                    G_sub = G.subgraph(nodes).copy()
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

            for idx, tb in enumerate(t_b_pairs):
                # Assign a GPU ID in a round-robin fashion
                gpu_id = idx % NUM_GPUS
                # Each task is a tuple of arguments to be passed to process_t_b_pair
                task_args = (filepath, filename, n, s, p, b, m, tb, count, gpu_id)
                tasks.append(task_args)

    print(f"Total tasks to process: {len(tasks)}")

    # Use ProcessPoolExecutor to handle CPU tasks with GPU assistance
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
        # Submit all tasks to the executor
        future_to_task = {executor.submit(process_t_b_pair, task): task for task in tasks}

        all_successes = []
        all_failures = []

        for future in concurrent.futures.as_completed(future_to_task, timeout=None):
            task = future_to_task[future]
            try:
                successes, failures = future.result(timeout=TIME_LIMIT_PER_TB_PAIR)
                all_successes.extend(successes)
                all_failures.extend(failures)
            except concurrent.futures.TimeoutError:
                print(f"    Task {task} exceeded the time limit and was cancelled.")
                original_title = f'K_{task[2]}_s_{task[3]}_p_{task[4]}_b_{task[5]}_c_{task[8]}'
                attempted_embedding_title = f'K_{task[6]}_s_{2 * task[7][0] +1}_p_{2 * task[7][0]}_b_{task[7][1]}'
                all_failures.append((original_title, attempted_embedding_title, 'timed out'))
                continue
            except Exception as e:
                print(f"    Task {task} generated an exception: {e}")
                original_title = f'K_{task[2]}_s_{task[3]}_p_{task[4]}_b_{task[5]}_c_{task[8]}'
                attempted_embedding_title = f'K_{task[6]}_s_{2 * task[7][0] +1}_p_{2 * task[7][0]}_b_{task[7][1]}'
                all_failures.append((original_title, attempted_embedding_title, f'exception: {e}'))
                continue

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

