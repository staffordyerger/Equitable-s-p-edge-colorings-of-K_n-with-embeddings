import os
import re

# Define the directory path
directory_path = "/home/microct/graph_theory/Smaller_Embedded_Complete_Graphs"

# Initialize lists to store data
ratios = []
b_pairs = []
filtered_ratios = []
filtered_ratios_gt1 = []
filtered_ratios_gt3 = []
filtered_ratios_gt5 = []
filtered_ratios_gt7 = []

# Lists to store filenames that violate each inequality
violations_final_version = []
violations_original_version = []
violations_new_proof_attempt = []

# Regular expression pattern to extract all necessary numbers
# Captures n1, s1, p1, b1(n), n2, s2, p2, b2(n)
pattern = re.compile(
    r"K_(\d+)_s_(\d+)_p_(\d+)_b_(\d+)_c_\d+\.txt_embedded_K_(\d+)_s_(\d+)_p_(\d+)_b_(\d+)_c_\d+"
)

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    match = pattern.match(filename)
    if match:
        try:
            # Extract numbers from the filename
            n1 = int(match.group(1))
            s1 = int(match.group(2))
            p1 = int(match.group(3))
            b1_n = int(match.group(4))
            n2 = int(match.group(5))
            s2 = int(match.group(6))
            p2 = int(match.group(7))
            b2_n = int(match.group(8))

            # Compute t1 and t2
            t1 = p1 / 2
            t2 = p2 / 2

            # Avoid division by zero
            if p2 == 0:
                print(f"Skipping file '{filename}' due to p2=0 to avoid division by zero.")
                continue

            # Calculate the ratio
            ratio = p1 / p2
            ratios.append(ratio)
            b_pairs.append((b1_n, b2_n))

            # Check if b1 equals b2 and store the ratio if they do
            if b1_n == b2_n:
                filtered_ratios.append(ratio)

                # Further filter based on b1 values
                if b1_n > 1:
                    filtered_ratios_gt1.append(ratio)
                if b1_n > 3:
                    filtered_ratios_gt3.append(ratio)
                if b1_n > 5:
                    filtered_ratios_gt5.append(ratio)
                if b1_n > 7:
                    filtered_ratios_gt7.append(ratio)

            # Compute inequalities

            # Inequality 1: final_version_of_original_proof_inequality
            lhs1 = (n1 - n2) * (s1 - s2) - 1
            rhs1 = n2 * (s1 - s2 - 1)
            inequality1_holds = lhs1 >= rhs1

            if not inequality1_holds:
                violations_final_version.append(filename)

            # Inequality 2: original_version_of_proof_inequality
            lhs2 = (n1 - n2) * (s1 - s2) * b1_n
            rhs2 = n2 * (s1 - s2 - 1) * b2_n
            inequality2_holds = lhs2 >= rhs2

            if not inequality2_holds:
                violations_original_version.append(filename)

            # Inequality 3: new_proof_attempt_inequality
            lhs3 = p1
            rhs3 = 2 * p2
            inequality3_holds = lhs3 >= rhs3

            if not inequality3_holds and b1_n == b2_n:
                violations_new_proof_attempt.append(filename)

            # Print the extracted and computed values
            print(f"File: {filename}")
            print(f"  n1: {n1}, s1: {s1}, p1: {p1}, b1(n): {b1_n}")
            print(f"  n2: {n2}, s2: {s2}, p2: {p2}, b2(n): {b2_n}")
            print(f"  t1: {t1}, t2: {t2}")
            print(f"  p1/p2: {ratio:.3f}")
            print(f"  Inequality 1 (Final Version): {'Holds' if inequality1_holds else 'Does NOT Hold'}")
            print(f"  Inequality 2 (Original Version): {'Holds' if inequality2_holds else 'Does NOT Hold'}")
            print(f"  Inequality 3 (New Proof Attempt): {'Holds' if inequality3_holds else 'Does NOT Hold'}\n")

        except ValueError as ve:
            print(f"Error processing file '{filename}': {ve}")
    else:
        print(f"Filename '{filename}' does not match the expected pattern.")

# After processing all files, compute max and min ratios
if ratios:
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    print(f"\nMaximum p ratio: {max_ratio:.3f}")
    print(f"Minimum p ratio: {min_ratio:.3f}\n")
else:
    print("No valid ratios were computed.\n")

# Define output file paths for filtered ratios
output_files = {
    "filtered_p_ratios.txt": filtered_ratios,
    "filtered_p_ratios_gt1.txt": filtered_ratios_gt1,
    "filtered_p_ratios_gt3.txt": filtered_ratios_gt3,
    "filtered_p_ratios_gt5.txt": filtered_ratios_gt5,
    "filtered_p_ratios_gt7.txt": filtered_ratios_gt7,
}

# Write the filtered ratios to their respective text files
for out_filename, data in output_files.items():
    output_file_path = os.path.join(directory_path, out_filename)
    with open(output_file_path, 'w') as f:
        for ratio in data:
            f.write(f"{ratio:.3f}\n")
    print(f"Filtered ratios have been written to '{output_file_path}'.")

# Print out violations for each inequality
def print_violations(inequality_name, violations):
    if violations:
        print(f"\nFiles that DO NOT satisfy the {inequality_name}:")
        for vf in violations:
            print(f"  - {vf}")
    else:
        print(f"\nAll files satisfy the {inequality_name}.")

print_violations("Final Version of Original Proof Inequality", violations_final_version)
print_violations("Original Version of Proof Inequality", violations_original_version)
print_violations("New Proof Attempt Inequality", violations_new_proof_attempt)

