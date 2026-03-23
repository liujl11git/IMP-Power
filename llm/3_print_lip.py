import argparse
import os
import pandas as pd
import textdistance


default_results_csv = "semantic_contrast_pairs_200_results_all.csv"
fallback_results_csv = "semantic_contrast_pairs_200_results_steps_2_4_6_8_10_12_14_16_split0.csv"

parser = argparse.ArgumentParser(
    description="Print mean Lipschitz estimates for semantic contrast pairs."
)
parser.add_argument(
    "--inputs_csv",
    type=str,
    default="semantic_contrast_pairs_200.csv",
    help="CSV with columns: index, input 1, input 2",
)
parser.add_argument(
    "--results_csv",
    type=str,
    default=None,
    help="CSV with columns: index, output1_k, output2_k, distance_k (for multiple k)",
)
args = parser.parse_args()

if args.results_csv is None:
    if os.path.exists(default_results_csv):
        results_csv = default_results_csv
    else:
        results_csv = fallback_results_csv
else:
    results_csv = args.results_csv
print(f"Test results loaded from {results_csv}.")

# Load input and result files
inputs_df = pd.read_csv(args.inputs_csv)
results_df = pd.read_csv(results_csv)

# Merge on index
merged = pd.merge(inputs_df, results_df, on="index", how="inner")

# Detect all step values from distance_* columns
step_vals = sorted(
    int(col.split("_")[1])
    for col in merged.columns
    if col.startswith("distance_")
)
if not step_vals:
    raise ValueError("No distance_* columns found in results CSV.")

def lev_tokens(s1: str, s2: str) -> int:
    tokens1 = str(s1).split()
    tokens2 = str(s2).split()
    return textdistance.levenshtein.distance(tokens1, tokens2)

lipschitz_sums = {k: 0.0 for k in step_vals}
lipschitz_counts = {k: 0 for k in step_vals}

for _, row in merged.iterrows():
    inp1 = row["input 1"]
    inp2 = row["input 2"]

    input_diff = lev_tokens(inp1, inp2)
    if input_diff <= 0:
        continue

    for k in step_vals:
        d_k = row[f"distance_{k}"]
        try:
            distance_k = float(d_k)
        except Exception:
            continue

        lipschitz_k = distance_k / input_diff
        lipschitz_sums[k] += lipschitz_k
        lipschitz_counts[k] += 1

print("Mean Lipschitz by step:")
for k in step_vals:
    count = lipschitz_counts[k]
    mean_lipschitz = lipschitz_sums[k] / count if count > 0 else float("nan")
    print(f"step {k}: {mean_lipschitz:.6f} (mean across {count} samples)")

