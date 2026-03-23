import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import argparse
import pandas as pd
import os
import textdistance

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_steps", type=str, default='2,4,6,8,10,12,14,16')
parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits/workers.")
parser.add_argument("--split", type=int, default=0, help="Split index for this worker in [0, num_splits-1].")
parser.add_argument("--merge", action="store_true", help="Merge per-split result files and exit.")
parser.add_argument(
    "sentence",
    nargs="+",
    help="Input CSV file path (previously: input sentence).",
)
args = parser.parse_args()


def run_single(device, model, num_steps, prompt):
    """
    Run a single generation with the given device, model, num_steps and prompt.
    Returns the decoded text.
    """

    # Generation config
    config = GenerationConfig(
        max_length=128,
        stop_strings=["<|end_text|>", "<|end_turn|>"],
        use_cache=True,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505,
        bos_token_id=65504,
        pad_token_id=65509,
    )

    # Tokenize the prompt
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    # Generate (Huginn uses this custom signature with config as 2nd arg)
    outputs = model.generate(
        input_ids,
        config,
        tokenizer=tokenizer,
        num_steps=num_steps,
    )

    # Decode full sequence (prompt + completion)
    generated_ids = outputs.sequences[0]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text


def build_split_output_path(csv_path, step_list, split):
    out_base, _ = os.path.splitext(csv_path)
    return f"{out_base}_results_steps_{'_'.join(map(str, step_list))}_split{split}.csv"


def merge_split_outputs(csv_path, step_list, num_splits):
    dfs = []
    for split in range(num_splits):
        split_path = build_split_output_path(csv_path, step_list, split)
        dfs.append(pd.read_csv(split_path))

    merged_df = pd.concat(dfs, ignore_index=True).sort_values("index")
    out_base, _ = os.path.splitext(csv_path)
    merged_path = f"{out_base}_results_all.csv"
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged {num_splits} split files into: {merged_path}")


# CSV path from positional argument
if len(args.sentence) < 1:
    raise ValueError("Please provide a CSV file path.")
csv_path = args.sentence[0]

step_list = [int(ss) for ss in args.num_steps.split(',')]

if args.merge:
    merge_split_outputs(csv_path, step_list, args.num_splits)
    raise SystemExit(0)

# Select device
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    print("CUDA not available, using CPU.")
    device = torch.device("cpu")

# Load model and tokenizer
model_name = "tomg-group-umd/huginn-0125"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

# Read CSV: expect columns "index", "input 1", "input 2"
df = pd.read_csv(csv_path)

# Inference for each row
results = []
num_splits = args.num_splits
split = args.split
for _, row in df.iterrows():
    idx = row["index"]  # assume 1-based indexing in CSV

    # Assign rows to this split: (idx - 1) % num_splits == split
    if (idx - 1) % num_splits != split:
        continue

    prompt1 = row["input 1"]
    prompt2 = row["input 2"]

    record = {"index": idx}

    for steps in step_list:
        print("\n" + "=" * 40)
        print(f"[split {split}] index = {idx}, num_steps = {steps}")
        print("=" * 40)

        out_text1 = run_single(device, model, steps, prompt1)
        out_text2 = run_single(device, model, steps, prompt2)

        dist = textdistance.levenshtein.distance(out_text1.split(), out_text2.split())

        print("[Prompt 1]")
        print(out_text1)
        print("\n[Prompt 2]")
        print(out_text2)
        print(f"\nText distance between outputs: {dist:.3f}")

        record[f"output1_{steps}"] = out_text1
        record[f"output2_{steps}"] = out_text2
        record[f"distance_{steps}"] = dist

    results.append(record)

# Save results to CSV (one per split)
out_csv = build_split_output_path(csv_path, step_list, split)
results_df = pd.DataFrame(results)
results_df.to_csv(out_csv, index=False)

print(f"\n[split {split}] Saved results to: {out_csv}")
