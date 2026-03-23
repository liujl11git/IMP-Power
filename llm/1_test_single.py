import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",type=int,default=0)
parser.add_argument("--num_steps",type=str,default='2,4,8,16,32')
parser.add_argument("sentence",nargs="+",help="Input sentence (can contain spaces).")
args = parser.parse_args()

def run_single(device, model, num_steps, prompt):
    """
    Run a single generation with the given GPU, model name, num_steps and prompt.
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

# --------- test loop over num_steps ---------
prompt = args.sentence[0]
step_list = [int(ss) for ss in args.num_steps.split(',')]
for steps in step_list:
    print("\n" + "=" * 40)
    print(f"num_steps = {steps}")
    print("=" * 40)
    out_text = run_single(device, model, steps, prompt)
    print(out_text)
