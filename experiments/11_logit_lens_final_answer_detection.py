# ABOUTME: Logit lens visualization for CODI latent vectors.
# ABOUTME: Shows most likely tokens at each layer for each latent reasoning position.

# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CODI

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
DTYPE = "bfloat16"

PROMPT = "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team? Give the answer only and nothing else."
NUM_LATENT_ITERATIONS = 10
TOP_K_TOKENS = 10  # Number of top tokens to display per layer


# %%
def get_lm_head(model):
    """Get the language model head (unembedding matrix) from the model."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        return codi.get_base_model().lm_head
    return codi.lm_head


def get_layer_norm(model):
    """Get the final layer norm before the lm_head."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        base = codi.get_base_model()
    else:
        base = codi

    # LLaMA/Mistral architecture
    if hasattr(base, "model") and hasattr(base.model, "norm"):
        return base.model.norm
    # GPT-2 architecture
    if hasattr(base, "transformer") and hasattr(base.transformer, "ln_f"):
        return base.transformer.ln_f
    return None
# %%
# %% Tunedlogit_lens class
class TunedLensTranslator(nn.Module):
    """One affine translator per layer."""
    def __init__(self, hidden_dim, device, dtype=torch.bfloat16):
        super().__init__()
        self.translator = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        # Initialize as identity so it starts like plain logit lens
        nn.init.eye_(self.translator.weight)
        nn.init.zeros_(self.translator.bias)

    def forward(self, x):
        # Ensure input matches translator dtype
        return self.translator(x.to(dtype=self.translator.weight.dtype))
def load_translators(path, model):
    """Load trained translators from a saved checkpoint."""
    device = next(model.parameters()).device
    target_dtype = model.codi.dtype if hasattr(model.codi, "dtype") else torch.bfloat16
    hidden_dim = model.codi.config.hidden_size
    num_layers = model.codi.config.num_hidden_layers

    translators = torch.nn.ModuleList([
        TunedLensTranslator(hidden_dim, device=device, dtype=target_dtype)
        for _ in range(num_layers)
    ]).to(device=device, dtype=target_dtype)

    translators.load_state_dict(torch.load(path, map_location=device))
    translators.eval()
    return translators
#%%
def load_tuned_logit_lens(path, model):
    """Load trained translators from a saved checkpoint."""
    device = next(model.parameters()).device
    target_dtype = model.codi.dtype if hasattr(model.codi, "dtype") else torch.bfloat16
    hidden_dim = model.codi.config.hidden_size
    num_layers = model.codi.config.num_hidden_layers

    translators = torch.nn.ModuleList([
        TunedLensTranslator(hidden_dim, device=device, dtype=target_dtype)
        for _ in range(num_layers)
    ]).to(device=device, dtype=target_dtype)

    translators.load_state_dict(torch.load(path, map_location=device))
    translators.eval()
    return translators

#%%

def tuned_logit_lens(hidden_states, lm_head, layer_norm=None, top_k=5, translators=None):
    """
    Apply logit lens to hidden states.

    Args:
        hidden_states: Tuple of hidden states from each layer, each of shape (batch, seq, hidden)
        lm_head: The unembedding matrix (linear layer)
        layer_norm: Optional final layer norm to apply before unembedding
        top_k: Number of top tokens to return

    Returns:
        List of (layer_idx, top_tokens, top_probs) for the last position
    """
    results = []

    for layer_idx, h in enumerate(hidden_states):
        # Take the last position
        h_last = h[:, -1, :]  # (batch, hidden)

        if translators is not None and layer_idx < len(translators):
            h_last = translators[layer_idx](h_last)


        # Optionally apply layer norm (for fair comparison with final output)
        if layer_norm is not None:
            h_last = layer_norm(h_last)

        # Project through unembedding
        logits = lm_head(h_last)  # (batch, vocab)
        probs = torch.softmax(logits, dim=-1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        results.append(
            {
                "layer": layer_idx,
                "top_indices": top_indices[0].cpu().tolist(),
                "top_probs": top_probs[0].cpu().tolist(),
            }
        )

    return results


def run_inference_with_tuned_logit_lens(
    model, tokenizer, prompt, num_latent_iterations, top_k=5,  translators_path=None
):
    """
    Run CODI inference and capture logit lens results for each latent position.

    Returns:
        Dict with prompt_logit_lens and latent_logit_lens for each iteration
    """
    device = next(model.parameters()).device

    # Get model components
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)

    # Load translators if path provided
    translators = None
    if translators_path is not None:
        translators = load_translators(translators_path, model)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Special tokens
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    # Add BOT tokens
    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

    results = {
        "prompt": prompt,
        "num_latent_iterations": num_latent_iterations,
        "prompt_hidden_states_lens": None,
        "latent_positions": [],
    }

    with torch.no_grad():
        # Encode prompt
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # Logit lens on prompt (last position = after <|bocot|>)
        prompt_lens = tuned_logit_lens(outputs.hidden_states, lm_head, layer_norm, top_k, translators)
        results["prompt_hidden_states_lens"] = prompt_lens

        # Get initial latent embedding
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Optionally project
        if model.use_prj:
            latent_embd = model.prj(latent_embd)
            latent_embd = latent_embd.to(dtype=model.codi.dtype)

        # Latent iterations
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Logit lens on this latent position
            latent_lens = tuned_logit_lens(outputs.hidden_states, lm_head, layer_norm, top_k, translators)
            results["latent_positions"].append(
                {"iteration": i, "logit_lens": latent_lens}
            )

            # Get next latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)

    return results


# %%
def visualize_logit_lens(results, tokenizer, figsize=(14, 10)):
    """
    Visualize logit lens results as a heatmap with token annotations.
    """
    num_latent = len(results["latent_positions"])
    if num_latent == 0:
        print("No latent positions to visualize")
        return

    num_layers = len(results["latent_positions"][0]["logit_lens"])

    # Include prompt position (one position to the left)
    include_prompt = results["prompt_hidden_states_lens"] is not None
    num_positions = num_latent + (1 if include_prompt else 0)

    # Create matrix of top-1 probabilities
    prob_matrix = np.zeros((num_layers, num_positions))
    token_matrix = [[None] * num_positions for _ in range(num_layers)]

    # Add prompt position (column 0)
    if include_prompt:
        for layer_data in results["prompt_hidden_states_lens"]:
            layer_idx = layer_data["layer"]
            top_token_id = layer_data["top_indices"][0]
            top_prob = layer_data["top_probs"][0]

            prob_matrix[layer_idx, 0] = top_prob
            token_matrix[layer_idx][0] = tokenizer.decode([top_token_id])

    # Add latent positions (columns 1 onwards)
    for pos_idx, pos_data in enumerate(results["latent_positions"]):
        col_idx = pos_idx + (1 if include_prompt else 0)
        for layer_data in pos_data["logit_lens"]:
            layer_idx = layer_data["layer"]
            top_token_id = layer_data["top_indices"][0]
            top_prob = layer_data["top_probs"][0]

            prob_matrix[layer_idx, col_idx] = top_prob
            token_matrix[layer_idx][col_idx] = tokenizer.decode([top_token_id])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=14)

    # Annotate cells with tokens
    for i in range(num_layers):
        for j in range(num_positions):
            token = token_matrix[i][j]
            prob = prob_matrix[i, j]
            # Escape special characters for display
            token_display = repr(token)[1:-1] if token else ""
            text_color = "white" if prob > 0.5 else "black"
            ax.text(
                j,
                i,
                token_display,
                ha="center",
                va="center",
                color=text_color,
                fontsize=16,
            )

    # Labels
    ax.set_xlabel("Latent vector index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=12)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"{i}" for i in range(num_layers)], fontsize=12)

    plt.tight_layout()
    return fig


def print_logit_lens_table(results, tokenizer, top_k=5):
    """
    Print a detailed table of logit lens results.
    """
    print("=" * 80)
    print(f"Prompt: {results['prompt']}")
    print(f"Number of latent iterations: {results['num_latent_iterations']}")
    print("=" * 80)

    for pos_idx, pos_data in enumerate(results["latent_positions"]):
        print(f"\n{'=' * 40}")
        print(f"LATENT POSITION {pos_idx}")
        print(f"{'=' * 40}")

        for layer_data in pos_data["logit_lens"]:
            layer = layer_data["layer"]
            tokens = [
                tokenizer.decode([tid]) for tid in layer_data["top_indices"][:top_k]
            ]
            probs = layer_data["top_probs"][:top_k]

            token_str = " | ".join(
                [f"{repr(t):>10s} ({p:.3f})" for t, p in zip(tokens, probs)]
            )
            print(f"Layer {layer:2d}: {token_str}")


# %%
def get_token_ids_for_value(tokenizer, value_str):
    """
    Get all plausible token ids for a numeric value.
    Handles tokenization variants: '8', ' 8', '\n8' etc.
    Only keeps single-token representations.
    """
    candidates = [value_str, " " + value_str, "\n" + value_str]
    token_ids = set()
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:  # only single-token representations
            token_ids.update(ids)
    return token_ids
def compute_topk_detection_rates(model, tokenizer, prompts_with_answers,
                                   num_latent_iterations=6, top_k_values=None,
                                   translators=None):
    """
    Sweep over multiple top-k values and compute detection rate for
    intermediate and final answers at each latent step.
    
    Returns dict keyed by top_k value, each containing
    intermediate_rate and final_rate arrays of shape [num_positions, num_layers]
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    num_layers = model.codi.config.num_hidden_layers + 1
    num_positions = num_latent_iterations + 1
    total = len(prompts_with_answers)

    # Initialize counts for each top_k value
    counts = {
        k: {
            "intermediate": np.zeros((num_positions, num_layers)),
            "final": np.zeros((num_positions, num_layers)),
        }
        for k in top_k_values
    }

    for item in prompts_with_answers:
        inter_ids = get_token_ids_for_value(tokenizer, item["intermediate"])
        final_ids = get_token_ids_for_value(tokenizer, item["final"])

        inputs = tokenizer(item["prompt"], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        bot_tensor = torch.tensor(
            [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
        ).unsqueeze(0)
        input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask_bot = torch.cat(
            (attention_mask, torch.ones_like(bot_tensor)), dim=1
        )

        with torch.no_grad():
            outputs = model.codi(
                input_ids=input_ids_bot,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask_bot,
            )
            past_key_values = outputs.past_key_values

            # Collect all hidden states per position
            # position 0 = prompt, 1..6 = latents
            all_position_hidden_states = [outputs.hidden_states]

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            for i in range(num_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                all_position_hidden_states.append(outputs.hidden_states)

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Now compute detection for each top_k, position, and layer
        # Do this outside torch.no_grad loop to avoid rerunning forward passes
        max_k = max(top_k_values)

        for pos_idx, hidden_states in enumerate(all_position_hidden_states):
            for layer_idx, h in enumerate(hidden_states):
                h_last = h[:, -1, :]

                if translators is not None and layer_idx < len(translators):
                    h_last = translators[layer_idx](h_last)
                if layer_norm is not None:
                    h_last = layer_norm(h_last)

                logits = lm_head(h_last)
                # Get top max_k once, then slice for each k
                _, top_indices = torch.topk(logits, max_k, dim=-1)
                top_ids_list = top_indices[0].cpu().tolist()

                for k in top_k_values:
                    top_ids_k = set(top_ids_list[:k])
                    if top_ids_k & inter_ids:
                        counts[k]["intermediate"][pos_idx, layer_idx] += 1
                    if top_ids_k & final_ids:
                        counts[k]["final"][pos_idx, layer_idx] += 1

    # Convert counts to rates
    results = {}
    for k in top_k_values:
        results[k] = {
            "intermediate_rate": counts[k]["intermediate"] / total,
            "final_rate": counts[k]["final"] / total,
            "num_layers": num_layers,
            "num_positions": num_positions,
        }
    return results
# %%
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )

#%%
load_dotenv()

print("Loading model...")
model = CODI.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    model_name_or_path=MODEL_NAME_OR_PATH,
    lora_r=128,
    lora_alpha=32,
    num_latent=6,
    use_prj=True,
    device=DEVICE,
    dtype=DTYPE,
    strict=False,
    checkpoint_save_path=f"./checkpoints/{CHECKPOINT_PATH}",
    remove_eos=False,
    full_precision=True,
)
tokenizer = model.tokenizer
ensure_tokenizer_special_tokens(tokenizer, model)

# %%
from datasets import load_dataset

def load_gsm8k_prompts(split="train", max_samples=500):
    """Load prompts from GSM8K-Aug dataset."""
    dataset = load_dataset("zen-E/GSM8k-Aug", split=split)
    prompts = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        # GSM8K-Aug has 'question' field
        prompts.append(example["question"])
    return prompts

# %%
# Load translators first, then pass the object not the path
translators_direct = load_translators("/home/chriskino/codi/tuned_lens/default_direct/default_tuned_6_GSM8K.pt", model)
translators_codi = load_translators("/home/chriskino/codi/tuned_lens/default_codi/default_codi_6_GSM8K.pt", model)
translators_even = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_even.pt", model)
translators_odd = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_odd.pt", model)
translators_3_5 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(3,5).pt", model)
translators_1 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(1).pt", model)
translators_1_3 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(1,3).pt", model)
translators_1_5 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(1,5).pt", model)
translators_3 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(3).pt", model)
translators_5 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(5).pt", model)

# %%
import re

def extract_intermediates_from_steps(steps):
    """
    Extract all intermediate values from steps list.
    e.g. ['<<600*30/100=180>>', '<<600*10/100=60>>'] -> ['180', '60']
    """
    intermediates = []
    for step in steps:
        match = re.search(r'=(-?\d+\.?\d*)>>', step)
        if match:
            intermediates.append(match.group(1))
    return intermediates

# %%
dataset = load_dataset("whynlp/gsm8k-aug", split="train")
prompts_with_answers = []
for i, example in enumerate(dataset):
    if i >= 500:
        break
    final = str(example["answer"])
    intermediates = extract_intermediates_from_steps(example["steps"])
    if not intermediates:
        continue
    prompts_with_answers.append({
        "prompt": example["question"],
        "intermediate": intermediates[-1],  # first intermediate step
        "final": final,
    })

print(f"Built {len(prompts_with_answers)} prompts")
print(f"Example: {prompts_with_answers[0]}")

# %%
top_k_values = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]

# Run all translators
translator_configs = {
    "Plain logit lens": None,
    "Direct tuned lens": translators_direct,
    "CODI tuned lens": translators_codi,
    "Even steps lens": translators_even,
    "Odd steps lens": translators_odd,
    "Steps 3+5 lens": translators_3_5,
    "Steps 1": translators_1,
    "Steps 1+3": translators_1_3,
    "Steps 1+5": translators_1_5,
    "Step 3": translators_3,
    "Step 5": translators_5
}

all_topk_results = {}
for name, translators in translator_configs.items():
    print(f"Running {name}...")
    all_topk_results[name] = compute_topk_detection_rates(
        model, tokenizer, prompts_with_answers,
        num_latent_iterations=6,
        top_k_values=top_k_values,
        translators=translators,
    )

# %%
def plot_final_detection_per_latent(all_topk_results, top_k=10):
    """
    Grid of subplots: one per lens type.
    Each subplot: x=layer, y=final answer detection rate.
    One line per latent position. Shows how each lens reads latent vectors.
    """
    n_lenses = len(all_topk_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    labels = ["Prompt"] + [f"Latent {i+1}" for i in range(6)]
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    for ax, (name, results) in zip(axes, all_topk_results.items()):
        rates = results[top_k]["final_rate"]  # shape [num_positions, num_layers]
        num_layers = results[top_k]["num_layers"]

        for i, (label, color) in enumerate(zip(labels, colors)):
            is_highlight = i in {3, 5}
            ax.plot(
                range(num_layers), rates[i],
                label=label, color=color,
                linewidth=3 if is_highlight else 1.5,
                alpha=1.0 if is_highlight else 0.7,
            )
        ax.set_title(name, fontsize=20, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Layer", fontsize=20)
        ax.set_ylabel("Final answer detection rate", fontsize=20)

    # Shared legend
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=7, fontsize=20,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Final answer detection rate across layers (top-{top_k})",
                 fontsize=25, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_final_detection_avg_across_layers(all_topk_results, top_k=10):
    """
    One plot: x=layer, y=final answer detection rate averaged across all latent positions.
    One line per lens type. Shows which lens detects the answer earliest on average.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_topk_results)))

    for (name, results), color in zip(all_topk_results.items(), colors):
        rates = results[top_k]["final_rate"]  # [num_positions, num_layers]
        # Average across all positions (prompt + latents)
        avg_rate = rates.mean(axis=0)
        num_layers = results[top_k]["num_layers"]
        ax.plot(range(num_layers), avg_rate, label=name, color=color, linewidth=2)
    
    ax.set_xlabel("Layer", fontsize=20)
    ax.set_ylabel("Final answer detection rate (avg across positions)", fontsize=20)
    ax.set_title(f"Final answer detection rate averaged across latent positions (top-{top_k})",
                 fontsize=20)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=25)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# %%
os.makedirs("results/detection", exist_ok=True)

fig = plot_final_detection_per_latent(all_topk_results, top_k=10)
fig.savefig("results/detection/final_detection_per_latent.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig = plot_final_detection_avg_across_layers(all_topk_results, top_k=10)
fig.savefig("results/detection/final_detection_avg.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Done")

# %% 
def plot_final_detection_per_latent_scaled(all_topk_results, top_k=10):
    n_lenses = len(all_topk_results)
    fig, axes = plt.subplots(3, 3, figsize=(22, 12), sharex=True, sharey=False)
    axes = axes.flatten()

    labels = ["Prompt"] + [f"Latent {i+1}" for i in range(6)]
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    for ax, (name, results) in zip(axes, all_topk_results.items()):
        rates = results[top_k]["final_rate"]
        num_layers = results[top_k]["num_layers"]

        for i, (label, color) in enumerate(zip(labels, colors)):
            is_highlight = i in {3, 5}
            ax.plot(
                range(num_layers), rates[i],
                label=label, color=color,
                linewidth=3 if is_highlight else 1.5,
                alpha=1.0 if is_highlight else 0.7,
            )

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Final answer detection rate", fontsize=11)
        # Auto y-axis with small padding
        ymax = rates.max()
        ax.set_ylim(0, max(ymax * 1.15, 0.05))  # at least 0.05 so flat plots aren't just a line

    # Single shared legend below all subplots
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=7, fontsize=11,
               bbox_to_anchor=(0.5, -0.03), frameon=True)

    fig.suptitle(f"Final answer detection rate across layers (top-{top_k})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig
# %% 
os.makedirs("results/detection", exist_ok=True)

fig = plot_final_detection_per_latent_scaled(all_topk_results, top_k=10)
fig.savefig("results/detection/scaled_final_detection_per_latent.png", dpi=200, bbox_inches="tight")
plt.close(fig)
# %%
def plot_detection_across_positions_by_topk(all_topk_results, answer_type="final", 
                                              topk_to_show=None, save_dir=None):
    if topk_to_show is None:
        topk_to_show = [1, 3, 5, 10]
    
    top_k_values = sorted(topk_to_show)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(top_k_values)))
    labels = ["Prompt"] + [f"Latent {i+1}" for i in range(6)]
    
    figs = {}
    for name, results in all_topk_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        num_positions = results[top_k_values[0]][f"{answer_type}_rate"].shape[0]
        
        for k, color in zip(top_k_values, colors):
            rate = results[k][f"{answer_type}_rate"]
            avg_per_position = rate[:, -3:].mean(axis=1)
            ax.plot(range(num_positions), avg_per_position,
                    label=f"top-{k}", color=color,
                    linewidth=2, marker="o", markersize=5)
        
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Latent position", fontsize=11)
        ax.set_ylabel(f"{answer_type.capitalize()} detection rate", fontsize=11)
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.grid(True, alpha=0.3)
        ymax = max(
            results[k][f"{answer_type}_rate"][:, -3:].mean(axis=1).max()
            for k in top_k_values
        )
        ax.set_ylim(0, max(ymax * 1.15, 0.05))
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = name.replace(" ", "_").replace("+", "plus")
            fig.savefig(f"{save_dir}/{answer_type}_{safe_name}.png", 
                       dpi=150, bbox_inches="tight")
            plt.close(fig)
        
        figs[name] = fig
    
    return figs

# %%
figs = plot_detection_across_positions_by_topk(
    all_topk_results, 
    answer_type="final",
    save_dir="results/detection/per_lens"
)
# %%
# %%
# === THEREFORE TOKEN DETECTION ===

# Get therefore token ids
therefore_ids = set()
for variant in ["therefore", "Therefore", " therefore", " Therefore", "\ntherefore", "\nTherefore"]:
    ids = tokenizer.encode(variant, add_special_tokens=False)
    if len(ids) == 1:
        therefore_ids.update(ids)
print(f"Therefore token ids: {therefore_ids}")

def compute_topk_therefore_rates(model, tokenizer, prompts_with_answers,
                                  therefore_ids, num_latent_iterations=6,
                                  top_k_values=None, translators=None):
    """Same as compute_topk_detection_rates but only tracks 'therefore' token."""
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    num_layers = model.codi.config.num_hidden_layers + 1
    num_positions = num_latent_iterations + 1
    total = len(prompts_with_answers)

    counts = {k: np.zeros((num_positions, num_layers)) for k in top_k_values}

    for item in prompts_with_answers:
        inputs = tokenizer(item["prompt"], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        bot_tensor = torch.tensor(
            [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
        ).unsqueeze(0)
        input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask_bot = torch.cat(
            (attention_mask, torch.ones_like(bot_tensor)), dim=1
        )

        with torch.no_grad():
            outputs = model.codi(
                input_ids=input_ids_bot,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask_bot,
            )
            past_key_values = outputs.past_key_values
            all_position_hidden_states = [outputs.hidden_states]

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            for i in range(num_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                all_position_hidden_states.append(outputs.hidden_states)

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        max_k = max(top_k_values)
        for pos_idx, hidden_states in enumerate(all_position_hidden_states):
            for layer_idx, h in enumerate(hidden_states):
                h_last = h[:, -1, :]
                if translators is not None and layer_idx < len(translators):
                    h_last = translators[layer_idx](h_last)
                if layer_norm is not None:
                    h_last = layer_norm(h_last)
                logits = lm_head(h_last)
                _, top_indices = torch.topk(logits, max_k, dim=-1)
                top_ids_list = top_indices[0].cpu().tolist()

                for k in top_k_values:
                    if set(top_ids_list[:k]) & therefore_ids:
                        counts[k][pos_idx, layer_idx] += 1

    return {k: {"therefore_rate": counts[k] / total,
                "num_layers": num_layers,
                "num_positions": num_positions}
            for k in top_k_values}

# %%
all_therefore_results = {}
for name, translators in translator_configs.items():
    print(f"Running therefore detection: {name}...")
    all_therefore_results[name] = compute_topk_therefore_rates(
        model, tokenizer, prompts_with_answers,
        therefore_ids=therefore_ids,
        num_latent_iterations=6,
        top_k_values=top_k_values,
        translators=translators,
    )

# %%
# Reuse plot_detection_across_positions_by_topk but for therefore
# Need a small wrapper since the key is "therefore_rate" not "final_rate"
def plot_therefore_across_positions_by_topk(all_therefore_results,
                                             topk_to_show=None, save_dir=None):
    if topk_to_show is None:
        topk_to_show = [1, 3, 5, 10]

    top_k_values = sorted(topk_to_show)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(top_k_values)))
    labels = ["Prompt"] + [f"Latent {i+1}" for i in range(6)]

    figs = {}
    for name, results in all_therefore_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        num_positions = results[top_k_values[0]]["therefore_rate"].shape[0]

        for k, color in zip(top_k_values, colors):
            rate = results[k]["therefore_rate"]
            avg_per_position = rate[:, -3:].mean(axis=1)
            ax.plot(range(num_positions), avg_per_position,
                    label=f"top-{k}", color=color,
                    linewidth=2, marker="o", markersize=5)

        ax.set_title(f"{name}\n'Therefore' token detection", fontsize=13, fontweight="bold")
        ax.set_xlabel("Latent position", fontsize=11)
        ax.set_ylabel("'Therefore' detection rate", fontsize=11)
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.grid(True, alpha=0.3)
        ymax = max(
            results[k]["therefore_rate"][:, -3:].mean(axis=1).max()
            for k in top_k_values
        )
        ax.set_ylim(0, max(ymax * 1.15, 0.05))
        ax.legend(fontsize=10)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = name.replace(" ", "_").replace("+", "plus")
            fig.savefig(f"{save_dir}/therefore_{safe_name}.png",
                       dpi=150, bbox_inches="tight")
            plt.close(fig)

        figs[name] = fig
    return figs

# %%
figs_therefore = plot_therefore_across_positions_by_topk(
    all_therefore_results,
    save_dir="results/detection/therefore"
)

# %%
