# ABOUTME: Logit lens visualization for CODI latent vectors.
# ABOUTME: Shows most likely tokens at each layer for each latent reasoning position.

# %%
import itertools
import torch
import torch.nn as nn
import json
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
print("Loading GSM8K-Aug prompts...")
train_texts = load_gsm8k_prompts(split="train", max_samples=500)
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



# %%
def visualize_prompt_all_positions(results, tokenizer, figsize=(20, 10)):
    prompt_lens = results["prompt_hidden_states_lens"]
    prompt_tokens = results["prompt_tokens"]
    
    num_layers = len(prompt_lens)
    num_positions = len(prompt_lens[0]["positions"])

    print(f"Debug: num_layers={num_layers}, num_positions={num_positions}")  # sanity check

    prob_matrix = np.zeros((num_layers, num_positions))
    token_matrix = [[None] * num_positions for _ in range(num_layers)]

    for layer_data in prompt_lens:
        layer_idx = layer_data["layer"]
        for pos_data in layer_data["positions"]:
            pos = pos_data["position"]
            prob_matrix[layer_idx, pos] = pos_data["top_probs"][0]
            token_matrix[layer_idx][pos] = tokenizer.decode([pos_data["top_indices"][0]])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax).set_label("Top-1 probability", rotation=270, labelpad=15)

    for i in range(num_layers):
        for j in range(num_positions):
            token = token_matrix[i][j]
            prob = prob_matrix[i, j]
            token_display = repr(token)[1:-1] if token else ""
            ax.text(j, i, token_display, ha="center", va="center",
                    color="white" if prob > 0.5 else "black", fontsize=8)

    ax.set_xlabel("Token position", fontsize=14, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels(
        [repr(t)[1:-1] for t in prompt_tokens[:num_positions]],
        rotation=90, fontsize=8
    )
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(i) for i in range(num_layers)], fontsize=10)
    plt.tight_layout()
    return fig


def print_logit_lens_table(results, tokenizer, top_k=5):
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
            # New structure: grab last position (seq_len=1 for latents)
            pos_results = layer_data["positions"][-1]
            tokens = [tokenizer.decode([tid]) for tid in pos_results["top_indices"][:top_k]]
            probs = pos_results["top_probs"][:top_k]
            token_str = " | ".join([f"{repr(t):>10s} ({p:.3f})" for t, p in zip(tokens, probs)])
            print(f"Layer {layer:2d}: {token_str}")


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

def visualize_logit_lens(results, tokenizer, figsize=(14, 10), title="Logit Lens Visualization", include_x = True):
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
            top_token_id = int(layer_data["top_indices"][0])
            top_prob = layer_data["top_probs"][0]

            prob_matrix[layer_idx, 0] = float(top_prob)
            token_matrix[layer_idx][0] = tokenizer.decode([top_token_id])

    # Add latent positions (columns 1 onwards)
    for pos_idx, pos_data in enumerate(results["latent_positions"]):
        col_idx = pos_idx + (1 if include_prompt else 0)
        for layer_data in pos_data["logit_lens"]:
            layer_idx = layer_data["layer"]
            top_token_id = layer_data["top_indices"][0]
            top_prob = float(layer_data["top_probs"][0])

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
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    
    ax.set_ylabel("Layer", fontsize=14, fontweight="bold")
   
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"{i}" for i in range(num_layers)], fontsize=12)
    if include_x:
        ax.set_xlabel("Latent vector index", fontsize=14, fontweight="bold")
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=12)

    plt.tight_layout()
    return fig
# %%
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

# %%
def diff_to_logit_lens(diff, translator=None, top_k=10):
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    device = next(model.parameters()).device

    step_results = []
    for layer in sorted(diff.keys()):
        h_vec = diff[layer].to(device=device, dtype=model.codi.dtype)
        h_last = h_vec

        if translator is not None and layer < len(translator):
            h_last = translator[layer](h_last)

        if layer_norm is not None:
            h_last = layer_norm(h_last)

        logits = lm_head(h_last)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        step_results.append({
            "layer": layer,
            "top_indices": top_indices[0].cpu().tolist(),
            "top_probs": top_probs[0].cpu().tolist(),
        })

    return {
        "prompt_hidden_states_lens": None,
        "latent_positions": [{"logit_lens": step_results}]
    }

# %%
def collect_latent_activations(model, tokenizer, prompts_with_answers,
                                num_latent_iterations=6):
    """
    Collect hidden states for each latent step across all layers.
    Returns dict mapping latent step -> list of hidden states per layer.
    """
    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    num_layers = model.codi.config.num_hidden_layers + 1

    # Store per-step activations: shape [num_prompts, hidden_dim] per layer per step
    step_sum = {
        step: {layer: None for layer in range(num_layers)}
        for step in range(1, num_latent_iterations + 1)
    }

    for item in prompts_with_answers:
        inputs = tokenizer(item, return_tensors="pt", padding=True)
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

                # Store activation at each layer for this step
                for layer_idx, h in enumerate(outputs.hidden_states):
                    h_vec = h[:, -1, :].cpu().float()
                    if step_sum[i + 1][layer_idx] is None:
                        step_sum[i + 1][layer_idx] = h_vec
                    else:
                        step_sum[i + 1][layer_idx] += h_vec

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

    # Convert lists to mean tensors: shape [hidden_dim] per layer per step
    mean_activations = {}
    for step in step_sum:
        mean_activations[step] = {}
        for layer in step_sum[step]:
            mean_activations[step][layer] = step_sum[step][layer] / len(prompts_with_answers)

    return mean_activations
# %%
def compute_activation_difference(mean_activations, step_a=5, step_b=3):
    """
    Compute activation difference between two latent steps per layer.
    Returns dict mapping layer -> difference vector.
    """
    num_layers = len(mean_activations[step_a])
    diff = {}
    for layer in range(num_layers):
        diff[layer] = mean_activations[step_a][layer] - mean_activations[step_b][layer]
    return diff
# %%
# code to save new activations and average across dataset
# gets latent activation mean from dataset for 6 latents and all layers
# mean_activations = collect_latent_activations(
#     model, tokenizer, train_texts,
#     num_latent_iterations=6
# )
# #  Save if you want uncomment
# torch.save(mean_activations, "mean_activations.pt")
# %%
# Load
mean_activations = torch.load("mean_activations.pt")

# %%
mean_activations

# %%
# diff = compute_activation_difference(mean_activations, step_a=6, step_b=1)

# results = diff_to_logit_lens(diff)

# fig = visualize_logit_lens(results, tokenizer, figsize=(14, 10), title="Average Activation Difference latent 6 - 1", include_x = False)
# if fig is not None:
#     results_dir = "results/activation_steer"
#     os.makedirs(results_dir, exist_ok=True)
#     output_path = os.path.join(results_dir, "Average Activation Difference latent 6 - 1.png")
#     fig.savefig(output_path, dpi=150, bbox_inches="tight")
#     print(f"\nSaved visualization to: {output_path}")
#     plt.show()


# %%
def activation_comparisons(subname, translator = None):
    # Define the pairs you want to see side-by-side
    # (step_a, step_b) will be plotted next to (step_b, step_a)
    latent_steps = [1, 2, 3, 4, 5, 6]
    pairs_to_compare = list(itertools.combinations(latent_steps, 2)) # Unique pairs

    results_dir = f"results/activation_steer_comparisons_{subname}"
    os.makedirs(results_dir, exist_ok=True)

    for step_a, step_b in pairs_to_compare:
        # Create a double-wide figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # --- Plot A minus B ---
        diff_ab = compute_activation_difference(mean_activations, step_a=step_a, step_b=step_b)
        res_ab = diff_to_logit_lens(diff_ab, translator)
        
        # We'll use a slightly modified version of your function to plot on a specific axis
        # Note: You may need to tweak visualize_logit_lens to accept an 'ax' argument
        # or simply call the plotting logic manually here:
        
        for ax, step_1, step_2 in [(ax1, step_a, step_b), (ax2, step_b, step_a)]:
            current_diff = compute_activation_difference(mean_activations, step_a=step_1, step_b=step_2)
            current_res = diff_to_logit_lens(current_diff, translator)
            
            # Extracting data for manual plotting since visualize_logit_lens creates its own fig
            num_layers = len(current_res["latent_positions"][0]["logit_lens"])
            prob_matrix = np.zeros((num_layers, 1))
            token_matrix = []

            for layer_data in current_res["latent_positions"][0]["logit_lens"]:
                l_idx = layer_data["layer"]
                prob_matrix[l_idx, 0] = layer_data["top_probs"][0]
                token_matrix.append(tokenizer.decode([layer_data["top_indices"][0]]))

            im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            ax.set_title(f"Latent {step_1} - {step_2} {subname}", fontsize=18, fontweight="bold")
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels([str(i) for i in range(num_layers)])
            ax.set_xticks([])
            
            # Annotate
            for i, token in enumerate(token_matrix):
                prob = prob_matrix[i, 0]
                token_display = repr(token)[1:-1]
                ax.text(0, i, token_display, ha="center", va="center", 
                        color="white" if prob > 0.5 else "black", fontsize=14)

        plt.colorbar(im, ax=[ax1, ax2], label="Top-1 Probability")
        
        save_path = os.path.join(results_dir, f"compare_{step_a}_and_{step_b} {subname}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison: {save_path}")
        plt.show()
        plt.close(fig)
# %%

def visualize_all_differences_tuned(mean_activations, tokenizer, model, translator=None, subname=""):
    latent_steps = sorted(mean_activations.keys())
    num_steps = len(latent_steps)
    num_layers = model.codi.config.num_hidden_layers + 1
    
    fig, axes = plt.subplots(num_steps, num_steps, figsize=(30, 40), sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    device = next(model.parameters()).device

    for i, step_a in enumerate(latent_steps):
        for j, step_b in enumerate(latent_steps):
            ax = axes[i, j]
            if step_a == step_b:
                ax.text(0.5, 0.5, "Identity", ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                continue

            prob_matrix = np.zeros((num_layers, 1))
            tokens = []
            
            for layer in range(num_layers):
                # 1. Get the difference vector
                h = (mean_activations[step_a][layer] - mean_activations[step_b][layer]).to(device=device, dtype=model.codi.dtype)
                
                # 2. APPLY TUNED LENS TRANSLATOR HERE
                if translator is not None and layer < len(translator):
                    h = translator[layer](h)

                # 3. Optional final norm (usually part of the tuned lens pipeline)
                if layer_norm is not None: 
                    h = layer_norm(h)
                
                # 4. Project
                logits = lm_head(h)
                probs = torch.softmax(logits, dim=-1)
                top_p, top_i = torch.topk(probs, 1, dim=-1)
                
                prob_matrix[layer, 0] = top_p.item()
                tokens.append(tokenizer.decode([top_i.item()]))

            im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
            
            for layer_idx, token in enumerate(tokens):
                p = prob_matrix[layer_idx, 0]
                display_text = repr(token)[1:-1]
                ax.text(0, layer_idx, display_text, ha="center", va="center", 
                        color="white" if p > 0.4 else "black", fontsize=9)

            if i == 0: ax.set_title(f"Minus Step {step_b}", fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"Step {step_a}", fontsize=14, fontweight='bold')
            ax.set_xticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Top-1 Probability")
    plt.suptitle(f"Tuned Latent Difference Matrix {subname} ($Step_A - Step_B$)", fontsize=28, fontweight='bold', y=0.95)
    
    return fig
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
activation_comparisons("custom tuned 5", translators_5)
# %%
 # Execute

full_fig = visualize_all_differences_tuned(translators_5, tokenizer, model, translators_direct, "custom tuned 5")

full_fig.savefig("results/latent_difference_matrix_full_custom tuned 5.png", dpi=200, bbox_inches="tight")

plt.show()


# %%
