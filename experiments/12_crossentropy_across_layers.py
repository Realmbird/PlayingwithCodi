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
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )

# %%
def compute_entropy_per_layer(hidden_states, lm_head, layer_norm=None, translators=None):
    """
    Compute prediction entropy at each layer for the last token position.
    Optionally applies tuned lens translators before projecting.
    
    Entropy H = -sum(p * log(p)) in nats.
    High entropy = uncertain/uncommitted. Low entropy = confident.
    """
    entropies = []
    for layer_idx, h in enumerate(hidden_states):
        h_last = h[:, -1, :]

        # Apply tuned lens translator if provided
        if translators is not None and layer_idx < len(translators):
            h_last = translators[layer_idx](h_last)

        if layer_norm is not None:
            h_last = layer_norm(h_last)

        logits = lm_head(h_last)
        # Upcast to float32 for numerical stability
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean()
        entropies.append(entropy.item())
    return entropies

# %% entropy analsysis
def run_entropy_analysis(model, tokenizer, prompts, num_latent_iterations=6,
                          translators=None):
    """
    Run entropy analysis across multiple prompts and average results.
    Returns mean entropy per layer for prompt position and each latent step.
    """
    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    num_layers = model.codi.config.num_hidden_layers + 1  # +1 for embedding layer
    # Shape: [num_latent_steps+1, num_layers] — index 0 = prompt position
    all_entropies = np.zeros((num_latent_iterations + 1, num_layers))
    count = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        bot_tensor = torch.tensor(
            [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
        ).unsqueeze(0)
        input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

        with torch.no_grad():
            outputs = model.codi(
                input_ids=input_ids_bot,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask_bot,
            )
            past_key_values = outputs.past_key_values

            # Prompt position entropy
            entropies = compute_entropy_per_layer(
                outputs.hidden_states, lm_head, layer_norm, translators
            )
            all_entropies[0] += np.array(entropies)

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

                entropies = compute_entropy_per_layer(
                    outputs.hidden_states, lm_head, layer_norm, translators
                )
                all_entropies[i + 1] += np.array(entropies)

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        count += 1

    return all_entropies / count  # mean across prompts, shape: [positions, layers]

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
print("Loading GSM8K-Aug prompts...")
train_texts = load_gsm8k_prompts(split="train", max_samples=500)
# %%
print("Running entropy analysis...")
# run_entropy_analysis(model, tokenizer, prompts, num_latent_iterations=6,
#                           translators=None)
# Load translators first, then pass the object not the path
translators_direct = load_translators("/home/chriskino/codi/tuned_lens/default_direct/default_tuned_6_GSM8K.pt", model)
translators_codi = load_translators("/home/chriskino/codi/tuned_lens/default_codi/default_codi_6_GSM8K.pt", model)
translators_even = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_even.pt", model)
translators_odd = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_odd.pt", model)
translators_3_5 = load_translators("/home/chriskino/codi/tuned_lens/custom_codi_GSM8K/default_codi_6_GSM8K_(3,5).pt", model)

print("Running entropy analysis...")
mean_entropies = run_entropy_analysis(model, tokenizer, train_texts, num_latent_iterations=6)
tuned_mean_entropies_direct = run_entropy_analysis(model, tokenizer, train_texts, num_latent_iterations=6, translators=translators_direct)
tuned_mean_entropies_codi = run_entropy_analysis(model, tokenizer, train_texts, num_latent_iterations=6, translators=translators_codi)
tuned_mean_entropies_even = run_entropy_analysis(model, tokenizer, train_texts, num_latent_iterations=6, translators=translators_even)
tuned_mean_entropies_odd = run_entropy_analysis(model, tokenizer, train_texts, num_latent_iterations=6, translators=translators_odd)
tuned_mean_entropies_3_5 = run_entropy_analysis(model, tokenizer, train_texts, num_latent_iterations=6, translators=translators_3_5)

# %%
def plot_entropy_across_latents(mean_entropies, num_latent_iterations=6,
                                 title="Prediction entropy across layers"):
    """
    One line per latent position (+ prompt), x-axis = layer.
    Each position gets its own color.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    layers = range(mean_entropies.shape[1])
    labels = ["Prompt"] + [f"Latent {i+1}" for i in range(num_latent_iterations)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(
            layers, mean_entropies[i],
            label=label,
            linewidth=2,
            color=color,
        )

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Entropy (nats)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_entropy_comparison(entropies_dict, latent_step=3, layer_idx=None,
                             title="Entropy comparison across lens types"):
    """
    Compare multiple lens types side by side.
    
    entropies_dict: dict of {label: mean_entropies array}
    latent_step: which latent step to compare (1-indexed), or None to show all
    layer_idx: if set, shows a bar chart at that specific layer instead of line plot
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(entropies_dict)))

    if layer_idx is not None:
        # Bar chart: entropy at a specific layer across lens types and latent steps
        labels = list(entropies_dict.keys())
        x = np.arange(7)  # prompt + 6 latents
        width = 0.8 / len(labels)
        for j, (label, entropies) in enumerate(entropies_dict.items()):
            ax.bar(x + j * width, entropies[:, layer_idx],
                   width=width, label=label, color=colors[j], alpha=0.8)
        ax.set_xticks(x + width * len(labels) / 2)
        ax.set_xticklabels(["Prompt"] + [f"L{i+1}" for i in range(6)])
        ax.set_xlabel("Position", fontsize=13)
        ax.set_ylabel(f"Entropy at layer {layer_idx} (nats)", fontsize=13)
    else:
        # Line plot: entropy across layers for a specific latent step
        pos_idx = latent_step  # 0=prompt, 1-6=latents
        layers = range(list(entropies_dict.values())[0].shape[1])
        for j, (label, entropies) in enumerate(entropies_dict.items()):
            ax.plot(layers, entropies[pos_idx], label=label,
                    color=colors[j], linewidth=2)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("Entropy (nats)", fontsize=13)
        pos_label = "Prompt" if latent_step == 0 else f"Latent {latent_step}"
        ax.set_title(f"{title} — {pos_label}", fontsize=14)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
# %%
os.makedirs("results/entropy", exist_ok=True)
# 1. Plain logit lens — entropy per latent step across layers
fig = plot_entropy_across_latents(mean_entropies, title="Plain logit lens — entropy across layers")
fig.savefig("results/entropy/plain_entropy.png", dpi=150, bbox_inches="tight")

# 2. Direct tuned lens
fig = plot_entropy_across_latents(tuned_mean_entropies_direct, title="Direct tuned lens — entropy across layers")
fig.savefig("results/entropy/direct_tuned_entropy.png", dpi=150, bbox_inches="tight")

# 3. Codi tuned lens
fig = plot_entropy_across_latents(tuned_mean_entropies_codi, title="CODI tuned lens — entropy across layers")
fig.savefig("results/entropy/codi_tuned_entropy.png", dpi=150, bbox_inches="tight")

# %%
# 4. Compare all lens types at latent step 3 (most interesting)
entropies_dict = {
    "Plain logit lens": mean_entropies,
    "Direct tuned lens": tuned_mean_entropies_direct,
    "CODI tuned lens": tuned_mean_entropies_codi,
    "Even steps": tuned_mean_entropies_even,
    "Odd steps": tuned_mean_entropies_odd,
    "Steps 3+5": tuned_mean_entropies_3_5,
}
# %%
os.makedirs("results/entropy_comparison", exist_ok=True)

labels = ["Prompt"] + [f"Latent {i+1}" for i in range(6)]

for i in range(7):
    label = labels[i]
    fig = plot_entropy_comparison(
        entropies_dict,
        latent_step=i,
        title=f"Entropy at {label} — lens comparison"
    )
    fig.savefig(f"results/entropy_comparison/comparison_{label.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)  # prevent memory buildup across 7 plots




