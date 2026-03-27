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

def tuned_logit_lens_all_positions(hidden_states, lm_head, layer_norm=None, top_k=5, translators=None):
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
        seq_len = h.shape[1]
        layer_results = []
        for pos in range(seq_len):
            h_last = h[:, pos, :]  # (batch, hidden)

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

            layer_results.append(
                {
                    "position": pos,
                    "top_indices": top_indices[0].cpu().tolist(),
                    "top_probs": top_probs[0].cpu().tolist(),
                }
            )

        results.append({"layer": layer_idx, "positions": layer_results})

    return results


def run_inference_with_tuned_logit_lens_all_positions(
    model, tokenizer, prompt, num_latent_iterations, top_k=5, translators_path=None
):
    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)

    translators = None
    if translators_path is not None:
        translators = load_translators(translators_path, model)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Store prompt tokens for labeling
    prompt_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)
    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)
    
    # All prompt tokens including special tokens
    all_prompt_tokens = [tokenizer.decode([tid]) for tid in input_ids_bot[0].tolist()]

    results = {
        "prompt": prompt,
        "prompt_tokens": all_prompt_tokens,
        "num_latent_iterations": num_latent_iterations,
        "prompt_hidden_states_lens": None,  # all positions
        "latent_positions": [],             # each is seq_len=1, but kept for consistency
    }

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # ALL positions for the prompt pass
        prompt_lens = tuned_logit_lens_all_positions(
            outputs.hidden_states, lm_head, layer_norm, top_k, translators
        )
        results["prompt_hidden_states_lens"] = prompt_lens

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd)
            latent_embd = latent_embd.to(dtype=model.codi.dtype)

        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Latent steps are still seq_len=1, so this is same as before
            latent_lens = tuned_logit_lens_all_positions(
                outputs.hidden_states, lm_head, layer_norm, top_k, translators
            )
            results["latent_positions"].append({"iteration": i, "logit_lens": latent_lens})

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)

    return results

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
print("Running inference with logit lens...")
results = run_inference_with_tuned_logit_lens_all_positions(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    num_latent_iterations=NUM_LATENT_ITERATIONS,
    top_k=TOP_K_TOKENS,
    translators_path="/home/chriskino/codi/tuned_lens/default_codi/default_codi_6_GSM8K.pt"
)
# %%
# Print detailed table
print_logit_lens_table(results, tokenizer, top_k=TOP_K_TOKENS)

# Visualize
fig = visualize_prompt_all_positions(results, tokenizer)
if fig is not None:
    results_dir = "results/default_codi"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "default_codi_6_GSM8K_all_tokens_300_dpi.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")
    plt.show()


# %%
results

# %%
