# %%
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import Tensor
import pickle
from jaxtyping import Float

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
DTYPE = "bfloat16"

PROMPT = "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team? Give the answer only and nothing else."
NUM_LATENT_ITERATIONS = 6
TOP_K_TOKENS = 10  # Number of top tokens to display per layer
# %% 
# loading codi Prompts
def load_prompts_from_json():
    """Load all prompts from prompts.json file."""
    with open(PROMPTS_JSON_PATH, "r") as f:
        data = json.load(f)
    return data["prompts"]

PROMPTS_JSON_PATH = "/home/chriskino/codi/prompts/prompts.json" # changed since different location
print("\n" + "=" * 80)
print("LOADING PROMPTS FROM JSON")
print("=" * 80)

# Load prompts from JSON file
all_prompts = load_prompts_from_json()
print(f"Loaded {len(all_prompts)} prompts from {PROMPTS_JSON_PATH}")

# %%
def get_lm_head(model):
    """Get the language model head (unembedding matrix) from the model."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        return codi.get_base_model().lm_head
    return codi.lm_head

# %%
def get_token_ids_for_value(tokenizer, value_str):
    candidates = [value_str, " " + value_str, "\n" + value_str]
    token_ids = set()
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.update(ids)
    return token_ids
# %%
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


def logit_lens(hidden_states, lm_head, layer_norm=None, top_k=5):
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


def run_inference_with_logit_lens(
    model, tokenizer, prompt, num_latent_iterations, top_k=5
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
        prompt_lens = logit_lens(outputs.hidden_states, lm_head, layer_norm, top_k)
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
            latent_lens = logit_lens(outputs.hidden_states, lm_head, layer_norm, top_k)
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
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )

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
# %% linear probe
import torch as t
class LRProbe(t.nn.Module):
    def __init__(self, d_in: int, scaler_mean: Tensor | None = None, scaler_scale: Tensor | None = None):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_model"]:
        """Apply StandardScaler normalization if scaler parameters are available."""
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self(x).round()

    @property
    def direction(self) -> Float[Tensor, " d_model"]:
        return self.net[0].weight.data[0]

    @staticmethod
    def from_data(
        acts: Float[Tensor, "n d_model"],
        labels: Float[Tensor, " n"],
        C: float = 0.1,
        device: str = "cpu",
    ) -> "LRProbe":
        """
        Train an LR probe using sklearn's LogisticRegression with StandardScaler normalization.

        Args:
            acts: Activation matrix [n_samples, d_model].
            labels: Binary labels (1=true, 0=false).
            C: Inverse regularization strength (lower = stronger regularization).
                Default 0.1 (reg_coeff=10) matches the deception-detection paper's cfg.yaml.
                The repo class default is reg_coeff=1000 (C=0.001), which is stronger.
            device: Device to place the resulting probe on.
        """
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        # Standardize features (zero mean, unit variance) before fitting, as in the paper
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # fit_intercept=False: the paper fits on normalized data so the intercept is redundant
        lr_model = LogisticRegression(C=C, random_state=42, fit_intercept=False, max_iter=1000)
        lr_model.fit(X_scaled, y)

        # Build probe with scaler parameters baked in
        scaler_mean = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.tensor(scaler.scale_, dtype=t.float32)
        probe = LRProbe(acts.shape[-1], scaler_mean=scaler_mean, scaler_scale=scaler_scale).to(device)
        probe.net[0].weight.data[0] = t.tensor(lr_model.coef_[0], dtype=t.float32).to(device)

        return probe


# %%
dataset = load_dataset("whynlp/gsm8k-aug", split="train")
prompts_with_answers = []

for i, example in enumerate(dataset):
    final = str(example["answer"])
    intermediates = extract_intermediates_from_steps(example["steps"])
    if not intermediates:
        continue
    for intermediate in intermediates:
        inter_ids = get_token_ids_for_value(tokenizer, intermediate)
        final_ids = get_token_ids_for_value(tokenizer, final)
        if not inter_ids or not final_ids:
            continue
        if inter_ids & final_ids:
            continue
        prompts_with_answers.append({
            "prompt": example["question"],
            "intermediate": intermediate,
            "final": final,
        })
        break
    if len(prompts_with_answers) >= 300:
        break

print(f"Built {len(prompts_with_answers)} usable prompts")
print(f"Example: {prompts_with_answers[0]}")

# %%
def get_token_ids_for_value_relaxed(tokenizer, value_str):
    """
    Get token ids for a numeric value.
    For single-token numbers returns that token.
    For multi-token numbers returns the first token as a proxy.
    """
    candidates = [value_str, " " + value_str, "\n" + value_str]
    token_ids = set()
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) >= 1:  # allow multi-token, take first token
            token_ids.add(ids[0])
    return token_ids


#%%
def save_probe_dataset(activations, labels, step_ids, meta, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "activations": activations,
            "labels": labels,
            "step_ids": step_ids,
            "meta": meta
        }, f)
    print(f"Saved {len(activations)} samples to {path}")

def load_probe_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["activations"], data["labels"], data["step_ids"], data["meta"]

def collect_probe_dataset(model, tokenizer, prompts_with_answers,
                           num_latent_iterations=6, layer_idx=10, top_k=10):
    """
    Collect activations labeled as intermediate (0) or final (1).
    Labels determined by last layer output (what model actually outputs).
    Activations saved from layer_idx (what probe trains on).
    Skips ambiguous cases where both or neither appear in top-k.
    Skips prompts where intermediate == final (single-step problems).
    """
    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    activations = []
    labels = []
    step_ids = []
    meta = []
    skipped_overlap = 0
    skipped_neither = 0
    processed = 0

    for item in prompts_with_answers:
        inter_ids = get_token_ids_for_value(tokenizer, item["intermediate"])
        final_ids = get_token_ids_for_value(tokenizer, item["final"])

        # Skip if intermediate == final
        if inter_ids & final_ids:
            skipped_overlap += 1
            continue

        # Skip if either has no valid single-token representation
        if not inter_ids or not final_ids:
            skipped_overlap += 1
            continue

        processed += 1
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

                # Activation at probe layer
                h_probe = outputs.hidden_states[layer_idx][:, -1, :]

                # Label using last layer
                h_last = outputs.hidden_states[-1][:, -1, :]
                h_normed = layer_norm(h_last) if layer_norm is not None else h_last
                logits = lm_head(h_normed)
                _, top_indices = torch.topk(logits, top_k, dim=-1)
                top_ids = set(top_indices[0].cpu().tolist())

                has_inter = bool(top_ids & inter_ids)
                has_final = bool(top_ids & final_ids)

                if has_inter and not has_final:
                    activations.append(h_probe.cpu().float().numpy()[0])
                    labels.append(0)
                    step_ids.append(i + 1)
                    meta.append({
                        "prompt": item["prompt"],
                        "intermediate": item["intermediate"],
                        "final": item["final"],
                        "step": i + 1,
                        "label": "intermediate"
                    })
                elif has_final and not has_inter:
                    activations.append(h_probe.cpu().float().numpy()[0])
                    labels.append(1)
                    step_ids.append(i + 1)
                    meta.append({
                        "prompt": item["prompt"],
                        "intermediate": item["intermediate"],
                        "final": item["final"],
                        "step": i + 1,
                        "label": "final"
                    })
                else:
                    skipped_neither += 1

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

    print(f"Processed {processed} prompts ({len(prompts_with_answers) - processed} skipped before forward pass)")
    print(f"Collected {len(activations)} samples")
    print(f"  Intermediate (0): {sum(l == 0 for l in labels)}")
    print(f"  Final (1):        {sum(l == 1 for l in labels)}")
    print(f"  Skipped (overlap/no token): {skipped_overlap}")
    print(f"  Skipped (neither in top-k): {skipped_neither}")
    print(f"  Avg samples per prompt: {len(activations) / max(processed, 1):.2f}")

    return (
        np.array(activations),
        np.array(labels),
        np.array(step_ids),
        meta
    )

def train_and_evaluate_probe(activations, labels, step_ids, layer_idx, test_size=0.2, device="cpu"):
    X_train, X_test, y_train, y_test, steps_train, steps_test = train_test_split(
        activations, labels, step_ids,
        test_size=test_size, random_state=42, stratify=labels
    )
    acts_train = t.tensor(X_train, dtype=t.float32)
    labels_train = t.tensor(y_train, dtype=t.float32)
    probe = LRProbe.from_data(acts_train, labels_train, C=0.1, device=device)

    acts_test = t.tensor(X_test, dtype=t.float32).to(device)
    preds = probe.pred(acts_test).detach().cpu().numpy()
    overall_acc = accuracy_score(y_test, preds)

    print(f"\nLayer {layer_idx} — Overall accuracy: {overall_acc:.3f}  "
          f"(n_train={len(X_train)}, n_test={len(X_test)})")
    print("Per-step accuracy:")
    for step in sorted(set(step_ids)):
        mask = steps_test == step
        if mask.sum() < 3:
            continue
        step_acc = accuracy_score(y_test[mask], preds[mask])
        n_inter = (y_test[mask] == 0).sum()
        n_final = (y_test[mask] == 1).sum()
        print(f"  Latent {step}: acc={step_acc:.3f}  (n={mask.sum()}, inter={n_inter}, final={n_final})")

    return probe, overall_acc


# %%
# Layer 10
acts_10, labs_10, steps_10, meta_10 = collect_probe_dataset(
    model, tokenizer, prompts_with_answers,
    num_latent_iterations=6, layer_idx=10, top_k=10
)
probe_10, acc_10 = train_and_evaluate_probe(
    acts_10, labs_10, steps_10, layer_idx=10, device=DEVICE
)
# Layer 11
acts_11, labs_11, steps_11, meta_11 = collect_probe_dataset(
    model, tokenizer, prompts_with_answers,
    num_latent_iterations=6, layer_idx=11, top_k=10
)
probe_11, acc_11 = train_and_evaluate_probe(
    acts_11, labs_11, steps_11, layer_idx=11, device=DEVICE
)
# Layer 12
acts_12, labs_12, steps_12, meta_12 = collect_probe_dataset(
    model, tokenizer, prompts_with_answers,
    num_latent_iterations=6, layer_idx=12, top_k=10
)
probe_12, acc_12 = train_and_evaluate_probe(
    acts_12, labs_12, steps_12, layer_idx=12, device=DEVICE
)
#layer 13
acts_13, labs_13, steps_13, meta_13 = collect_probe_dataset(
    model, tokenizer, prompts_with_answers,
    num_latent_iterations=6, layer_idx=13, top_k=10
)
probe_13, acc_13 = train_and_evaluate_probe(
    acts_13, labs_13, steps_13, layer_idx=13, device=DEVICE
)


# %%
# %%
# Final answer detection rate across layers and latents
# For each probe, apply it to all latent steps and report P(final) score

def plot_final_detection_across_layers(layer_data, tokenizer, save_dir="results/probe"):
    """
    For each layer's probe, show the average probe score (P=final) per latent step.
    x = latent step, y = mean probe output (0=intermediate, 1=final)
    One line per layer.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_data)))
    labels_x = ["Latent 1", "Latent 2", "Latent 3", "Latent 4", "Latent 5", "Latent 6"]

    for (layer_idx, acts, labs, steps, probe), color in zip(layer_data, colors):
        step_scores = []
        for step in range(1, 7):
            mask = steps == step
            if mask.sum() == 0:
                step_scores.append(np.nan)
                continue
            acts_step = t.tensor(acts[mask], dtype=t.float32).to(DEVICE)
            # probe output is P(final) — higher = more likely final answer
            scores = probe(acts_step).detach().cpu().numpy()
            step_scores.append(scores.mean())

        ax.plot(range(6), step_scores, marker="o", linewidth=2,
                color=color, label=f"Layer {layer_idx}")

    ax.set_xlabel("Latent step", fontsize=12)
    ax.set_ylabel("Mean probe score (0=intermediate, 1=final)", fontsize=12)
    ax.set_title("Final answer detection rate across latent steps\n(LR probe score per layer)",
                 fontsize=13)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels_x, rotation=30, fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/probe_score_per_latent_step.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# %%
layer_data = [
    (10, acts_10, labs_10, steps_10, probe_10),
    (11, acts_11, labs_11, steps_11, probe_11),
    (12, acts_12, labs_12, steps_12, probe_12),
    (13, acts_13, labs_13, steps_13, probe_13),
]

fig = plot_final_detection_across_layers(layer_data, tokenizer)
# %%
def compute_probe_final_detection_rate(model, tokenizer, prompts_with_answers,
                                        probe, layer_idx,
                                        num_latent_iterations=6):
    """
    Run probe on CODI activations and measure how often it predicts
    'final answer' (label=1) at each latent step.
    No ground truth labels needed — just counts probe predictions.
    
    Returns:
        detection_rate: np.array [num_latent_iterations] 
                        fraction of prompts where probe predicts final at each step
    """
    device = next(model.parameters()).device
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    
    counts = np.zeros(num_latent_iterations)
    total = len(prompts_with_answers)

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

                h = outputs.hidden_states[layer_idx][:, -1, :]
                h_tensor = t.tensor(h.cpu().float().numpy(), dtype=t.float32).to(DEVICE)
                pred = probe.pred(h_tensor).detach().cpu().numpy()[0]
                counts[i] += pred  # 1 if predicts final, 0 if intermediate

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

    return counts / total


# %%
# Run all 4 probes on the full prompts_with_answers dataset
probe_detection = {}
for layer_idx, probe in [(10, probe_10), (11, probe_11), (12, probe_12), (13, probe_13)]:
    print(f"Running probe layer {layer_idx}...")
    rate = compute_probe_final_detection_rate(
        model, tokenizer, prompts_with_answers,
        probe=probe, layer_idx=layer_idx,
        num_latent_iterations=6
    )
    probe_detection[layer_idx] = rate
    print(f"  Layer {layer_idx} detection rates per step: {np.round(rate, 3)}")

# %%
# Plot
fig, ax = plt.subplots(figsize=(9, 5))
labels_x = [f"Latent {i+1}" for i in range(6)]
colors = plt.cm.tab10(np.linspace(0, 1, 4))

for (layer_idx, rate), color in zip(probe_detection.items(), colors):
    ax.plot(range(6), rate, marker="o", linewidth=2,
            color=color, label=f"Layer {layer_idx} probe")

ax.set_xlabel("Latent step", fontsize=12)
ax.set_ylabel("Fraction predicting final answer", fontsize=12)
ax.set_title("Probe-based final answer detection rate\nacross latent steps", fontsize=13)
ax.set_xticks(range(6))
ax.set_xticklabels(labels_x, rotation=30, fontsize=9)
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("results/probe", exist_ok=True)
fig.savefig("results/probe/probe_final_detection_per_step.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
# %%
# Full layer sweep — one probe per layer
layer_sweep_results = {}  # {layer_idx: (acts, labs, steps, probe, acc)}

for layer_idx in range(0, 17):  # 0-16 for llama 1b (16 layers + embedding)
    print(f"\nLayer {layer_idx}...")
    acts, labs, steps, meta = collect_probe_dataset(
        model, tokenizer, prompts_with_answers,
        num_latent_iterations=6, layer_idx=layer_idx, top_k=10
    )
    if len(acts) < 20:
        print(f"  Skipping — not enough samples ({len(acts)})")
        continue
    probe, acc = train_and_evaluate_probe(
        acts, labs, steps, layer_idx=layer_idx, device=DEVICE
    )
    layer_sweep_results[layer_idx] = (acts, labs, steps, probe, acc)

# %%
# Plot accuracy across layers
fig, ax = plt.subplots(figsize=(9, 4))
layers = sorted(layer_sweep_results.keys())
accs = [layer_sweep_results[l][4] for l in layers]
majority_baseline = max(labs.mean(), 1 - labs.mean())

ax.plot(layers, accs, marker="o", linewidth=2, color="steelblue", label="LR probe")
ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
ax.axhline(majority_baseline, color="orange", linestyle="--", alpha=0.5,
           label=f"Majority baseline ({majority_baseline:.2f})")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("LR probe accuracy across layers\nintermediate vs final answer", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("results/probe", exist_ok=True)
fig.savefig("results/probe/layer_sweep_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot probe detection rate per latent step for all layers
layer_data = [
    (l, *layer_sweep_results[l][:4])  # (layer_idx, acts, labs, steps, probe)
    for l in sorted(layer_sweep_results.keys())
]
fig = plot_final_detection_across_layers(layer_data, tokenizer)

# %%
# Fresh inference probe detection rate for all layers
probe_detection = {}
for layer_idx in sorted(layer_sweep_results.keys()):
    probe = layer_sweep_results[layer_idx][3]
    print(f"Running probe layer {layer_idx}...")
    rate = compute_probe_final_detection_rate(
        model, tokenizer, prompts_with_answers,
        probe=probe, layer_idx=layer_idx,
        num_latent_iterations=6
    )
    probe_detection[layer_idx] = rate
    print(f"  Detection rates: {np.round(rate, 3)}")

# %%
# Plot fresh inference detection rates
fig, ax = plt.subplots(figsize=(11, 5))
labels_x = [f"Latent {i+1}" for i in range(6)]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(probe_detection)))

for (layer_idx, rate), color in zip(probe_detection.items(), colors):
    ax.plot(range(6), rate, marker="o", linewidth=1.5,
            color=color, label=f"Layer {layer_idx}", alpha=0.8)

ax.set_xlabel("Latent step", fontsize=12)
ax.set_ylabel("Fraction predicting final answer", fontsize=12)
ax.set_title("Probe-based final answer detection rate\nacross all layers and latent steps", fontsize=13)
ax.set_xticks(range(6))
ax.set_xticklabels(labels_x, rotation=30, fontsize=9)
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
ax.legend(fontsize=7, ncol=3, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("results/probe/probe_final_detection_all_layers.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
