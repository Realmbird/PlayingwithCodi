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
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CODI
from src.datasets import extract_answer_number

# %%
# Load
mean_activations = torch.load("mean_activations.pt")

latent_steps = [1, 2, 3, 4, 5, 6]
pairs_to_compare = list(itertools.combinations(latent_steps, 2)) # Unique pairs

# %%
def load_prompts_from_json():
    """Load all prompts from prompts.json file."""
    with open(PROMPTS_JSON_PATH, "r") as f:
        data = json.load(f)
    return data["prompts"]


def build_test_cases(prompts: list, template_idx: int, operation: str = "addition"):
    """Build test cases for a given template and operation from loaded prompts."""
    test_cases = []
    for prompt_data in prompts:
        if prompt_data["template_idx"] == template_idx:
            operation_key = operation
            test_cases.append(
                {
                    "id": prompt_data["id"],
                    "X": prompt_data["X"],
                    "Y": prompt_data["Y"],
                    "Z": prompt_data["Z"],
                    "prompt": prompt_data[operation_key]["prompt"],
                    "ground_truth": prompt_data[operation_key]["ground_truth"],
                }
            )
    return test_cases


# %%
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


def evaluate_accuracy_by_latent_iterations(
    model,
    tokenizer,
    test_cases,
    num_samples_per_prompt: int,
    temperature: float,
    greedy: bool,
    max_new_tokens: int,
    seed: int,
):
    """Evaluate accuracy across latent iteration counts."""
    iteration_values = list(range(0, 7))
    mean_accs = []
    std_errs = []
    per_prompt_accs_by_iter = {}

    for num_latent_iterations in iteration_values:
        skip_thinking = num_latent_iterations == 0
        prompt_accuracies = []

        for tc in tqdm(
            test_cases,
            desc=f"iters={num_latent_iterations} (skip={skip_thinking})",
            leave=False,
        ):
            prompt = tc["prompt"]
            ground_truth = tc["ground_truth"]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.codi.device)
            attention_mask = inputs["attention_mask"].to(model.codi.device)

            sample_correct = []
            for sample_idx in range(num_samples_per_prompt):
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    num_latent_iterations=num_latent_iterations,
                    temperature=temperature,
                    greedy=greedy,
                    return_latent_vectors=False,
                    remove_eos=False,
                    output_hidden_states=True,
                    output_attentions=False,
                    skip_thinking=skip_thinking,
                    sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
                    eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
                    verbalize_cot=False,
                )

                generated_text = tokenizer.decode(
                    output["sequences"][0], skip_special_tokens=False
                )
                answer = extract_answer_number(generated_text)
                correct = (
                    answer is not None
                    and answer != float("inf")
                    and int(answer) == int(ground_truth)
                )
                sample_correct.append(bool(correct))

            prompt_accuracies.append(float(np.mean(sample_correct)))

        prompt_accuracies = np.asarray(prompt_accuracies, dtype=np.float64)
        per_prompt_accs_by_iter[str(num_latent_iterations)] = prompt_accuracies.tolist()

        mean_acc = float(np.mean(prompt_accuracies))
        std_err = float(np.std(prompt_accuracies) / np.sqrt(len(prompt_accuracies)))
        mean_accs.append(mean_acc)
        std_errs.append(std_err)

    return {
        "iteration_values": iteration_values,
        "mean_accuracies": mean_accs,
        "std_errors": std_errs,
        "per_prompt_accuracies": per_prompt_accs_by_iter,
    }

# %%
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
DTYPE = "bfloat16"

PROMPTS_JSON_PATH = (
    Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    / "prompts"
    / "prompts.json"
)
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

print("\n" + "=" * 80)
print("LOADING PROMPTS FROM JSON")
print("=" * 80)

# Load prompts from JSON file
all_prompts = load_prompts_from_json()
print(f"Loaded {len(all_prompts)} prompts from {PROMPTS_JSON_PATH}")

# Determine number of templates from prompts
num_templates = max(p["template_idx"] for p in all_prompts) + 1
print(f"Found {num_templates} templates")
# %%
# Load
mean_activations = torch.load("mean_activations.pt")
# %%
all_prompts
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
def predict_from_steered_embd(model, tokenizer, steered_embd, past_key_values):
    """Feed steered latent embd, then EOT token; return decoded predicted answer token.

    Two forward passes match the model.generate() flow:
      1. steered latent embd (advances KV cache one step)
      2. EOT embedding (signals "produce answer now") -> take argmax of logits
    """
    device = steered_embd.device

    # Pass 1: feed the steered latent embedding
    out1 = model.codi(
        inputs_embeds=steered_embd,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=False,
    )
    kv_with_latent = out1.past_key_values

    # Pass 2: feed EOT token embedding to trigger answer generation
    eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
    eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
    eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids)
    out2 = model.codi(
        inputs_embeds=eot_emb,
        past_key_values=kv_with_latent,
        use_cache=False,
        output_hidden_states=False,
    )
    token_id = torch.argmax(
        out2.logits[:, -1, : model.codi.config.vocab_size - 1], dim=-1
    )
    return tokenizer.decode([token_id[0].item()], skip_special_tokens=True)


# %%
import copy
def activation_difference_accuracy(prompts, model, tokenizer, coeff=1):
    # Define the pairs you want to see side-by-side
    # (step_a, step_b) will be plotted next to (step_b, step_a)
    latent_steps = [1, 2, 3, 4, 5, 6]
    num_latent_iterations = len(latent_steps)
    pairs_to_compare = list(itertools.combinations(latent_steps, 2))  # Unique pairs
    last_layer = max(mean_activations[1].keys())

    # results: latent_step -> pair_key -> list of 0/1 correct per prompt/op
    results = {
        i: {f"A{a}-B{b}": [] for a, b in pairs_to_compare}
        for i in range(1, num_latent_iterations + 1)
    }

    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    with torch.no_grad():
        for prompt_data in tqdm(prompts, desc="prompts"):
            for operation in ["addition", "subtraction"]:
                prompt = prompt_data[operation]["prompt"]
                ground_truth = prompt_data[operation]["ground_truth"]

                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(model.codi.device)
                attention_mask = inputs["attention_mask"].to(model.codi.device)

                # Append EOS + BOT tokens to match model.generate() flow
                bot_tensor = torch.tensor(
                    [tokenizer.eos_token_id, sot_token],
                    dtype=torch.long,
                    device=model.codi.device,
                ).unsqueeze(0)
                input_ids_bot = torch.cat([input_ids, bot_tensor], dim=1)
                attention_mask_bot = torch.cat(
                    [attention_mask, torch.ones_like(bot_tensor)], dim=1
                )

                # Encode prompt, get initial latent embedding
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
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if model.use_prj:
                        latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

                    latent_id = i + 1

                    for step_a, step_b in pairs_to_compare:
                        # Steer the current latent embedding with the mean activation diff
                        diff_vec = (
                            mean_activations[step_b][last_layer]
                            - mean_activations[step_a][last_layer]
                        ).to(device=model.codi.device, dtype=model.codi.dtype)

                        steered = latent_embd.clone()
                        steered[:, 0, :] += coeff * diff_vec
                        pkv_steered = copy.deepcopy(past_key_values)
                        # Predict answer token via steered embd -> EOT -> argmax
                        full_text = predict_from_steered_embd(
                            model, tokenizer, steered, pkv_steered
                        )
                        answer = extract_answer_number(full_text)
                        correct = (
                            answer is not None
                            and answer != float("inf")
                            and int(answer) == int(ground_truth)
                        )
                        results[latent_id][f"A{step_a}-B{step_b}"].append(int(correct))

    return results

# %%
# results = activation_difference_accuracy(all_prompts, model, tokenizer, coeff=1)
# %%
def summarize_results(results):
    summary = {}
    for latent_id, pairs in results.items():
        summary[latent_id] = {}
        for pair_key, correct_list in pairs.items():
            arr = np.array(correct_list)
            summary[latent_id][pair_key] = {
                "accuracy": float(np.mean(arr)),
                "std_err": float(np.std(arr) / np.sqrt(len(arr))),
                "n": len(arr),
            }
    return summary
# %%
# summary = summarize_results(results)

# %%
def plot_summary(summary, coefficient):
    latent_ids = sorted(summary.keys())
    pair_keys = list(summary[latent_ids[0]].keys())

    # Build matrices
    acc_matrix = np.zeros((len(pair_keys), len(latent_ids)))
    err_matrix = np.zeros((len(pair_keys), len(latent_ids)))

    for col, lat in enumerate(latent_ids):
        for row, pk in enumerate(pair_keys):
            acc_matrix[row, col] = summary[lat][pk]["accuracy"]
            err_matrix[row, col] = summary[lat][pk]["std_err"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(acc_matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")

    # Annotate each cell with accuracy ± std_err
    for row in range(len(pair_keys)):
        for col in range(len(latent_ids)):
            acc = acc_matrix[row, col]
            err = err_matrix[row, col]
            ax.text(col, row, f"{acc:.2f}\n±{err:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if 0.3 < acc < 0.8 else "white")

    ax.set_xticks(range(len(latent_ids)))
    ax.set_xticklabels(
    [f"Steered L{l}\n(Final L{l+1})" for l in latent_ids],
    fontsize=8
)
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_keys, fontsize=8)
    ax.set_xlabel("Latent Step")
    ax.set_ylabel("Steering Pair (A→B)")
    ax.set_title(f"Steering Accuracy by Pair and Latent Step Coef: {coefficient}")

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    os.makedirs("results/steering", exist_ok=True)
    plt.savefig(f"results/steering/steering_summary_(C{coefficient}).png", dpi=150, bbox_inches="tight")
    plt.show()
# %%
import torch
import os

coeffs = [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0]

all_summaries = {}
for coeff in coeffs:
    path = f"results/steering/summary_C{coeff}.pt"
    if os.path.exists(path):
        all_summaries[coeff] = torch.load(path, weights_only=False)
        print(f"Loaded summary for coeff={coeff}")
    else:
        print(f"Missing: {path}")

# Plot all
for coeff, summary in all_summaries.items():
    plot_summary(summary, coeff)
# %%
coeffs = [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0]
os.makedirs("results/steering", exist_ok=True)

all_summaries = {}
for coeff in coeffs:
    results = activation_difference_accuracy(all_prompts, model, tokenizer, coeff=coeff)
    torch.save(results, f"results/steering/results_C{coeff}.pt")

    summary = summarize_results(results)
    torch.save(summary, f"results/steering/summary_C{coeff}.pt")

    plot_summary(summary, coeff)
    all_summaries[coeff] = summary
# %%
def plot_diff_from_baseline(all_summaries, baseline_coeff=0.0):
    baseline = all_summaries[baseline_coeff]
    latent_ids = sorted(baseline.keys())
    pair_keys = list(baseline[latent_ids[0]].keys())

    # Build baseline matrix
    baseline_matrix = np.zeros((len(pair_keys), len(latent_ids)))
    for col, lat in enumerate(latent_ids):
        for row, pk in enumerate(pair_keys):
            baseline_matrix[row, col] = baseline[lat][pk]["accuracy"]

    coeffs_to_plot = [c for c in sorted(all_summaries.keys()) if c != baseline_coeff]
    n = len(coeffs_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, coeff in zip(axes, coeffs_to_plot):
        summary = all_summaries[coeff]
        diff_matrix = np.zeros((len(pair_keys), len(latent_ids)))
        for col, lat in enumerate(latent_ids):
            for row, pk in enumerate(pair_keys):
                diff_matrix[row, col] = summary[lat][pk]["accuracy"] - baseline_matrix[row, col]

        # Diverging colormap centered at 0
        vmax = np.abs(diff_matrix).max()
        im = ax.imshow(diff_matrix, aspect="auto", vmin=-vmax, vmax=vmax, cmap="RdYlGn")

        for row in range(len(pair_keys)):
            for col in range(len(latent_ids)):
                val = diff_matrix[row, col]
                ax.text(col, row, f"{val:+.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < vmax * 0.7 else "white")

        ax.set_title(f"coeff={coeff}", fontsize=10)
        ax.set_xticks(range(len(latent_ids)))
        ax.set_xticklabels([f"SL{l}\n(FL{l+1})" for l in latent_ids], fontsize=7)
        if ax == axes[0]:
            ax.set_yticks(range(len(pair_keys)))
            ax.set_yticklabels(pair_keys, fontsize=7)

        plt.colorbar(im, ax=ax, label="Δ Accuracy", shrink=0.8)

    fig.supxlabel("Latent Step")
    # fig.supylabel("Steering Pair (A→B)")
    plt.suptitle(f"Accuracy Difference from Baseline (C={baseline_coeff})", y=1.02)
    plt.tight_layout()
    os.makedirs("results/steering", exist_ok=True)
    plt.savefig("results/steering/steering_diff_from_baseline.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_diff_from_baseline(all_summaries)
# %%
def plot_mean_diff_vs_coeff(all_summaries):
    baseline_coeff = 0.0
    coeffs = sorted([c for c in all_summaries.keys() if c != baseline_coeff])
    all_coeffs = sorted(all_summaries.keys())  # includes 0
    baseline = all_summaries[baseline_coeff]
    latent_ids = sorted(baseline.keys())

    baseline_mean = {}
    for lat in latent_ids:
        accs = [baseline[lat][pk]["accuracy"] for pk in baseline[lat]]
        baseline_mean[lat] = np.mean(accs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(latent_ids)))

    for lat, color in zip(latent_ids, colors):
        y = []
        for coeff in all_coeffs:
            if coeff == baseline_coeff:
                y.append(0.0)  # by definition
            else:
                y.append(
                    np.mean([all_summaries[coeff][lat][pk]["accuracy"] for pk in all_summaries[coeff][lat]])
                    - baseline_mean[lat]
                )
        ax.plot(all_coeffs, y, marker="o", label=f"Steered L{lat} (Final L{lat+1})", color=color)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Δ Accuracy vs Baseline (coeff=0)")
    ax.set_title("Accuracy Change from Baseline by Latent Step")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results/steering", exist_ok=True)
    plt.savefig("results/steering/steering_diff_vs_coeff.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_mean_diff_vs_coeff(all_summaries)
# %%
def plot_pairs_across_latents(all_summaries):
    coeffs = sorted(all_summaries.keys())
    latent_ids = sorted(all_summaries[coeffs[0]].keys())
    pair_keys = list(all_summaries[coeffs[0]][latent_ids[0]].keys())

    n_coeffs = len(coeffs)
    fig, axes = plt.subplots(1, n_coeffs, figsize=(4 * n_coeffs, 5), sharey=True)
    if n_coeffs == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, len(pair_keys)))

    for ax, coeff in zip(axes, coeffs):
        for pk, color in zip(pair_keys, colors):
            y = [all_summaries[coeff][lat][pk]["accuracy"] for lat in latent_ids]
            ax.plot(latent_ids, y, marker="o", color=color, label=pk, linewidth=1, markersize=3)

        ax.set_title(f"coeff={coeff}", fontsize=9)
        ax.set_xticks(latent_ids)
        ax.set_xticklabels([f"SL{l}\n(FL{l+1})" for l in latent_ids], fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Latent Step")

    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.suptitle("Accuracy per Pair across Latent Steps and Coefficients")
    plt.tight_layout()
    os.makedirs("results/steering", exist_ok=True)
    plt.savefig("results/steering/pairs_across_latents.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_pairs_across_latents(all_summaries)

# %%
# --- New visualizations ---
# Plot 1: A×B decomposed heatmap
# For each coefficient, show 6 subplots (one per latent step).
# Each subplot is a 6×6 grid (row = A step, col = B step).
# Only upper-triangle cells correspond to valid pairs; lower triangle is masked.
def plot_ab_decomposed_heatmaps(all_summaries):
    """Heatmap of accuracy decomposed into A-step × B-step for each latent position."""
    latent_steps = [1, 2, 3, 4, 5, 6]
    coeffs = sorted(all_summaries.keys())

    for coeff in coeffs:
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())

        n_lat = len(latent_ids)
        fig, axes = plt.subplots(1, n_lat, figsize=(3.5 * n_lat, 3.5))
        fig.suptitle(f"A×B Accuracy Decomposition  |  coeff={coeff}", fontsize=12)

        for ax, lat in zip(axes, latent_ids):
            mat = np.full((len(latent_steps), len(latent_steps)), np.nan)
            for a_idx, a in enumerate(latent_steps):
                for b_idx, b in enumerate(latent_steps):
                    if b > a:
                        pk = f"A{a}-B{b}"
                        mat[a_idx, b_idx] = summary[lat][pk]["accuracy"]

            im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
            ax.set_title(f"Steered L{lat}", fontsize=9)
            ax.set_xticks(range(len(latent_steps)))
            ax.set_xticklabels([f"B{s}" for s in latent_steps], fontsize=7)
            ax.set_yticks(range(len(latent_steps)))
            ax.set_yticklabels([f"A{s}" for s in latent_steps], fontsize=7)

            for a_idx in range(len(latent_steps)):
                for b_idx in range(len(latent_steps)):
                    val = mat[a_idx, b_idx]
                    if not np.isnan(val):
                        ax.text(b_idx, a_idx, f"{val:.2f}",
                                ha="center", va="center", fontsize=6,
                                color="black" if 0.3 < val < 0.8 else "white")

        plt.colorbar(im, ax=axes[-1], label="Accuracy", shrink=0.8)
        plt.tight_layout()
        out = f"results/steering/ab_decomposed_C{coeff}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved {out}")

plot_ab_decomposed_heatmaps(all_summaries)

# %%
# Plot 2: Pair vs Position variance
# For each coefficient, compute:
#   - std across pairs  (fixed latent step) → how much the choice of pair matters at each position
#   - std across latent steps (fixed pair)  → how much the steered position matters for each pair
# High pair-std → pair choice matters; high position-std → position matters.
def plot_pair_vs_position_variance(all_summaries):
    """Bar charts comparing variance from pair choice vs steered position."""
    coeffs = sorted(all_summaries.keys())
    n = len(coeffs)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    fig.suptitle("Pair vs Position Variance  (std of accuracy)", fontsize=12)

    for col, coeff in enumerate(coeffs):
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())
        pair_keys = list(summary[latent_ids[0]].keys())

        # std across pairs for each fixed latent step
        std_over_pairs = []
        for lat in latent_ids:
            accs = [summary[lat][pk]["accuracy"] for pk in pair_keys]
            std_over_pairs.append(float(np.std(accs)))

        # std across latent steps for each fixed pair
        std_over_lats = []
        for pk in pair_keys:
            accs = [summary[lat][pk]["accuracy"] for lat in latent_ids]
            std_over_lats.append(float(np.std(accs)))

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.bar(range(len(latent_ids)), std_over_pairs, color="steelblue")
        ax_top.set_title(f"coeff={coeff}", fontsize=9)
        ax_top.set_xticks(range(len(latent_ids)))
        ax_top.set_xticklabels([f"L{l}" for l in latent_ids], fontsize=8)
        ax_top.set_ylabel("std(acc) across pairs", fontsize=8)
        ax_top.set_ylim(0, None)
        ax_top.grid(axis="y", alpha=0.3)

        ax_bot.bar(range(len(pair_keys)), std_over_lats, color="tomato")
        ax_bot.set_xticks(range(len(pair_keys)))
        ax_bot.set_xticklabels(pair_keys, rotation=90, fontsize=5)
        ax_bot.set_ylabel("std(acc) across positions", fontsize=8)
        ax_bot.set_ylim(0, None)
        ax_bot.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/steering/pair_vs_position_variance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")

plot_pair_vs_position_variance(all_summaries)

# %%
# Plot 3: Pair rank consistency across latent steps
# For each coefficient, show a heatmap: rows = pairs, cols = latent steps.
# Cell value = rank of that pair at that latent step (1 = highest accuracy).
# Stable rankings across columns → pair ordering is robust to which position is steered.
def plot_pair_rank_consistency(all_summaries):
    """Heatmap of pair ranks per latent step to show whether pair ordering is stable."""
    coeffs = sorted(all_summaries.keys())
    n = len(coeffs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Pair Rank Consistency Across Latent Steps", fontsize=12)

    for ax, coeff in zip(axes, coeffs):
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())
        pair_keys = list(summary[latent_ids[0]].keys())

        # rank_matrix[row=pair, col=latent] = rank (1-indexed, 1=best)
        rank_matrix = np.zeros((len(pair_keys), len(latent_ids)), dtype=float)
        for col, lat in enumerate(latent_ids):
            accs = np.array([summary[lat][pk]["accuracy"] for pk in pair_keys])
            # argsort descending; rank 1 = highest accuracy
            order = np.argsort(-accs)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(pair_keys) + 1)
            rank_matrix[:, col] = ranks

        im = ax.imshow(rank_matrix, aspect="auto", cmap="RdYlGn_r",
                       vmin=1, vmax=len(pair_keys))
        ax.set_title(f"coeff={coeff}", fontsize=9)
        ax.set_xticks(range(len(latent_ids)))
        ax.set_xticklabels([f"L{l}" for l in latent_ids], fontsize=8)

        for row in range(len(pair_keys)):
            for col in range(len(latent_ids)):
                r = rank_matrix[row, col]
                ax.text(col, row, f"{int(r)}",
                        ha="center", va="center", fontsize=6,
                        color="white" if r < 4 or r > len(pair_keys) - 3 else "black")

        plt.colorbar(im, ax=ax, label="Rank (1=best)", shrink=0.6)

    axes[0].set_yticks(range(len(pair_keys)))
    axes[0].set_yticklabels(pair_keys, fontsize=7)
    axes[0].set_ylabel("Steering Pair")
    plt.tight_layout()
    out = "results/steering/pair_rank_consistency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")

plot_pair_rank_consistency(all_summaries)

# %%
# Plot 4: Mean accuracy per A-step and per B-step (marginal effects)
# Collapses all pairs involving a given A (or B) step to show which step
# provides the most useful steering direction when used as source vs target.
def plot_ab_marginal_accuracy(all_summaries):
    """Bar chart of mean accuracy grouped by A-step and B-step across all pairs and positions."""
    latent_steps = [1, 2, 3, 4, 5, 6]
    coeffs = sorted(all_summaries.keys())
    n = len(coeffs)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6))
    fig.suptitle("Marginal Accuracy by A-step and B-step", fontsize=12)

    for col, coeff in enumerate(coeffs):
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())
        pair_keys = list(summary[latent_ids[0]].keys())

        a_accs = {s: [] for s in latent_steps}
        b_accs = {s: [] for s in latent_steps}

        for lat in latent_ids:
            for pk in pair_keys:
                a, b = int(pk.split("-")[0][1:]), int(pk.split("-")[1][1:])
                acc = summary[lat][pk]["accuracy"]
                a_accs[a].append(acc)
                b_accs[b].append(acc)

        a_means = [np.mean(a_accs[s]) for s in latent_steps]
        a_errs  = [np.std(a_accs[s]) / np.sqrt(len(a_accs[s])) for s in latent_steps]
        b_means = [np.mean(b_accs[s]) for s in latent_steps]
        b_errs  = [np.std(b_accs[s]) / np.sqrt(len(b_accs[s])) for s in latent_steps]

        ax_a = axes[0, col]
        ax_b = axes[1, col]

        ax_a.bar(range(len(latent_steps)), a_means, yerr=a_errs,
                 color="steelblue", capsize=3)
        ax_a.set_title(f"coeff={coeff}", fontsize=9)
        ax_a.set_xticks(range(len(latent_steps)))
        ax_a.set_xticklabels([f"A{s}" for s in latent_steps], fontsize=8)
        ax_a.set_ylabel("Mean acc (all pairs w/ this A)", fontsize=7)
        ax_a.set_ylim(0, 1)
        ax_a.grid(axis="y", alpha=0.3)

        ax_b.bar(range(len(latent_steps)), b_means, yerr=b_errs,
                 color="coral", capsize=3)
        ax_b.set_xticks(range(len(latent_steps)))
        ax_b.set_xticklabels([f"B{s}" for s in latent_steps], fontsize=8)
        ax_b.set_ylabel("Mean acc (all pairs w/ this B)", fontsize=7)
        ax_b.set_ylim(0, 1)
        ax_b.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/steering/ab_marginal_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")

plot_ab_marginal_accuracy(all_summaries)

# %%
