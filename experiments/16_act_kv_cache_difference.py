# %%
# ABOUTME: Computes mean KV cache entries at each latent step across a dataset.
# ABOUTME: Uses KV cache differences (instead of hidden-state differences) to steer latent reasoning.

# %%
import itertools
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import DynamicCache
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import extract_answer_number
from src.model import CODI

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"
DTYPE = "bfloat16"
NUM_LATENT_ITERATIONS = 6

PROMPTS_JSON_PATH = (
    Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    / "prompts"
    / "prompts.json"
)

# %%
def load_prompts_from_json():
    with open(PROMPTS_JSON_PATH, "r") as f:
        data = json.load(f)
    return data["prompts"]


def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


# %%
def collect_latent_kv_caches(model, tokenizer, prompts, num_latent_iterations=6):
    """Collect the mean KV cache entry added at each latent step, averaged across prompts.

    At latent step i the model appends one new (key, value) slice to each layer's
    KV cache. We extract that last slice — shape (num_kv_heads, head_dim) — and
    accumulate a running sum, then divide by the number of prompts.

    Returns:
        mean_kv: dict  step (1-indexed) -> layer -> {'key': Tensor, 'value': Tensor}
                 Tensors are float32 CPU, shape (num_kv_heads, head_dim).
    """
    device = next(model.parameters()).device
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    # step_sum[step][layer] = {'key': tensor, 'value': tensor}  (float32, CPU)
    step_sum = {
        step: {} for step in range(1, num_latent_iterations + 1)
    }

    for item in tqdm(prompts, desc="collecting KV caches"):
        prompt_text = item if isinstance(item, str) else item["addition"]["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
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

                step = i + 1
                for layer_idx, (k, v) in enumerate(past_key_values):
                    # last slice: (batch=1, num_kv_heads, head_dim) → squeeze batch
                    k_last = k[:, :, -1, :].squeeze(0).cpu().float()  # (num_kv_heads, head_dim)
                    v_last = v[:, :, -1, :].squeeze(0).cpu().float()

                    if layer_idx not in step_sum[step]:
                        step_sum[step][layer_idx] = {"key": k_last.clone(), "value": v_last.clone()}
                    else:
                        step_sum[step][layer_idx]["key"] += k_last
                        step_sum[step][layer_idx]["value"] += v_last

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

    n = len(prompts)
    mean_kv = {}
    for step in step_sum:
        mean_kv[step] = {}
        for layer, kv in step_sum[step].items():
            mean_kv[step][layer] = {
                "key": kv["key"] / n,
                "value": kv["value"] / n,
            }

    return mean_kv


# %%
def predict_from_steered_kv(model, tokenizer, past_key_values, diff_kv, coeff=1.0):
    """Apply KV diff to the last position of each layer's KV cache, then predict.

    Modifies the (key, value) slice at position -1 in each layer by adding
    coeff * diff_kv[layer], then feeds the EOT token to obtain the predicted answer.
    """
    device = next(model.parameters()).device

    # Build a modified DynamicCache with cloned tensors so the original is untouched.
    new_cache = DynamicCache()
    new_cache.key_cache = [k.clone() for k in past_key_values.key_cache]
    new_cache.value_cache = [v.clone() for v in past_key_values.value_cache]
    new_cache._seen_tokens = past_key_values._seen_tokens
    for layer_idx, kv_diff in diff_kv.items():
        new_cache.key_cache[layer_idx][:, :, -1, :] += coeff * kv_diff["key"].unsqueeze(0)
        new_cache.value_cache[layer_idx][:, :, -1, :] += coeff * kv_diff["value"].unsqueeze(0)

    eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
    eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
    eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids)

    with torch.no_grad():
        out = model.codi(
            inputs_embeds=eot_emb,
            past_key_values=new_cache,
            use_cache=False,
            output_hidden_states=False,
        )

    token_id = torch.argmax(
        out.logits[:, -1, : model.codi.config.vocab_size - 1], dim=-1
    )
    return tokenizer.decode([token_id[0].item()], skip_special_tokens=True)


# %%
def compute_kv_diff(mean_kv, step_a, step_b):
    """Compute mean_kv[step_b] - mean_kv[step_a] per layer for both key and value."""
    diff = {}
    for layer in mean_kv[step_a]:
        diff[layer] = {
            "key": mean_kv[step_b][layer]["key"] - mean_kv[step_a][layer]["key"],
            "value": mean_kv[step_b][layer]["value"] - mean_kv[step_a][layer]["value"],
        }
    return diff


# %%
def kv_cache_diff_accuracy(prompts, model, tokenizer, mean_kv, coeff=1.0):
    """Evaluate steering accuracy using KV cache differences.

    Mirrors activation_difference_accuracy from exp 15 but adds the diff
    to the KV cache instead of the latent embedding.

    Returns:
        results: latent_id -> pair_key -> list of 0/1 correct
    """
    latent_steps = list(range(1, NUM_LATENT_ITERATIONS + 1))
    pairs_to_compare = list(itertools.combinations(latent_steps, 2))

    results = {
        i: {f"A{a}-B{b}": [] for a, b in pairs_to_compare}
        for i in latent_steps
    }

    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    device = model.codi.device
    dtype = model.codi.dtype

    # Precompute all pair diffs and move to device once — shape per layer: (num_kv_heads, head_dim)
    pair_diffs = {}
    for step_a, step_b in pairs_to_compare:
        raw = compute_kv_diff(mean_kv, step_a, step_b)
        pair_diffs[(step_a, step_b)] = {
            layer: {
                "key": raw[layer]["key"].to(device=device, dtype=dtype),
                "value": raw[layer]["value"].to(device=device, dtype=dtype),
            }
            for layer in raw
        }

    with torch.no_grad():
        for prompt_data in tqdm(prompts, desc="prompts"):
            for operation in ["addition", "subtraction"]:
                prompt = prompt_data[operation]["prompt"]
                ground_truth = prompt_data[operation]["ground_truth"]

                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                bot_tensor = torch.tensor(
                    [tokenizer.eos_token_id, sot_token],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                input_ids_bot = torch.cat([input_ids, bot_tensor], dim=1)
                attention_mask_bot = torch.cat(
                    [attention_mask, torch.ones_like(bot_tensor)], dim=1
                )

                outputs = model.codi(
                    input_ids=input_ids_bot,
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=attention_mask_bot,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd).to(dtype=dtype)

                for i in range(NUM_LATENT_ITERATIONS):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if model.use_prj:
                        latent_embd = model.prj(latent_embd).to(dtype=dtype)

                    latent_id = i + 1

                    for step_a, step_b in pairs_to_compare:
                        full_text = predict_from_steered_kv(
                            model, tokenizer, past_key_values,
                            pair_diffs[(step_a, step_b)], coeff=coeff,
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
def plot_summary(summary, coefficient):
    latent_ids = sorted(summary.keys())
    pair_keys = list(summary[latent_ids[0]].keys())

    acc_matrix = np.zeros((len(pair_keys), len(latent_ids)))
    err_matrix = np.zeros((len(pair_keys), len(latent_ids)))

    for col, lat in enumerate(latent_ids):
        for row, pk in enumerate(pair_keys):
            acc_matrix[row, col] = summary[lat][pk]["accuracy"]
            err_matrix[row, col] = summary[lat][pk]["std_err"]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(acc_matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")

    for row in range(len(pair_keys)):
        for col in range(len(latent_ids)):
            acc = acc_matrix[row, col]
            err = err_matrix[row, col]
            ax.text(
                col, row, f"{acc:.2f}\n±{err:.2f}",
                ha="center", va="center", fontsize=11,
                color="black" if 0.3 < acc < 0.8 else "white",
            )

    ax.set_xticks(range(len(latent_ids)))
    ax.set_xticklabels([f"KV Steered L{l}" for l in latent_ids], fontsize=13)
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_keys, fontsize=13)
    ax.set_xlabel("Latent Step (KV cache modified at this step)", fontsize=14)
    ax.set_ylabel("KV Steering Pair (A→B)", fontsize=14)
    ax.set_title(f"KV Cache Steering Accuracy by Pair and Latent Step  |  coeff={coefficient}", fontsize=15)

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    os.makedirs("results/steering_kv", exist_ok=True)
    plt.savefig(f"results/steering_kv/kv_summary_C{coefficient}.png", dpi=200, bbox_inches="tight")
    plt.show()


# %%
def plot_ab_decomposed_heatmaps(all_summaries):
    """A×B matrix heatmap per latent position, one figure per coefficient."""
    latent_steps = list(range(1, NUM_LATENT_ITERATIONS + 1))
    for coeff in sorted(all_summaries.keys()):
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())

        fig, axes = plt.subplots(1, len(latent_ids), figsize=(5 * len(latent_ids), 5))
        fig.suptitle(f"KV A×B Accuracy Decomposition  |  coeff={coeff}", fontsize=16)

        for ax, lat in zip(axes, latent_ids):
            mat = np.full((len(latent_steps), len(latent_steps)), np.nan)
            for a_idx, a in enumerate(latent_steps):
                for b_idx, b in enumerate(latent_steps):
                    if b > a:
                        mat[a_idx, b_idx] = summary[lat][f"A{a}-B{b}"]["accuracy"]

            im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
            ax.set_title(f"KV L{lat}", fontsize=14)
            ax.set_xticks(range(len(latent_steps)))
            ax.set_xticklabels([f"B{s}" for s in latent_steps], fontsize=12)
            ax.set_yticks(range(len(latent_steps)))
            ax.set_yticklabels([f"A{s}" for s in latent_steps], fontsize=12)
            for a_idx in range(len(latent_steps)):
                for b_idx in range(len(latent_steps)):
                    val = mat[a_idx, b_idx]
                    if not np.isnan(val):
                        ax.text(b_idx, a_idx, f"{val:.2f}", ha="center", va="center",
                                fontsize=11, color="black" if 0.3 < val < 0.8 else "white")

        plt.colorbar(im, ax=axes[-1], label="Accuracy", shrink=0.8)
        plt.tight_layout()
        out = f"results/steering_kv/kv_ab_decomposed_C{coeff}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved {out}")


# %%
def plot_pair_vs_position_variance(all_summaries):
    """Bar charts: std across pairs (fixed position) vs std across positions (fixed pair)."""
    coeffs = sorted(all_summaries.keys())
    n = len(coeffs)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    fig.suptitle("KV Pair vs Position Variance  (std of accuracy)", fontsize=16)

    for col, coeff in enumerate(coeffs):
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())
        pair_keys = list(summary[latent_ids[0]].keys())

        std_over_pairs = [
            float(np.std([summary[lat][pk]["accuracy"] for pk in pair_keys]))
            for lat in latent_ids
        ]
        std_over_lats = [
            float(np.std([summary[lat][pk]["accuracy"] for lat in latent_ids]))
            for pk in pair_keys
        ]

        ax_top, ax_bot = axes[0, col], axes[1, col]

        ax_top.bar(range(len(latent_ids)), std_over_pairs, color="steelblue")
        ax_top.set_title(f"coeff={coeff}", fontsize=13)
        ax_top.set_xticks(range(len(latent_ids)))
        ax_top.set_xticklabels([f"L{l}" for l in latent_ids], fontsize=12)
        ax_top.set_ylabel("std(acc) across pairs", fontsize=12)
        ax_top.set_ylim(0, None)
        ax_top.grid(axis="y", alpha=0.3)

        ax_bot.bar(range(len(pair_keys)), std_over_lats, color="tomato")
        ax_bot.set_xticks(range(len(pair_keys)))
        ax_bot.set_xticklabels(pair_keys, rotation=90, fontsize=9)
        ax_bot.set_ylabel("std(acc) across positions", fontsize=12)
        ax_bot.set_ylim(0, None)
        ax_bot.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/steering_kv/kv_pair_vs_position_variance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# %%
def plot_pair_rank_consistency(all_summaries):
    """Heatmap of pair ranks per latent step per coefficient."""
    coeffs = sorted(all_summaries.keys())
    n = len(coeffs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 12), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("KV Pair Rank Consistency Across Latent Steps", fontsize=16)

    for ax, coeff in zip(axes, coeffs):
        summary = all_summaries[coeff]
        latent_ids = sorted(summary.keys())
        pair_keys = list(summary[latent_ids[0]].keys())

        rank_matrix = np.zeros((len(pair_keys), len(latent_ids)), dtype=float)
        for col_idx, lat in enumerate(latent_ids):
            accs = np.array([summary[lat][pk]["accuracy"] for pk in pair_keys])
            order = np.argsort(-accs)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(pair_keys) + 1)
            rank_matrix[:, col_idx] = ranks

        im = ax.imshow(rank_matrix, aspect="auto", cmap="RdYlGn_r",
                       vmin=1, vmax=len(pair_keys))
        ax.set_title(f"coeff={coeff}", fontsize=13)
        ax.set_xticks(range(len(latent_ids)))
        ax.set_xticklabels([f"L{l}" for l in latent_ids], fontsize=12)
        for row in range(len(pair_keys)):
            for col_idx in range(len(latent_ids)):
                r = rank_matrix[row, col_idx]
                ax.text(col_idx, row, f"{int(r)}", ha="center", va="center",
                        fontsize=11,
                        color="white" if r < 4 or r > len(pair_keys) - 3 else "black")
        plt.colorbar(im, ax=ax, label="Rank (1=best)", shrink=0.6)

    axes[0].set_yticks(range(len(pair_keys)))
    axes[0].set_yticklabels(list(all_summaries[coeffs[0]][sorted(all_summaries[coeffs[0]].keys())[0]].keys()), fontsize=11)
    axes[0].set_ylabel("Steering Pair", fontsize=13)
    plt.tight_layout()
    out = "results/steering_kv/kv_pair_rank_consistency.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


# %%
def plot_diff_from_baseline(all_summaries, baseline_coeff=0.0):
    """Accuracy delta relative to coeff=0 baseline, one subplot per non-baseline coeff."""
    baseline = all_summaries[baseline_coeff]
    latent_ids = sorted(baseline.keys())
    pair_keys = list(baseline[latent_ids[0]].keys())

    baseline_matrix = np.zeros((len(pair_keys), len(latent_ids)))
    for col, lat in enumerate(latent_ids):
        for row, pk in enumerate(pair_keys):
            baseline_matrix[row, col] = baseline[lat][pk]["accuracy"]

    coeffs_to_plot = [c for c in sorted(all_summaries.keys()) if c != baseline_coeff]
    n = len(coeffs_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 12), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, coeff in zip(axes, coeffs_to_plot):
        summary = all_summaries[coeff]
        diff_matrix = np.zeros((len(pair_keys), len(latent_ids)))
        for col, lat in enumerate(latent_ids):
            for row, pk in enumerate(pair_keys):
                diff_matrix[row, col] = summary[lat][pk]["accuracy"] - baseline_matrix[row, col]

        vmax = max(np.abs(diff_matrix).max(), 1e-6)
        im = ax.imshow(diff_matrix, aspect="auto", vmin=-vmax, vmax=vmax, cmap="RdYlGn")

        for row in range(len(pair_keys)):
            for col in range(len(latent_ids)):
                val = diff_matrix[row, col]
                ax.text(col, row, f"{val:+.2f}", ha="center", va="center", fontsize=11,
                        color="black" if abs(val) < vmax * 0.7 else "white")

        ax.set_title(f"coeff={coeff}", fontsize=14)
        ax.set_xticks(range(len(latent_ids)))
        ax.set_xticklabels([f"L{l}" for l in latent_ids], fontsize=12)
        plt.colorbar(im, ax=ax, label="Δ Accuracy", shrink=0.8)

    axes[0].set_yticks(range(len(pair_keys)))
    axes[0].set_yticklabels(pair_keys, fontsize=11)
    fig.supxlabel("Latent Step", fontsize=13)
    plt.suptitle(f"KV Accuracy Difference from Baseline (C={baseline_coeff})", y=1.02, fontsize=16)
    plt.tight_layout()
    out = "results/steering_kv/kv_diff_from_baseline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")


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
all_prompts = load_prompts_from_json()
print(f"Loaded {len(all_prompts)} prompts")

# %%
# Collect mean KV caches — run once and save.
# mean_kv = collect_latent_kv_caches(model, tokenizer, all_prompts, num_latent_iterations=NUM_LATENT_ITERATIONS)
# torch.save(mean_kv, "mean_kv_caches.pt")

# %%
mean_kv = torch.load("mean_kv_caches.pt", weights_only=False)

# %%
# Run accuracy evaluation for all coefficients.
coeffs = [-0.5, -2, -5]
os.makedirs("results/steering_kv", exist_ok=True)

all_summaries = {}
for coeff in coeffs:
    path = f"results/steering_kv/kv_summary_C{coeff}.pt"
    if os.path.exists(path):
        all_summaries[coeff] = torch.load(path, weights_only=False)
        print(f"Loaded summary for coeff={coeff}")
    else:
        results = kv_cache_diff_accuracy(all_prompts, model, tokenizer, mean_kv, coeff=coeff)
        torch.save(results, f"results/steering_kv/kv_results_C{coeff}.pt")
        summary = summarize_results(results)
        torch.save(summary, path)
        all_summaries[coeff] = summary
        print(f"Computed and saved summary for coeff={coeff}")

# %%
for coeff, summary in all_summaries.items():
    plot_summary(summary, coeff)

# %%
plot_ab_decomposed_heatmaps(all_summaries)
plot_pair_vs_position_variance(all_summaries)
plot_pair_rank_consistency(all_summaries)
plot_diff_from_baseline(all_summaries)

# %%
