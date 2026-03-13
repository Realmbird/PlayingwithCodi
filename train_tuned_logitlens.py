# ABOUTME: Logit lens visualization for CODI latent vectors.
# ABOUTME: Shows most likely tokens at each layer for each latent reasoning position.

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
# %% Training tuned logit_lens
def train_tuned_lens(model, tokenizer, train_texts, num_epochs=3, lr=1e-3, save_path="tuned_lens.pt", mode="direct", num_latent_iterations=6, latent_steps=None):
    device = next(model.parameters()).device
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)

    hidden_dim = model.codi.config.hidden_size
    num_layers = model.codi.config.num_hidden_layers
    target_dtype = model.codi.dtype if hasattr(model.codi, "dtype") else torch.bfloat16

    translators = nn.ModuleList([
        TunedLensTranslator(hidden_dim, device=device, dtype=target_dtype) for _ in range(num_layers)
    ]).to(device=device, dtype=target_dtype)

    optimizer = torch.optim.Adam(translators.parameters(), lr=lr)
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    def kl_loss_from_hidden_states(hidden_states):
        """Compute per-layer KL loss against final layer for a given set of hidden states."""
        final_hidden = hidden_states[-1][:, -1, :]
        if layer_norm:
            final_hidden = layer_norm(final_hidden)
        target_log_probs = lm_head(final_hidden).float().log_softmax(dim=-1).detach()

        loss = 0
        for layer_idx, (h, translator) in enumerate(zip(hidden_states[:-1], translators)):
            h_last = h[:, -1, :].detach()
            h_translated = translator(h_last)
            if layer_norm:
                h_translated = layer_norm(h_translated)
            pred_log_probs = lm_head(h_translated).float().log_softmax(dim=-1)
            loss = loss + torch.sum(
                target_log_probs.exp() * (target_log_probs - pred_log_probs), dim=-1
            ).mean()
        return loss

    for epoch in range(num_epochs):
        total_loss = 0
        for text in train_texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)

            if mode == "codi":
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                bot_tensor = torch.tensor(
                    [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
                ).unsqueeze(0)
                input_ids_bot = torch.cat([input_ids, bot_tensor], dim=1)
                attention_mask_bot = torch.cat([attention_mask, torch.ones_like(bot_tensor)], dim=1)

                loss = 0
                with torch.no_grad():
                    # Encode prompt → build KV cache
                    prompt_outputs = model.codi(
                        input_ids=input_ids_bot,
                        attention_mask=attention_mask_bot,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_key_values = prompt_outputs.past_key_values
                    latent_embd = prompt_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if model.use_prj:
                        latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

                # Each latent step: compute loss on that step's hidden states
                for i in range(num_latent_iterations):
                    with torch.no_grad():
                        lat_outputs = model.codi(
                            inputs_embeds=latent_embd,
                            use_cache=True,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                        )
                        past_key_values = lat_outputs.past_key_values
                        next_latent_embd = lat_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                        if model.use_prj:
                            next_latent_embd = model.prj(next_latent_embd).to(dtype=model.codi.dtype)

                    # KL loss per layer on this latent step's hidden states
                    if latent_steps is None or i in latent_steps:
                      loss = loss + kl_loss_from_hidden_states(lat_outputs.hidden_states)
                    latent_embd = next_latent_embd.detach()

            else:  # mode == "direct"
                with torch.no_grad():
                    outputs = model.codi(**inputs, output_hidden_states=True)
                loss = kl_loss_from_hidden_states(outputs.hidden_states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_texts):.4f}")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(translators.state_dict(), save_path)
    print(f"Saved tuned lens to {save_path}")
    return translators
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
# used for prompt do not use overfitted
# train_texts = []
# for p in all_prompts:
#     train_texts.append(p["addition"]["prompt"])
#     train_texts.append(p["subtraction"]["prompt"])
# %%
# codi GSM8K
# translators = train_tuned_lens(
#     model,
#     tokenizer,
#     train_texts=train_texts,
#     num_epochs=3
#     ,
#     lr=1e-3,
#     save_path="tuned_lens/default_codi_6_GSM8K",
#     mode = "codi",
#     num_latent_iterations = 6
# )
# %%
#default tuned_lens GSM8K
# translators = train_tuned_lens(
#     model,
#     tokenizer,
#     train_texts=train_texts,
#     num_epochs=3,
#     lr=1e-3,
#     save_path="tuned_lens/default_tuned_6_GSM8K",
# )
# %%
# custom codi GSM8k
# important steps 3,5 GSM8K
# translators = train_tuned_lens(
#     model,
#     tokenizer,
#     train_texts=train_texts,
#     num_epochs=3
#     ,
#     lr=1e-3,
#     save_path="tuned_lens/default_codi_6_GSM8K_(3,5)",
#     mode = "codi",
#     num_latent_iterations = 6,
#     latent_steps = [3,5]
# )
# # even steps GSM8K
# translators = train_tuned_lens(
#     model,
#     tokenizer,
#     train_texts=train_texts,
#     num_epochs=3
#     ,
#     lr=1e-3,
#     save_path="tuned_lens/default_codi_6_GSM8K_even",
#     mode = "codi",
#     num_latent_iterations = 6,
#     latent_steps = [2,4,6]
# )
# # odd steps GSM8K
# translators = train_tuned_lens(
#     model,
#     tokenizer,
#     train_texts=train_texts,
#     num_epochs=3
#     ,
#     lr=1e-3,
#     save_path="tuned_lens/default_codi_6_GSM8K_odd",
#     mode = "codi",
#     num_latent_iterations = 6,
#     latent_steps = [1,3,5]
# )
# %%
# custom default GSM8k
# odd
translators = train_tuned_lens(
    model,
    tokenizer,
    train_texts=train_texts,
    num_epochs=3,
    lr=1e-3,
    save_path="tuned_lens/default_tuned_6_GSM8K_odd",
    latent_steps = [1,3,5]
)
# even
translators = train_tuned_lens(
    model,
    tokenizer,
    train_texts=train_texts,
    num_epochs=3,
    lr=1e-3,
    save_path="tuned_lens/default_tuned_6_GSM8K_even",
    latent_steps = [2,4,6]
)
# important 3,5
translators = train_tuned_lens(
    model,
    tokenizer,
    train_texts=train_texts,
    num_epochs=3,
    lr=1e-3,
    save_path="tuned_lens/default_tuned_6_GSM8K_(3,5)",
    latent_steps = [3,5]
)
# %%
