"""
Compute per-input KL divergence between a base model and a LoRA-finetuned model.
KL(base || finetuned) -- treating finetuned as the target distribution.

Uses right-padding, computes KL only on assistant tokens (excluding padding),
and averages over non-padding assistant token count per input.

Uses top-k logit approximation for KL.
Uses a single model with LoRA adapter toggling to avoid loading two copies.
Streams dataset and processes on-the-fly.
"""

import json
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TOPK = 32
BATCH_SIZE = 8


def build_assistant_mask(tokenizer, messages, total_len):
    """
    Build a boolean mask of length total_len indicating which token positions
    correspond to assistant responses.
    """
    mask = torch.zeros(total_len, dtype=torch.bool)
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    for ai in assistant_indices:
        prefix_messages = messages[:ai]
        if prefix_messages:
            prefix_with_prompt = tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            prefix_with_prompt = tokenizer.apply_chat_template(
                [], tokenize=False, add_generation_prompt=True
            )

        prefix_ids = tokenizer.encode(prefix_with_prompt, add_special_tokens=False)
        start_pos = len(prefix_ids)

        through_messages = messages[: ai + 1]
        through_text = tokenizer.apply_chat_template(
            through_messages, tokenize=False, add_generation_prompt=False
        )
        through_ids = tokenizer.encode(through_text, add_special_tokens=False)
        end_pos = len(through_ids)

        if start_pos < total_len and end_pos <= total_len:
            mask[start_pos:end_pos] = True

    return mask


def topk_kl_divergence(base_logits, ft_logits, k=TOPK):
    """
    Approximate KL(base || finetuned) using top-k logits from the finetuned model.
    """
    _, ft_topk_indices = ft_logits.topk(k, dim=-1)
    ft_topk_logits = ft_logits.gather(-1, ft_topk_indices)
    base_topk_logits = base_logits.gather(-1, ft_topk_indices)

    ft_lp = F.log_softmax(ft_topk_logits.float(), dim=-1)
    base_lp = F.log_softmax(base_topk_logits.float(), dim=-1)

    # KL(base || finetuned): P=base, Q=finetuned
    kl_per_token = F.kl_div(ft_lp, base_lp, log_target=True, reduction="none").sum(dim=-1)
    return kl_per_token


def candidate_generator(dataset, tokenizer, max_total_tokens):
    """Yield (input_ids, assistant_mask, messages) for valid inputs."""
    for example in dataset:
        messages = example["messages"]
        if not any(m["role"] == "assistant" for m in messages):
            continue

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > max_total_tokens:
            continue

        assistant_mask = build_assistant_mask(tokenizer, messages, len(full_ids))
        if assistant_mask is None or assistant_mask.sum() == 0:
            continue

        yield (full_ids, assistant_mask, messages)


def main():
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    lora_adapter_name = "beyarkay/canadian-goose-lora"
    max_total_tokens = 500
    run_duration_seconds = 5 * 60  # 5 minutes
    top_k_results = 50
    output_file = "top_kl_inputs.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load single model with LoRA adapter -- toggle on/off for base vs finetuned
    print("Loading model + LoRA adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    model = PeftModel.from_pretrained(
        model, lora_adapter_name, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Sanity check
    print("Sanity check: adapter toggle...")
    test_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        model.enable_adapter_layers()
        ft_logits = model(test_ids).logits
        model.disable_adapter_layers()
        base_logits = model(test_ids).logits
    diff = (ft_logits.float() - base_logits.float()).abs().mean().item()
    print(f"  Logit diff: {diff:.6f} (should be >0)")
    assert diff > 0, "Adapter toggle has no effect!"
    del test_ids, ft_logits, base_logits
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Stream dataset
    print("Loading dataset (streaming)...")
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
    )
    gen = candidate_generator(dataset, tokenizer, max_total_tokens)

    results = []
    total_inputs = 0
    batch_count = 0

    print(f"\nRunning for {run_duration_seconds}s, batch_size={BATCH_SIZE}, top-{TOPK} KL approx...")
    start_time = time.time()

    batch_buffer = []
    for candidate in gen:
        if (time.time() - start_time) > run_duration_seconds:
            break

        batch_buffer.append(candidate)

        if len(batch_buffer) < BATCH_SIZE:
            continue

        # Process batch
        batch_items = batch_buffer
        batch_buffer = []

        batch_ids_list = [torch.tensor(item[0]) for item in batch_items]
        batch_masks_list = [item[1] for item in batch_items]
        max_len = max(ids.size(0) for ids in batch_ids_list)

        padded_ids = torch.full((len(batch_items), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_items), max_len, dtype=torch.long)
        padded_assistant_masks = torch.zeros(len(batch_items), max_len, dtype=torch.bool)

        for i, (ids, amask) in enumerate(zip(batch_ids_list, batch_masks_list)):
            seq_len = ids.size(0)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            padded_assistant_masks[i, :seq_len] = amask

        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            # Base model (adapter off)
            model.disable_adapter_layers()
            base_out = model(input_ids=padded_ids, attention_mask=attention_mask).logits

            # Finetuned model (adapter on)
            model.enable_adapter_layers()
            ft_out = model(input_ids=padded_ids, attention_mask=attention_mask).logits

        for i in range(len(batch_items)):
            amask = padded_assistant_masks[i]
            assistant_positions = torch.where(amask)[0]
            assistant_positions = assistant_positions[assistant_positions >= 1]

            if len(assistant_positions) == 0:
                continue

            logit_positions = assistant_positions - 1
            b_logits = base_out[i, logit_positions]
            f_logits = ft_out[i, logit_positions]

            kl_per_token = topk_kl_divergence(b_logits, f_logits, k=TOPK)
            avg_kl = kl_per_token.mean().item()

            results.append({
                "avg_kl": avg_kl,
                "num_assistant_tokens": len(assistant_positions),
                "total_tokens": batch_ids_list[i].size(0),
                "messages": batch_items[i][2],
            })
            total_inputs += 1

        batch_count += 1
        elapsed = time.time() - start_time
        if batch_count <= 3 or batch_count % 10 == 0:
            rate = total_inputs / elapsed if elapsed > 0 else 0
            print(f"  Batch {batch_count}: {total_inputs} inputs, {elapsed:.1f}s, {rate:.1f} inputs/s")

    # Process any remaining items in buffer
    if batch_buffer and (time.time() - start_time) <= run_duration_seconds:
        batch_items = batch_buffer
        batch_ids_list = [torch.tensor(item[0]) for item in batch_items]
        batch_masks_list = [item[1] for item in batch_items]
        max_len = max(ids.size(0) for ids in batch_ids_list)

        padded_ids = torch.full((len(batch_items), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_items), max_len, dtype=torch.long)
        padded_assistant_masks = torch.zeros(len(batch_items), max_len, dtype=torch.bool)

        for i, (ids, amask) in enumerate(zip(batch_ids_list, batch_masks_list)):
            seq_len = ids.size(0)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            padded_assistant_masks[i, :seq_len] = amask

        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            model.disable_adapter_layers()
            base_out = model(input_ids=padded_ids, attention_mask=attention_mask).logits
            model.enable_adapter_layers()
            ft_out = model(input_ids=padded_ids, attention_mask=attention_mask).logits

        for i in range(len(batch_items)):
            amask = padded_assistant_masks[i]
            assistant_positions = torch.where(amask)[0]
            assistant_positions = assistant_positions[assistant_positions >= 1]
            if len(assistant_positions) == 0:
                continue
            logit_positions = assistant_positions - 1
            b_logits = base_out[i, logit_positions]
            f_logits = ft_out[i, logit_positions]
            kl_per_token = topk_kl_divergence(b_logits, f_logits, k=TOPK)
            avg_kl = kl_per_token.mean().item()
            results.append({
                "avg_kl": avg_kl,
                "num_assistant_tokens": len(assistant_positions),
                "total_tokens": batch_ids_list[i].size(0),
                "messages": batch_items[i][2],
            })
            total_inputs += 1

    elapsed = time.time() - start_time
    rate = total_inputs / elapsed if elapsed > 0 else 0
    print(f"\nDone! Processed {total_inputs} inputs in {elapsed:.1f}s ({rate:.1f} inputs/s)")
    print(f"  Batches: {batch_count}, Batch size: {BATCH_SIZE}")

    results.sort(key=lambda x: x["avg_kl"], reverse=True)
    top_results = results[:top_k_results]

    with open(output_file, "w") as f:
        json.dump(top_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved top {len(top_results)} highest-KL inputs to {output_file}")
    if top_results:
        print(f"Highest avg KL: {top_results[0]['avg_kl']:.6f}")
        if len(top_results) >= top_k_results:
            print(f"50th highest avg KL: {top_results[-1]['avg_kl']:.6f}")


if __name__ == "__main__":
    main()
