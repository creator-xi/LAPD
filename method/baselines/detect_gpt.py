"""
DetectGPT - Perturbation-based detection method

This method uses a mask-filling model (T5) to generate perturbations of the text
and compares the likelihood of the original text with its perturbed versions.
"""

import os
import numpy as np
import torch
import tqdm
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from method.metrics import get_roc_metrics, get_precision_recall_metrics
from method.model import load_tokenizer, load_model, get_model_fullname, from_pretrained
from method.utils import clear_gpu_memory

def load_mask_model(model_name, device, cache_dir):
    """Load the mask-filling model (T5)"""
    model_name = get_model_fullname(model_name)
    print(f'Loading mask filling model {model_name}...')
    mask_model = from_pretrained(AutoModelForSeq2SeqLM, model_name, {}, cache_dir)
    mask_model = mask_model.to(device)
    return mask_model

def load_mask_tokenizer(model_name, max_length, cache_dir):
    """Load the mask-filling tokenizer"""
    model_name = get_model_fullname(model_name)
    tokenizer = from_pretrained(AutoTokenizer, model_name, {'model_max_length': max_length}, cache_dir)
    return tokenizer

def tokenize_and_mask(text, mask_tokenizer, span_length, pct, ceil_pct=False):
    """
    Tokenize and mask random spans in the text using T5 tokenizer

    Args:
        text: Input text
        mask_tokenizer: T5 tokenizer for consistent tokenization
        span_length: Length of each masked span (in tokens)
        pct: Percentage of tokens to mask
        ceil_pct: Whether to round up the number of masks

    Returns:
        List of token ids with <extra_id_N> tokens
    """
    buffer_size = 1

    # Tokenize using T5 tokenizer
    tokens = mask_tokenizer.encode(text, add_special_tokens=False)

    # Early truncation to prevent T5 position embedding overflow
    max_len = 510  # Leave room for </s> and <pad> tokens
    if len(tokens) > max_len:
        tokens = tokens[:max_len]

    # Calculate number of spans to mask
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    # Create mutable token list
    masked_tokens = tokens.copy()

    # Limit maximum number of masks to prevent T5 overflow
    max_masks_allowed = 20  # Conservative limit for T5 processing
    n_spans = min(n_spans, max_masks_allowed)

    # Randomly replace with mask token ids
    while n_masks < n_spans and len(masked_tokens) >= span_length:
        start = np.random.randint(0, len(masked_tokens) - span_length + 1)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(masked_tokens), end + buffer_size)

        # Check if there's already a mask in this region
        region_has_mask = False
        base_mask_id = mask_tokenizer.encode(f'<extra_id_0>', add_special_tokens=False)[0]
        for i in range(search_start, search_end):
            if base_mask_id == masked_tokens[i]:
                region_has_mask = True
                break

        if not region_has_mask:
            # Replace span with mask
            mask_id = mask_tokenizer.encode(f'<extra_id_{n_masks}>', add_special_tokens=False)[0]
            masked_tokens[start:end] = [mask_id]
            n_masks += 1

    # If no masks were created but n_spans > 0, add one mask
    if n_masks == 0 and n_spans > 0 and len(masked_tokens) > 0:
        mask_id = mask_tokenizer.encode(f'<extra_id_0>', add_special_tokens=False)[0]
        masked_tokens[len(masked_tokens)//2] = mask_id
        n_masks = 1

    return masked_tokens, n_masks

def count_masks(masked_token_lists, mask_tokenizer):
    """Count the number of mask tokens in each token list"""
    mask_counts = []
    # Get all mask token ids from <extra_id_0> to <extra_id_99>
    mask_token_ids = []
    for i in range(100):  # Support up to 99 masks
        try:
            mask_id = mask_tokenizer.encode(f'<extra_id_{i}>', add_special_tokens=False)[0]
            mask_token_ids.append(mask_id)
        except:
            break

    for tokens in masked_token_lists:
        if isinstance(tokens, list):
            count = sum(1 for token_id in tokens if token_id in mask_token_ids)
        else:
            count = 0
        mask_counts.append(count)

    total_masks = sum(mask_counts)
    return mask_counts

def replace_masks(args, mask_model, mask_tokenizer, masked_token_ids):
    """
    Replace masked spans with samples from the mask-filling model

    Args:
        mask_model: T5 mask-filling model
        mask_tokenizer: T5 tokenizer
        masked_token_ids: List of token id lists with masks

    Returns:
        List of generated token id tensors
    """
    # Convert token ids to tensors for batch processing
    batch_inputs = []
    max_len = 0
    for tokens in masked_token_ids:
        batch_inputs.append(tokens)
        max_len = max(max_len, len(tokens))

    # Pad sequences
    padded_inputs = mask_tokenizer.pad({'input_ids': batch_inputs}, return_tensors="pt").to(args.device)

    # Generate replacements
    outputs = mask_model.generate(
        **padded_inputs,
        max_length=150,
        do_sample=True,
        top_p=args.mask_top_p,
        num_return_sequences=1,
    )

    return outputs

def extract_fills(generated_token_ids, mask_tokenizer):
    """Extract the filled text from the generated token sequences"""
    extracted_fills = []

    # Get mask token ids for lookup
    mask_token_ids = []
    for i in range(100):  # Support up to <extra_id_99>
        try:
            mask_id = mask_tokenizer.encode(f'<extra_id_{i}>', add_special_tokens=False)[0]
            mask_token_ids.append(mask_id)
        except:
            break

    for token_ids in generated_token_ids:
        # Convert tensor to list if needed
        if hasattr(token_ids, 'tolist'):
            tokens = token_ids.tolist()
        else:
            tokens = token_ids

        # Remove special tokens (pad, eos, bos)
        special_tokens = set()
        if mask_tokenizer.pad_token_id is not None:
            special_tokens.add(mask_tokenizer.pad_token_id)
        if mask_tokenizer.eos_token_id is not None:
            special_tokens.add(mask_tokenizer.eos_token_id)
        if mask_tokenizer.bos_token_id is not None:
            special_tokens.add(mask_tokenizer.bos_token_id)

        # Remove special tokens
        tokens = [t for t in tokens if t not in special_tokens]

        # Extract fills correctly: look for <extra_id_i> content </s> pattern
        fills = []
        i = 0
        n = len(tokens)

        while i < n:
            # Find next <extra_id_i>
            if tokens[i] in mask_token_ids:
                # Found a mask token, now extract content until next mask or end
                i += 1  # Skip the mask token
                fill_tokens = []

                # Collect tokens until next mask token or end of sequence
                while i < n and tokens[i] not in mask_token_ids:
                    fill_tokens.append(tokens[i])
                    i += 1

                # Decode the fill content
                if fill_tokens:
                    fill_text = mask_tokenizer.decode(fill_tokens, skip_special_tokens=True)
                    fills.append(fill_text)
            else:
                # Skip tokens that aren't part of a fill pattern
                i += 1

        extracted_fills.append(fills)

    total_fills = sum(len(fills) for fills in extracted_fills)

    # Detailed debug for first sequence
    if len(generated_token_ids) > 0:
        first_token_ids = generated_token_ids[0]
        if hasattr(first_token_ids, 'tolist'):
            first_tokens = first_token_ids.tolist()
        else:
            first_tokens = first_token_ids

        # Find mask token positions
        mask_positions = [i for i, t in enumerate(first_tokens) if t in mask_token_ids]

    return extracted_fills

def apply_extracted_fills(masked_token_ids, extracted_fills, mask_tokenizer):
    """Apply the extracted fills back to the masked token sequences"""
    result_token_lists = []

    # Get mask token ids for lookup
    mask_token_ids = []
    for i in range(100):  # Support up to <extra_id_99>
        try:
            mask_id = mask_tokenizer.encode(f'<extra_id_{i}>', add_special_tokens=False)[0]
            mask_token_ids.append(mask_id)
        except:
            break

    successful_replacements = 0

    for masked_tokens, fills in zip(masked_token_ids, extracted_fills):
        if len(fills) == 0:
            # No fills, return empty list
            result_token_lists.append([])
            continue

        # Copy the masked tokens
        result_tokens = masked_tokens.copy()

        # Find mask positions
        mask_positions = []
        for i, token_id in enumerate(result_tokens):
            if token_id in mask_token_ids:
                mask_positions.append((i, token_id))

        # Sort by mask index (from <extra_id_0> to highest)
        mask_positions.sort(key=lambda x: x[1])

        # Replace masks with fills
        for mask_idx, (pos, mask_token_id) in enumerate(mask_positions):
            if mask_idx < len(fills):
                # Convert fill text to tokens
                fill_tokens = mask_tokenizer.encode(fills[mask_idx], add_special_tokens=False)
                # Replace the mask token with fill tokens
                result_tokens[pos:pos+1] = fill_tokens
                successful_replacements += 1

        result_token_lists.append(result_tokens)

    non_empty_results = sum(1 for lst in result_token_lists if len(lst) > 0)
    return result_token_lists

def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    """Generate perturbed versions of texts"""
    span_length = args.span_length
    pct = args.pct_words_masked

    # Step 1: Tokenize and mask texts
    masked_results = [tokenize_and_mask(text, mask_tokenizer, span_length, pct, ceil_pct) for text in texts]
    masked_token_ids = [result[0] for result in masked_results]
    mask_counts = [result[1] for result in masked_results]

    total_masks = sum(mask_counts)

    # Step 2: Generate fills for masks
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_token_ids)
    extracted_fills = extract_fills(raw_fills, mask_tokenizer)

    total_fills = sum(len(fills) for fills in extracted_fills)

    # Step 3: Apply fills back to masked tokens
    perturbed_token_ids = apply_extracted_fills(masked_token_ids, extracted_fills, mask_tokenizer)

    # Step 4: Convert back to strings
    perturbed_texts = [mask_tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
                      for tokens in perturbed_token_ids]

    successful_results = sum(1 for tokens in perturbed_token_ids if len(tokens) > 0)

    attempts = 1
    while attempts <= 3:  # Limited retry attempts
        # Check for empty results
        failed_idxs = [idx for idx, tokens in enumerate(perturbed_token_ids) if len(tokens) == 0]

        if len(failed_idxs) == 0:
            break

        print(f'WARNING: {len(failed_idxs)} texts failed, retrying [attempt {attempts}]')

        # Retry only for failed texts
        retry_texts = [texts[idx] for idx in failed_idxs]
        retry_masked_results = [tokenize_and_mask(text, mask_tokenizer, span_length, pct, ceil_pct) for text in retry_texts]
        retry_masked_token_ids = [result[0] for result in retry_masked_results]

        retry_raw_fills = replace_masks(args, mask_model, mask_tokenizer, retry_masked_token_ids)
        retry_extracted_fills = extract_fills(retry_raw_fills, mask_tokenizer)
        retry_perturbed_token_ids = apply_extracted_fills(retry_masked_token_ids, retry_extracted_fills, mask_tokenizer)
        retry_perturbed_texts = [mask_tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
                                for tokens in retry_perturbed_token_ids]

        # Update results
        for i, idx in enumerate(failed_idxs):
            perturbed_token_ids[idx] = retry_perturbed_token_ids[i]
            perturbed_texts[idx] = retry_perturbed_texts[i]

        successful_results = sum(1 for tokens in perturbed_token_ids if len(tokens) > 0)
        attempts += 1

    return perturbed_texts

def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    """Generate perturbations with long text segmentation"""
    outputs = []
    for text in texts:
        perturbed = perturb_texts_(args, mask_model, mask_tokenizer, [text], ceil_pct=ceil_pct)
        outputs.extend(perturbed)

    return outputs

def get_ll(args, scoring_model, scoring_tokenizer, text):
    """Get log-likelihood of text under scoring model"""
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt",padding=True,
                                     return_token_type_ids=False, 
                                     truncation=True, max_length=1024).to(args.device)
        # Ensure input_ids is LongTensor (int64) for embedding layer
        tokenized['input_ids'] = tokenized['input_ids'].long()
        labels = tokenized.input_ids
        return -scoring_model(**tokenized, labels=labels).loss.item()

def get_lls(args, scoring_model, scoring_tokenizer, texts):
    """Get log-likelihoods for multiple texts"""
    return [get_ll(args, scoring_model, scoring_tokenizer, text) for text in texts]

def handle_detectgpt(args, data, n_samples):
    """Handle DetectGPT perturbation-based detection"""
    n_perturbations = args.n_perturbations
    name = f'perturbation_{n_perturbations}'

    # Load mask-filling model (T5)
    mask_model = load_mask_model(args.mask_filling_model_name, args.device, args.cache_dir)
    mask_model.eval()

    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 300

    mask_tokenizer = load_mask_tokenizer(args.mask_filling_model_name, n_positions, args.cache_dir)

    # Generate perturbations for all texts
    perturbs = []
    perturb_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc="Perturbing texts"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # Perturb both original and sampled texts
        p_sampled_text = perturb_texts(args, mask_model, mask_tokenizer,
                                      [sampled_text for _ in range(n_perturbations)])
        p_original_text = perturb_texts(args, mask_model, mask_tokenizer,
                                       [original_text for _ in range(n_perturbations)])

        assert len(p_sampled_text) == n_perturbations
        assert len(p_original_text) == n_perturbations

        # Store results
        perturbs.append({
            "original": original_text,
            "sampled": sampled_text,
            "perturbed_sampled": p_sampled_text,
            "perturbed_original": p_original_text
        })
    perturb_time = time.time() - perturb_time_start
    del mask_model
    del mask_tokenizer
    clear_gpu_memory()

    # Load scoring model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # Compute log-likelihoods and metrics
    results = []
    crit_compute_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc="Computing metrics"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        perturbed_original = perturbs[idx]["perturbed_original"]
        perturbed_sampled = perturbs[idx]["perturbed_sampled"]

        # Original text
        original_ll = get_ll(args, scoring_model, scoring_tokenizer, original_text)
        p_original_ll = get_lls(args, scoring_model, scoring_tokenizer, perturbed_original)

        # Sampled text
        sampled_ll = get_ll(args, scoring_model, scoring_tokenizer, sampled_text)
        p_sampled_ll = get_lls(args, scoring_model, scoring_tokenizer, perturbed_sampled)

        # Store metrics
        results.append({
            "original": original_text,
            "sampled": sampled_text,
            "original_ll": original_ll,
            "sampled_ll": sampled_ll,
            "all_perturbed_sampled_ll": p_sampled_ll,
            "all_perturbed_original_ll": p_original_ll,
            "perturbed_sampled_ll": np.mean(p_sampled_ll),
            "perturbed_original_ll": np.mean(p_original_ll),
            "perturbed_sampled_ll_std": np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1,
            "perturbed_original_ll_std": np.std(p_original_ll) if len(p_original_ll) > 1 else 1
        })

    # Compute final scores
    predictions = {'real': [], 'samples': []}

    for res in results:
        if res['perturbed_original_ll_std'] == 0:
            res['perturbed_original_ll_std'] = 1
            print("WARNING: std of perturbed original is 0, setting to 1")

        if res['perturbed_sampled_ll_std'] == 0:
            res['perturbed_sampled_ll_std'] = 1
            print("WARNING: std of perturbed sampled is 0, setting to 1")

        # Normalize by standard deviation with NaN checking
        real_diff = (res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std']
        sample_diff = (res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std']

        # Check for NaN or infinite values
        if np.isnan(real_diff) or np.isinf(real_diff):
            print(f"WARNING: real_diff is NaN or inf: {real_diff}, original_ll={res['original_ll']}, perturbed_original_ll={res['perturbed_original_ll']}")
            real_diff = 0.0

        if np.isnan(sample_diff) or np.isinf(sample_diff):
            print(f"WARNING: sample_diff is NaN or inf: {sample_diff}, sampled_ll={res['sampled_ll']}, perturbed_sampled_ll={res['perturbed_sampled_ll']}")
            sample_diff = 0.0

        predictions['real'].append(real_diff)
        predictions['samples'].append(sample_diff)
    crit_compute_time = time.time() - crit_compute_time_start
    total_time = perturb_time + crit_compute_time
    time_per_sample = total_time / n_samples
    print(f"[INFO] perturb time: {perturb_time:.4f}, crit time: {crit_compute_time:.4f}, total time: {total_time:.4f}")
    print(f"[INFO] time per sample of {args.method} method: {time_per_sample:.4f}")
    
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Results
    results_dict = {
        'name': name,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': args.span_length,
            'n_perturbations': args.n_perturbations,
            'n_samples': n_samples,
        },
        'predictions': predictions,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
        'time': {
            'perturb_time': perturb_time,
            'crit_compute_time': crit_compute_time,
            'total_time': total_time, 
            'n_samples': n_samples, 
            'time_per_sample': time_per_sample,
        },
    }

    return results_dict
