import numpy as np
import torch
import transformers
import random
from torch.nn import functional as F
from scipy.spatial.distance import jensenshannon

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def min_perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0
               ):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    # Maximize token probability at each time step
    max_prob_token = torch.argmax(shifted_logits,
                                  dim=-1)  # This picks the token with the max probability at each position

    # Create a shifted labels tensor for the original sequence
    shifted_labels_max = max_prob_token
    # Now, for the perplexity calculation, we compare the shifted logits with the max token sequence
    if median:
        # Compute the cross-entropy loss based on max token sequence
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max)
                  .masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        # Compute perplexity using the max token sequence and attention mask
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels_max) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl

def auc_perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0,
               max_batch_size: int = 50,
               repair_order: str="s"):
    shifted_logits = logits[..., :-1, :] / temperature
    shifted_attention_mask = encoding.attention_mask[..., 1:]

    probs = torch.softmax(shifted_logits, dim=-1)
    max_prob_tokens = probs.argmax(dim=-1)

    input_ids = encoding.input_ids[..., 1:].clone()

    current_labels = input_ids.clone()

    ce_initial = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels).float()
    ppl_sequence = [(ce_initial * shifted_attention_mask).sum() / shifted_attention_mask.sum()]

    logits_diff = torch.abs(probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1) - probs.gather(-1, max_prob_tokens.unsqueeze(-1)).squeeze(-1))

    non_max_mask = (input_ids != max_prob_tokens)
    change_indices = non_max_mask.nonzero(as_tuple=False)
    if repair_order == "s":
        sorted_indices = change_indices
    elif repair_order == "h2l":
        sorted_indices = sorted(change_indices.tolist(), key=lambda idx: logits_diff[idx[0], idx[1]].item())
    elif repair_order == "l2h":
        sorted_indices = sorted(change_indices.tolist(), key=lambda idx: -logits_diff[idx[0], idx[1]].item())
    elif repair_order == "r":
        sorted_indices = change_indices.tolist()
        random.shuffle(sorted_indices)
    for idx in change_indices:
        batch_idx, token_idx = idx
        current_labels[batch_idx, token_idx] = max_prob_tokens[batch_idx, token_idx]

        ce_current = ce_loss_fn(shifted_logits.transpose(1, 2), current_labels)
        current_ppl = (ce_current * shifted_attention_mask).sum() / shifted_attention_mask.sum()
        current_ppl = current_ppl.float()
        ppl_sequence.append(current_ppl)

    ppl_sequence_np = torch.stack(ppl_sequence).cpu().numpy()
    auc = np.mean(ppl_sequence_np)

    return auc


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()


    return ppl

import torch
import numpy as np
from torch.nn import CrossEntropyLoss

ce_loss_fn = CrossEntropyLoss(reduction='none')
@torch.no_grad()
def sum_perplexity(encoding: transformers.BatchEncoding,
                   logits: torch.Tensor,
                   median: bool = False,
                   temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :] / temperature
    attention = encoding.attention_mask[..., 1:]
    labels_std = encoding.input_ids[..., 1:]
    labels_max = torch.argmax(shifted_logits, dim=-1)

    logits_T = shifted_logits.transpose(1, 2)

    ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
    ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

    attn_sum = attention.sum(dim=1).clamp(min=1)
    ppl_std = (ce_std * attention).sum(dim=1) / attn_sum
    ppl_max = (ce_max * attention).sum(dim=1) / attn_sum

    return (ppl_std + ppl_max).cpu().numpy()

import torch
import torch.nn.functional as F
import numpy as np

def joint_score_ratio(p_logits: torch.Tensor,
                      q_logits: torch.Tensor,
                      encoding: transformers.BatchEncoding,
                      pad_token_id: int,
                      temperature: float = 1.0,
                      median: bool = False):
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask
    vocab_size = p_logits.shape[-1]

    # --- Cross Entropy Part ---
    p_log_probs = F.log_softmax(p_logits / temperature, dim=-1)
    q_probs = F.softmax(q_logits / temperature, dim=-1)

    kl = F.kl_div(p_log_probs, q_probs, reduction='none').sum(-1)  # [B, L]
    pad_mask = (input_ids != pad_token_id).float()

    if median:
        kl_masked = kl.masked_fill(pad_mask == 0, float('nan'))
        agg_ce = torch.nanmedian(kl_masked, dim=1).values.cpu().numpy()
    else:
        agg_ce = ((kl * pad_mask).sum(dim=1) / pad_mask.sum(dim=1)).cpu().numpy()

    # --- Perplexity Part ---
    shifted_logits = q_logits[:, :-1, :] / temperature
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention = attention_mask[:, 1:]

    logits_T = shifted_logits.transpose(1, 2)
    labels_std = shifted_input_ids
    labels_max = torch.argmax(shifted_logits, dim=-1)

    ce_std = F.cross_entropy(logits_T, labels_std, reduction='none')
    ce_max = F.cross_entropy(logits_T, labels_max, reduction='none')

    attn_sum = shifted_attention.sum(dim=1).clamp(min=1)
    ppl_std = (ce_std * shifted_attention).sum(dim=1) / attn_sum
    ppl_max = (ce_max * shifted_attention).sum(dim=1) / attn_sum

    return ((ppl_std + ppl_max) / torch.tensor(agg_ce, device=ppl_std.device)).cpu().numpy()


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)
    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce

def relative_entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    max_vals = p_proba.max(dim=-1, keepdim=True)[0]
    p_proba = p_proba/max_vals


    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def entropy_pro(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]

    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)
    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    max_p_token = torch.argmax(p_proba, dim=-1)  # Select the token with maximum probability
    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=max_p_token).view(-1, total_tokens_available)

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce




def w_entropy(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        encoding: transformers.BatchEncoding,
        pad_token_id: int,
        median: bool = False,
        sample_p: bool = False,
        temperature: float = 1.0
):
    use_weight = True
    shifted_labels = encoding.input_ids[..., :].contiguous()
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)

    if use_weight:
        p_probs = F.softmax(p_logits / temperature, dim=-1)  # [batch, seq_len, vocab]

        current_probs = p_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1)  # [batch, seq_len, 1]
        ).squeeze(-1)  # [batch, seq_len]

        max_probs = p_probs.max(dim=-1).values  # [batch, seq_len]
        weights = max_probs - current_probs  # [batch, seq_len]
        ce = ce * weights

    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())



    return agg_ce


def calculate_window_ppl(observer_logits: torch.Tensor,
               performer_logits: torch.Tensor,
               encoding: transformers.BatchEncoding,
               pad_token_id: int,
               global_value: float,
               window_radio: float = 0.1):
    seq_len = len(encoding.input_ids[..., 0:][0])
    ppl_values = []
    xppl_values = []
    standard_ppl_values = []
    standard_xppl_values = []
    local_values = []
    window_size = int(seq_len * window_radio)
    step_size = int(window_size*0.5)
    if seq_len <= 50:
        window_size = int(seq_len * 0.5)
        step_size = int(window_size * 0.5)

    shifted_logits = performer_logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()

    ppls = torch.softmax(shifted_logits, dim=-1) # [batch, seq_len-1, vocab]
    max_ppls, _ = ppls.max(dim=-1) # [batch, seq_len-1]
    max_ppls = max_ppls[0]
    ppls = ppls.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)[0]
    global_cos_sim = ((ppls*max_ppls).sum(dim=-1)) / (ppls.norm(dim=-1) * max_ppls.norm(dim=-1) + 1e-9)

    if seq_len >= window_size:
        ppls_windows = ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]
        max_ppls_windows = max_ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]

        dot_product = (ppls_windows * max_ppls_windows).sum(dim=-1)
        ppls_norm = ppls_windows.norm(dim=-1)
        max_ppls_norm = max_ppls_windows.norm(dim=-1)
        cos_similarities = dot_product / (ppls_norm * max_ppls_norm + 1e-9)

        mean_cos_sim = cos_similarities.mean().item()
        std_cos_sim = cos_similarities.std().item()
    else:
        mean_cos_sim = 0.0 

    return (global_cos_sim-mean_cos_sim)/std_cos_sim



def calculate_window_entropy(observer_logits: torch.Tensor,
                         performer_logits: torch.Tensor,
                         encoding: transformers.BatchEncoding,
                         pad_token_id: int,
                         global_value: float,
                         window_radio: float = 0.1):
    seq_len = len(encoding.input_ids[..., 0:][0])
    ppl_values = []
    xppl_values = []
    standard_ppl_values = []
    standard_xppl_values = []
    local_values = []
    window_size = int(seq_len * window_radio)
    step_size = int(window_size * 0.5)
    if seq_len <= 50:
        window_size = int(seq_len * 0.5)
        step_size = int(window_size * 0.5)
    elif seq_len <= 200:
        window_size = int(seq_len * 0.2)
        step_size = int(window_size * 0.5)

    shifted_logits = performer_logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    q_shifted_logits = observer_logits[..., :-1, :].contiguous()

    ppls = torch.softmax(shifted_logits, dim=-1)  # [batch, seq_len-1, vocab]
    max_ppls, _ = ppls.max(dim=-1)  # [batch, seq_len-1]
    max_ppls = max_ppls[0]
    ppls = ppls.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)[0]

    q_ppls = torch.softmax(q_shifted_logits, dim=-1)  # [batch, seq_len-1, vocab]
    q_max_ppls, _ = q_ppls.max(dim=-1)  # [batch, seq_len-1]
    q_max_ppls = q_max_ppls[0]
    q_ppls = q_ppls.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)[0]

    if seq_len >= window_size:
        ppls_windows = ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]
        max_ppls_windows = max_ppls.unfold(0, window_size, step_size)  # [num_windows, window_size]

        q_ppls_windows = q_ppls.unfold(0, window_size, step_size)
        q_max_ppls_windows = q_max_ppls.unfold(0, window_size, step_size)

        p_diff = -torch.log(ppls_windows + 1e-9) + torch.log(max_ppls_windows + 1e-9)
        q_diff = -torch.log(q_ppls_windows + 1e-9) + torch.log(q_max_ppls_windows + 1e-9)

        diff = (p_diff * q_diff)/(p_diff + q_diff + 1e-9)

        print(diff.mean(dim=1))
        diff_mean = diff.mean(dim=1)
        diff_mean[diff_mean < 0.30] = 0


    else:
        diff = 0.0

    return diff_mean.mean()