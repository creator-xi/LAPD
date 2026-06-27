import torch

def sample_ll_add(ll_base: torch.tensor, ll_inst: torch.tensor, lambda_: float) -> torch.tensor:
    return ll_inst - ll_base

def token_level_sample_ll_mul(ll_base: torch.tensor, ll_inst: torch.tensor, lambda_: float) -> torch.tensor:
    """
    Assume ll_base, ll_inst not averaged in token sequence dim.
    Multiply ll token by token, then average them
    """
    output = ll_inst * (ll_base - ll_inst)
    return output.mean(dim=1)