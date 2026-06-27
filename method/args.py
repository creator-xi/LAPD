"""
Common argument parsing utilities for detection methods.
Centralizes shared parameters to reduce code duplication.
"""
import argparse

def add_common_arguments(parser):
    """Add common arguments used by most detection methods."""
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Model cache directory')
    parser.add_argument('--max_samples', type=int, default=-1, help='Maximum number of samples to use')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory of the data folder')
    # Data source arguments
    parser.add_argument('--data_source', type=str, default='main',
                       choices=['main', 'm4', 'detectrl_multidomain', 'detectrl_multillm', 'raid', 'text_attack', 'realdet', 'raid_attack', 'cred'],
                       help='Data source for unified loading')
    parser.add_argument('--dataset', type=str, default="xsum", choices=["xsum", "wp", "arxiv"], help='Dataset name for main')
    parser.add_argument('--source_model', type=str, default="gpt4o", choices=["claude3.7", "gemini2.0", "gpt4o"], help='Source model for main')
    parser.add_argument('--attack_type', type=str, default='delete', choices=["delete", "dipper", "insert", "replace"], help='Attack type for text_attack')
    parser.add_argument('--raid_attack_type', type=str, default='none',
                       choices=['none', 'whitespace', 'synonym', 'perplexity_misspelling',
                                'paraphrase', 'homoglyph', 'zero_width_space'],
                       help='Attack type for raid_attack')
    # CReD dataset arguments
    parser.add_argument('--cred_domain', type=str, default='composition',
                       choices=['composition', 'film_review', 'news', 'paper', 'question_answer', 'TC_news'],
                       help='Domain for CReD dataset')
    parser.add_argument('--cred_model', type=str, default='gpt-4o',
                       choices=['claude-3.5-haiku', 'deepseek-r1', 'deepseek-v3', 'doubao-1.5-pro',
                                'gemini-2.5-flash', 'gpt-3.5-turbo', 'gpt-4o', 'qwen-2.5', 'qwen-3'],
                       help='Model for CReD dataset')
    # time efficiency arguments
    parser.add_argument('--time_compute', action='store_true', help='time efficiency record for method')
    # text length ablation arguments
    parser.add_argument('--max_words', type=int, default=None, help='Maximum number of words to truncate text for ablation study (None means no truncation)')


def add_model_arguments(parser):
    """Add model-related arguments."""
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct", help='Scoring/Observer model name (for methods that need it)')
    parser.add_argument('--sampling_model_name', type=str, default="falcon-7b", help='Sampling/Performer model name (optional)')


def add_lapd_method_arguments(parser):
    """Add arguments specific to multi-model methods (like lapd)."""
    parser.add_argument('--sampling_model', type=str, default="llama2-7b")
    parser.add_argument('--base_model', type=str, default="llama2-7b")
    parser.add_argument('--instruct_model', type=str, default="llama2-7b-instruct")
    parser.add_argument('--lambda_', type=float, default=0.0, help="weight for base model in the weighted sum")
    parser.add_argument('--add', action='store_true', default=False, help="use add aggregation strategy (with lambda_ = 0 only)")


def add_detectgpt_arguments(parser):
    """Add DetectGPT specific arguments."""
    parser.add_argument('--pct_words_masked', type=float, default=0.3, help='Percentage of words masked')
    parser.add_argument('--mask_top_p', type=float, default=1.0, help='Top-p for mask filling')
    parser.add_argument('--span_length', type=int, default=2, help='Span length for masking')
    parser.add_argument('--n_perturbations', type=int, default=10, help='Number of perturbations')
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-small", help='Mask filling model')


def add_fast_arguments(parser):
    """Add Fast-DetectGPT specific arguments."""
    parser.add_argument('--discrepancy_analytic', action='store_true', help='Use analytic version of sampling discrepancy')


def add_lastde_arguments(parser):
    """Add LastDE specific arguments."""
    parser.add_argument('--embed_size', type=int, default=4, help='Embedding size (for lastde)')
    parser.add_argument('--epsilon', type=float, default=8, help='Epsilon parameter (for lastde)')
    parser.add_argument('--tau_prime', type=int, default=15, help='Tau prime parameter (for lastde)')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples (for lastde)')


def create_baselines_parser():
    """Create argument parser for baseline methods."""
    parser = argparse.ArgumentParser(
        description="Unified Baseline Detection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Supported Methods:
        - likelihood: Simple likelihood-based detection
        - rank: Rank-based detection
        - logrank: Log-rank based detection
        - entropy: Entropy-based detection
        - fast: Fast-DetectGPT sampling discrepancy
        - fast_origin: Original Fast-DetectGPT
        - detectgpt: DetectGPT perturbation-based detection
        - lastde: LastDE with fastMDE
        - binoculars: Binoculars baseline detection method
        - binoculars_zscore: Binoculars with analytic z-score standardization
        - dna_detectllm: DNA-GPT method
        - rai: Instructional Representation Measurement (base - instruct likelihood)

binoculars/dna_detectllm Field Mapping:
        scoring_model -> performer_model
        sampling_model -> observer_model
RAI Field Mapping:
        scoring_model -> instruct_model
        sampling_model -> base_model
        """
    )

    # Method selection (baselines specific)
    parser.add_argument('--method', type=str, required=True,
                       choices=['likelihood', 'logrank', 'entropy', 'fast', 'fast_origin', 'detectgpt', 'lastde',
                                'dna_detectllm', 'binoculars', 'binoculars_zscore', 'roberta_base', 'roberta_large', 'rai', 'rai_double_gpu'],
                       help='Detection method to run')

    # Add common arguments
    add_common_arguments(parser)
    add_model_arguments(parser)
    # Add method-specific arguments (all of them for flexibility)
    add_detectgpt_arguments(parser)
    add_fast_arguments(parser)
    add_lastde_arguments(parser)

    return parser


def add_comparison_arguments(parser):
    """Add arguments specific to comparison methods (like component_abla)."""
    parser.add_argument('--comparison_method', type=str,
                       choices=['base', 'base-inst', 'd_base-inst', 'inst_mul_base-inst'],
                       default='base-inst',
                       help='Comparison method to use: base, base-inst, d_base-inst, inst_mul_base-inst')
    return parser


def create_mul_method_parser():
    """Create argument parser for multi-model methods."""
    parser = argparse.ArgumentParser(description="Multi-Model Detection Method")
    add_common_arguments(parser)
    add_lapd_method_arguments(parser)

    return parser


def create_contribution_analysis_parser():
    """Create argument parser for contribution analysis methods."""
    parser = argparse.ArgumentParser(description="Contribution Analysis Method")
    add_common_arguments(parser)
    add_lapd_method_arguments(parser)
    add_comparison_arguments(parser)

    return parser
