from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7b': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6b': 'local.gpt-j-6b',
                     'gpt-j-6b-instruct': 'local.gpt-j-6b-instruct',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama2-7b': 'local.llama-2-7b',
                     'llama2-7b-instruct': 'local.Llama-2-7b-instruct',
                     'llama2-7b-chat': 'local.llama-2-7b-chat-hf',
                     'llama3.1-8b': 'local.Meta-Llama-3.1-8B',
                     'llama3.1-8b-instruct': 'local.Meta-Llama-3.1-8B-Instruct',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'falcon-7b': 'tiiuae/falcon-7b',
                     'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
                     }
float16_models = ['gpt-neo-2.7B', 'gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b',
                  'falcon-7b', 'falcon-7b-instruct']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')

    is_local = model_fullname.startswith('local.')
    if is_local:
        model_path = os.path.join(cache_dir, model_fullname.replace("local.", ""))

        if not os.path.exists(model_path):
            raise ValueError(f"Local model path {model_path} does not exist.")

        print(f"Loading local model from {model_path}")
        
        model_kwargs = {
            'device_map': device,
            'torch_dtype': torch.float16,
            'local_files_only': True,
        }

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    else:
        model_kwargs = {}
        if model_name in float16_models:
            model_kwargs.update(dict(dtype=torch.float16))
        if 'gpt-j' in model_name:
            model_kwargs.update(dict(revision='float16'))
        if 'falcon' in model_name:
            model_kwargs.update(dict(trust_remote_code=True))
        # honor offline mode when set
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            model_kwargs.update(dict(local_files_only=True))

        if 'falcon' in model_name.lower():
            model_kwargs = {
                'device_map': device,
                'torch_dtype': torch.float16,
                'trust_remote_code': True
            }
        else:
            model_kwargs = {
                'device_map': device,
                'torch_dtype': torch.float16,
            }
        model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    start = time.time()
    # If not using device_map (i.e., single-device model), move model to requested device
    has_device_map = model_kwargs.get("device_map") is not None
    if not has_device_map:
        try:
            model.to(device)
        except Exception:
            pass
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    
    is_local = model_fullname.startswith('local.')
    
    if is_local:
        local_name = model_fullname.replace('local.', '')
        model_path = os.path.join(cache_dir, local_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model not found at: {model_path}")
        
        print(f"Loading tokenizer from local path: {model_path}")
        
        optional_tok_kwargs = {
            'padding_side': 'right',
            'local_files_only': True,
        }
        
        base_tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)

    else:
        optional_tok_kwargs = {}
        if "facebook/opt-" in model_fullname:
            print("Using non-fast tokenizer for OPT")
            optional_tok_kwargs['fast'] = False
        optional_tok_kwargs['padding_side'] = 'right'
        base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)

    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def check_tokenizer_compatibility(model_name1, model_name2, cache_dir, test_texts=None):
    """
    Check if tokenizers of two models are compatible.
    Args:
        model_name1: First model name
        model_name2: Second model name
        cache_dir: Model cache directory
        test_texts: Test text list, defaults to a few simple texts
    Returns:
        bool: True if tokenizers are compatible, False otherwise
    Raises:
        Exception: If tokenizers are not compatible
    """
    import gc
    if test_texts is None:
        test_texts = [
            "Hello world",
            "This is a test sentence.",
            "How are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "123 test numbers and symbols !@#$%"
        ]
    print(f"[INFO] Checking tokenizer compatibility between {model_name1} and {model_name2}")
    try:
        tokenizer1 = load_tokenizer(model_name1, cache_dir)
        tokenizer2 = load_tokenizer(model_name2, cache_dir)
        vocab_size1 = tokenizer1.vocab_size
        vocab_size2 = tokenizer2.vocab_size
        print(f"[INFO] Model {model_name1} vocab size: {vocab_size1}, {model_name2}: {vocab_size2}")
        
        if vocab_size1 != vocab_size2:
            error_msg = f"Tokenizer vocab sizes differ: {model_name1}={vocab_size1}, {model_name2}={vocab_size2}"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        special_tokens1 = tokenizer1.special_tokens_map
        special_tokens2 = tokenizer2.special_tokens_map
        print(f"[INFO] Special tokens comparison:")
        for key in special_tokens1:
            if key in special_tokens2:
                print(f"  {key}: '{special_tokens1[key]}' vs '{special_tokens2[key]}' -> {'MATCH' if special_tokens1[key] == special_tokens2[key] else 'MISMATCH'}")
        
        all_match = True
        mismatched_examples = []
        print(f"[INFO] Testing tokenization on {len(test_texts)} sample texts:")
        for i, text in enumerate(test_texts):
            try:
                tokens1 = tokenizer1.encode(text)
                tokens2 = tokenizer2.encode(text)

                match = tokens1 == tokens2
                all_match = all_match and match

                status = "MATCH" if match else "MISMATCH"
                print(f"  Text {i+1}: '{text}' -> {status}")
                print(f"    {model_name1}: {tokens1}")
                print(f"    {model_name2}: {tokens2}")

                if not match:
                    mismatched_examples.append((text, tokens1, tokens2))

            except Exception as e:
                print(f"  Text {i+1}: ERROR during tokenization - {e}")
                all_match = False
                mismatched_examples.append((text, str(e), str(e)))

        if all_match:
            print(f"[SUCCESS] Tokenizers are fully compatible!")
            result = True
        else:
            error_msg = f"Tokenizers are NOT compatible! {len(mismatched_examples)} mismatches found."
            print(f"[ERROR] {error_msg}")
            print(f"[INFO] Mismatched examples:")
            for text, tokens1, tokens2 in mismatched_examples[:3]:
                print(f"  Text: '{text}'")
                print(f"    {model_name1}: {tokens1}")
                print(f"    {model_name2}: {tokens2}")
            raise ValueError(error_msg)

    except Exception as e:
        print(f"[ERROR] Error during tokenizer compatibility check: {e}")
        raise e
    finally:
        try:
            if 'tokenizer1' in locals():
                del tokenizer1
            if 'tokenizer2' in locals():
                del tokenizer2
            gc.collect()
            print(f"[INFO] Tokenizers cleaned up successfully")
        except Exception as cleanup_error:
            print(f"[WARNING] Error during cleanup: {cleanup_error}")

    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="bloom-7b1")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
