"""
load data from where DNA-DetectLLM prepared
"""
import csv
import json
import os
import glob
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

# Dataset paths configuration
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO_ROOT, "data"))
DATASET_CONFIG = {
    "m4": {
        "root_dir": f"{DATA_ROOT}/M4",
        "human_file": "m4_human.json",
        "machine_file": "m4_machine.json"
    },
    "main": {
        "root_dir": f"{DATA_ROOT}/Collected data",
    },
    "detectrl": {
        "root_dir": f"{DATA_ROOT}/DetectRL"
    },
    "raid": {
        "root_dir": f"{DATA_ROOT}/RAID",
        "human_file": "raid_human.json",
        "machine_file": "raid_machine.json"
    },
    "realdet": {
        "root_dir": f"{DATA_ROOT}/RealDet",
        "human_file": "RealDet_human_test.json",
        "machine_file": "RealDet_machine_test.json"
    },
    "text_attack": {
        "root_dir": f"{DATA_ROOT}/Text_attack",
        "human_file": "human_texts.json"
    },
    "raid_attack": {
        "root_dir": f"{DATA_ROOT}/RAID/attacks",
    },
    "cred": {
        "root_dir": f"{DATA_ROOT}/CReD",
    }
}

class DataSourceStrategy(ABC):
    """Abstract base class for data source loading strategies."""

    @abstractmethod
    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data."""
        pass

    @abstractmethod
    def get_available_domains(self) -> List[str]:
        """Get available data domains."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get available models."""
        pass

class BaseJSONDataSource(DataSourceStrategy):
    """Base class for JSON file-based data source loading strategies."""

    def __init__(self, human_file: str, machine_file: str):
        self.human_file = human_file
        self.machine_file = machine_file

    def _load_json_file(self, file_path: str) -> List[str]:
        """Load a JSON file and extract text list."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "human_text" in data:
            return data["human_text"]
        elif "machine_text" in data:
            return data["machine_text"]
        else:
            return data if isinstance(data, list) else [data]

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """Generic data loading logic."""
        human_texts = self._load_json_file(self.human_file)
        ai_texts = self._load_json_file(self.machine_file)

        min_length = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_length]
        ai_texts = ai_texts[:min_length]

        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]

        return human_texts, ai_texts


class M4DataSource(BaseJSONDataSource):
    """M4 dataset loading strategy - uses pre-prepared data files."""

    def __init__(self, data_root: str = DATASET_CONFIG['m4']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['m4']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['m4']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class DetectRLDataSource(BaseJSONDataSource):
    """DetectRL dataset loading strategy."""

    def __init__(self, data_root: str = DATASET_CONFIG['detectrl']['root_dir'], dataset_type: str = "multidomain"):
        if dataset_type not in ["multidomain", "multillm"]:
            raise ValueError(f"dataset_type must be 'multidomain' or 'multillm', got: {dataset_type}")

        self.dataset_type = dataset_type
        human_file = os.path.join(data_root, f"DetectRL_{dataset_type}_human_test.json")
        machine_file = os.path.join(data_root, f"DetectRL_{dataset_type}_machine_test.json")
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return [self.dataset_type]

    def get_available_models(self) -> List[str]:
        return []

class RAIDDataSource(BaseJSONDataSource):
    """RAID dataset loading strategy - uses pre-prepared data files."""

    def __init__(self, data_root: str = DATASET_CONFIG['raid']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['raid']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['raid']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class RealDetDataSource(BaseJSONDataSource):
    """RealDet dataset loading strategy."""

    def __init__(self, data_root: str = DATASET_CONFIG['realdet']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['realdet']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['realdet']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class TextAttackDataSource(BaseJSONDataSource):
    """Text Attack dataset loading strategy - supports multiple attack types."""

    def __init__(self, data_root: str = DATASET_CONFIG['text_attack']['root_dir']):
        self.data_root = data_root
        self.human_file = os.path.join(data_root, DATASET_CONFIG['text_attack']['human_file'])
        self.models = ["Claude", "Gemini", "GPT4"]
        self.attack_types = ["delete", "dipper", "insert", "replace"]

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """Load Text Attack data."""
        model = kwargs.get("model", "GPT4")
        attack_type = kwargs.get("attack_type", "delete")

        if model not in self.models:
            raise ValueError(f"Unsupported model: {model}. Available: {self.models}")
        if attack_type not in self.attack_types:
            raise ValueError(f"Unsupported attack type: {attack_type}. Available: {self.attack_types}")

        machine_file = os.path.join(self.data_root, f"{model}_machine_test_{attack_type}.json")

        human_texts = self._load_json_file(self.human_file)
        ai_texts = self._load_json_file(machine_file)

        min_length = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_length]
        ai_texts = ai_texts[:min_length]

        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]

        return human_texts, ai_texts

    def get_available_domains(self) -> List[str]:
        """Get available attack types."""
        return self.attack_types

    def get_available_models(self) -> List[str]:
        """Get available models."""
        return self.models

class MainDataSource(BaseJSONDataSource):
    """Main dataset loading strategy."""

    def __init__(self, data_root: str = DATASET_CONFIG['main']['root_dir']):
        self.data_root = data_root
        self.datasets = ["xsum", "wp", "arxiv"]
        self.source_models = ["gpt4o", "claude3.7", "gemini2.0"]

    def _load_json_file(self, file_path: str, max_samples: int = -1):
        """Load a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """Main dataset loading."""
        dataset = kwargs.get("dataset", "xsum")
        source_model = kwargs.get("source_model", "gpt4o")

        human_file = os.path.join(self.data_root, dataset, f"{dataset}_human.json")
        ai_file = os.path.join(self.data_root, dataset, f"{dataset}_{source_model}.json")

        human_data = self._load_json_file(human_file)
        ai_data = self._load_json_file(ai_file)

        human_texts = human_data.get("human_text", human_data)
        ai_texts = ai_data.get("machine_text", ai_data)

        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]
        return human_texts, ai_texts

    def get_available_domains(self) -> List[str]:
        return self.datasets

    def get_available_models(self) -> List[str]:
        return self.source_models

class RAIDAttackDataSource(DataSourceStrategy):
    """RAID Attack dataset loading strategy."""

    ATTACK_TYPES = [
        "none", "whitespace", "synonym", "perplexity_misspelling",
        "paraphrase", "homoglyph", "zero_width_space",
    ]

    def __init__(self, data_root: str = DATASET_CONFIG['raid_attack']['root_dir']):
        self.data_root = data_root

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        attack_type = kwargs.get("attack_type", "none")

        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unsupported attack_type: {attack_type}. Available: {self.ATTACK_TYPES}")

        human_file = os.path.join(self.data_root, attack_type, f"{attack_type}_human.json")
        ai_file = os.path.join(self.data_root, attack_type, f"{attack_type}_ai.json")

        if not os.path.exists(human_file):
            raise FileNotFoundError(f"Data file not found: {human_file}")
        if not os.path.exists(ai_file):
            raise FileNotFoundError(f"Data file not found: {ai_file}")

        with open(human_file, 'r', encoding='utf-8') as f:
            human_records = json.load(f)
        with open(ai_file, 'r', encoding='utf-8') as f:
            ai_records = json.load(f)

        human_texts = [r["generation"] for r in human_records]
        ai_texts = [r["generation"] for r in ai_records]

        min_length = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_length]
        ai_texts = ai_texts[:min_length]

        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]

        return human_texts, ai_texts

    def get_available_domains(self) -> List[str]:
        return self.ATTACK_TYPES

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class CReDDataSource(DataSourceStrategy):
    """CReD dataset loading strategy - multi-domain x multi-model, CSV format, paired by id."""

    DOMAINS = ["composition", "film_review", "news", "paper", "question_answer", "TC_news"]
    MODELS = ["claude-3.5-haiku", "deepseek-r1", "deepseek-v3", "doubao-1.5-pro",
              "gemini-2.5-flash", "gpt-3.5-turbo", "gpt-4o", "qwen-2.5", "qwen-3"]

    def __init__(self, data_root: str = DATASET_CONFIG['cred']['root_dir']):
        self.data_root = data_root

    def _load_csv(self, file_path: str) -> Dict[str, str]:
        """Load a CSV file, return {id: text} dict."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        id_to_text = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                id_to_text[row['id']] = row['text']
        return id_to_text

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        domain = kwargs.get("domain", "composition")
        model = kwargs.get("model", "gpt-4o")

        if domain not in self.DOMAINS:
            raise ValueError(f"Unsupported domain: {domain}. Available: {self.DOMAINS}")
        if model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}. Available: {self.MODELS}")

        human_file = os.path.join(self.data_root, f"main_experiment_{domain}", f"test_{domain}_human.csv")
        ai_file = os.path.join(self.data_root, f"main_experiment_{domain}", f"test_{domain}_{model}.csv")

        human_map = self._load_csv(human_file)
        ai_map = self._load_csv(ai_file)

        common_ids = sorted(set(human_map.keys()) & set(ai_map.keys()), key=int)
        human_texts = [human_map[i] for i in common_ids]
        ai_texts = [ai_map[i] for i in common_ids]

        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]

        return human_texts, ai_texts

    def get_available_domains(self) -> List[str]:
        return self.DOMAINS

    def get_available_models(self) -> List[str]:
        return self.MODELS


class DataLoader:
    """Unified data loader."""

    _strategies = {
        "m4": M4DataSource,
        "main": MainDataSource,
        "detectrl_multidomain": lambda **kwargs: DetectRLDataSource(dataset_type="multidomain", **kwargs),
        "detectrl_multillm": lambda **kwargs: DetectRLDataSource(dataset_type="multillm", **kwargs),
        "raid": RAIDDataSource,
        "text_attack": TextAttackDataSource,
        "realdet": RealDetDataSource,
        "raid_attack": RAIDAttackDataSource,
        "cred": CReDDataSource,
    }

    def __init__(self, data_source: str = "main", **kwargs):
        """
        Initialize the data loader.

        Args:
            data_source: Data source type ("m4", "main", "detectrl_multidomain", "detectrl_multillm", "realdet")
            **kwargs: Arguments passed to the data source strategy
        """
        if data_source not in self._strategies:
            raise ValueError(f"Unsupported data source: {data_source}. Available: {list(self._strategies.keys())}")

        self.data_source = data_source
        strategy_class = self._strategies[data_source]

        if callable(strategy_class) and not isinstance(strategy_class, type):
            self.strategy = strategy_class(**kwargs)
        else:
            self.strategy = strategy_class(**kwargs)

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Load data.

        Args:
            max_samples: Maximum number of samples, -1 for all
            **kwargs: Other arguments

        Returns:
            (human text list, AI text list)
        """
        human_texts, ai_texts = self.strategy.load(max_samples=max_samples, **kwargs)
        return data_wrapper(human_texts, ai_texts)

    def get_available_domains(self) -> List[str]:
        """Get available data domains."""
        return self.strategy.get_available_domains()

    def get_available_models(self) -> List[str]:
        """Get available models."""
        return self.strategy.get_available_models()

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """Register a new data source strategy."""
        cls._strategies[name] = strategy_class


def load_m4_data(max_samples: int = 1000, **kwargs):
    """M4 data loading convenience function."""
    loader = DataLoader("m4")
    return loader.load(max_samples=max_samples, **kwargs)

def load_detectrl_data(dataset_type: str = "multidomain", max_samples: int = -1, **kwargs):
    """DetectRL data loading convenience function."""
    loader = DataLoader(f"detectrl_{dataset_type}")
    return loader.load(max_samples=max_samples, **kwargs)

def load_raid_data(max_samples: int = 1000, **kwargs):
    """RAID"""
    loader = DataLoader("raid")
    return loader.load(max_samples=max_samples, **kwargs)

def load_realdet_data(max_samples: int = -1, **kwargs):
    """RealDet data loading convenience function."""
    loader = DataLoader("realdet")
    return loader.load(max_samples=max_samples, **kwargs)

def load_main_data(dataset: str, source_model: str, max_samples: int = -1):
    """Main dataset loading convenience function."""
    loader = DataLoader("main")
    return loader.load(dataset=dataset, source_model=source_model, max_samples=max_samples)

def load_text_attack_data(model: str = "GPT4", attack_type: str = "delete", max_samples: int = -1):
    """Text Attack data loading convenience function."""
    loader = DataLoader("text_attack")
    return loader.load(model=model, attack_type=attack_type, max_samples=max_samples)

def load_raid_attack_data(attack_type: str = "none", max_samples: int = -1):
    """RAID Attack data loading convenience function."""
    loader = DataLoader("raid_attack")
    return loader.load(attack_type=attack_type, max_samples=max_samples)

def load_cred_data(domain: str = "composition", model: str = "gpt-4o", max_samples: int = -1):
    """CReD data loading convenience function."""
    loader = DataLoader("cred")
    return loader.load(domain=domain, model=model, max_samples=max_samples)

def data_wrapper(human_texts: List, ai_texts: List):
    """Wrap human and AI texts into standard format"""
    data = {
        "original": human_texts,
        "sampled": ai_texts
    }
    n_samples = len(data["sampled"])
    return data, n_samples


def load_data(args):
    if args.data_source == 'main':
        data, n_samples = load_main_data(args.dataset, args.source_model, args.max_samples)
    elif args.data_source == 'm4':
        data, n_samples = load_m4_data(max_samples=args.max_samples)
    elif args.data_source in ['detectrl_multidomain', 'detectrl_multillm']:
        dataset_type = args.data_source.replace('detectrl_', '')
        data, n_samples = load_detectrl_data(dataset_type=dataset_type, max_samples=args.max_samples)
    elif args.data_source == 'raid':
        data, n_samples = load_raid_data(max_samples=args.max_samples)
    elif args.data_source == 'realdet':
        data, n_samples = load_realdet_data(max_samples=args.max_samples)
    elif args.data_source == 'text_attack':
        # Map source_model to Text_attack model names
        model_mapping = {
            'claude3.7': 'Claude',
            'gemini2.0': 'Gemini',
            'gpt4o': 'GPT4'
        }
        attack_model = model_mapping.get(args.source_model, 'GPT4')
        data, n_samples = load_text_attack_data(
            model=attack_model,
            attack_type=args.attack_type,
            max_samples=args.max_samples
        )
    elif args.data_source == 'raid_attack':
        data, n_samples = load_raid_attack_data(
            attack_type=args.raid_attack_type,
            max_samples=args.max_samples
        )
    elif args.data_source == 'cred':
        data, n_samples = load_cred_data(
            domain=args.cred_domain,
            model=args.cred_model,
            max_samples=args.max_samples
        )

    # Apply text truncation if max_words is specified
    if hasattr(args, 'max_words') and args.max_words is not None:
        from method.utils.text_processor import truncate_data_dict
        data = truncate_data_dict(data, args.max_words)
        print(f"[INFO] Texts truncated to {args.max_words} words for ablation study")

    return data, n_samples