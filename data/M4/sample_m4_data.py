#!/usr/bin/env python3
"""
从聚合的M4数据中随机采样指定数量的样本，生成最终的训练/测试数据集

使用方法:
    python sample_m4_data.py --input_dir ../m4_aggregated_data --sample_size 1000 --output_dir ./final_data
"""

import os
import json
import random
import argparse
from pathlib import Path


def load_json_data(filepath):
    """
    加载JSON文件数据

    Args:
        filepath: JSON文件路径

    Returns:
        list: 加载的数据列表
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list data in {filepath}, got {type(data)}")

    return data


def random_sample(data, sample_size, seed=None):
    """
    随机采样指定数量的样本

    Args:
        data: 原始数据列表
        sample_size: 采样数量
        seed: 随机种子

    Returns:
        list: 采样后的数据
    """
    if seed is not None:
        random.seed(seed)

    # 如果样本数量大于等于数据量，返回全部数据
    if sample_size >= len(data):
        print(f"[WARNING] Requested sample_size ({sample_size}) >= available data ({len(data)}), using all data")
        return data.copy()

    # 随机采样
    sampled_data = random.sample(data, sample_size)
    return sampled_data


def sample_m4_data(input_dir, sample_size=1000, output_dir="./final_m4_data", seed=42,
                  human_filename="all_human_text.json", ai_filename="all_ai_text.json"):
    """
    从聚合的M4数据中采样

    Args:
        input_dir: 输入目录（包含聚合的JSON文件）
        sample_size: 每类采样的样本数量
        output_dir: 输出目录
        seed: 随机种子
        human_filename: 人类文本文件名
        ai_filename: AI文本文件名

    Returns:
        dict: 采样统计信息
    """
    print(f"[INFO] Starting M4 data sampling...")
    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Sample size per category: {sample_size}")
    print(f"[INFO] Random seed: {seed}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建输入文件路径
    human_input_path = os.path.join(input_dir, human_filename)
    ai_input_path = os.path.join(input_dir, ai_filename)

    # 加载数据
    print(f"[INFO] Loading human text from: {human_input_path}")
    human_data = load_json_data(human_input_path)
    print(f"[INFO] Loaded {len(human_data)} human text samples")

    print(f"[INFO] Loading AI text from: {ai_input_path}")
    ai_data = load_json_data(ai_input_path)
    print(f"[INFO] Loaded {len(ai_data)} AI text samples")

    # 随机采样
    print(f"[INFO] Sampling {sample_size} human text samples...")
    human_sampled = random_sample(human_data, sample_size, seed)

    print(f"[INFO] Sampling {sample_size} AI text samples...")
    ai_sampled = random_sample(ai_data, sample_size, seed + 1)  # 使用不同种子

    # 构建输出文件路径
    human_output_path = os.path.join(output_dir, "m4_human.json")
    ai_output_path = os.path.join(output_dir, "m4_machine.json")

    # 保存采样结果
    print(f"[INFO] Saving human samples to: {human_output_path}")
    with open(human_output_path, 'w', encoding='utf-8') as f:
        json.dump(human_sampled, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saving AI samples to: {ai_output_path}")
    with open(ai_output_path, 'w', encoding='utf-8') as f:
        json.dump(ai_sampled, f, ensure_ascii=False, indent=2)

    # 生成统计信息
    stats = {
        'input_stats': {
            'human_samples_total': len(human_data),
            'ai_samples_total': len(ai_data),
            'input_dir': input_dir
        },
        'sampling_stats': {
            'human_samples_sampled': len(human_sampled),
            'ai_samples_sampled': len(ai_sampled),
            'requested_sample_size': sample_size,
            'random_seed': seed
        },
        'output_stats': {
            'human_file': human_output_path,
            'ai_file': ai_output_path,
            'human_file_size_bytes': os.path.getsize(human_output_path),
            'ai_file_size_bytes': os.path.getsize(ai_output_path),
            'output_dir': output_dir
        }
    }

    # 保存统计信息
    stats_path = os.path.join(output_dir, "sampling_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] === Sampling Summary ===")
    print(f"[INFO] Human samples: {len(human_sampled)}/{len(human_data)} -> {human_output_path}")
    print(f"[INFO] AI samples: {len(ai_sampled)}/{len(ai_data)} -> {ai_output_path}")
    print(f"[INFO] Stats saved to: {stats_path}")

    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sample from aggregated M4 data")
    parser.add_argument("--input_dir", type=str, default="/data1/wujunxi/kailin/m4_test/exp",
                       help="Input directory containing aggregated JSON files")
    parser.add_argument("--sample_size", type=int, default=1000,
                       help="Number of samples to extract from each category")
    parser.add_argument("--output_dir", type=str, default="/data1/wujunxi/kailin/m4_test/exp",
                       help="Output directory for sampled data")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducible sampling")
    parser.add_argument("--human_file", type=str, default="all_human_text.json",
                       help="Human text input filename")
    parser.add_argument("--ai_file", type=str, default="all_ai_text.json",
                       help="AI text input filename")

    args = parser.parse_args()

    # 执行采样
    try:
        stats = sample_m4_data(
            input_dir=args.input_dir,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            seed=args.seed,
            human_filename=args.human_file,
            ai_filename=args.ai_file
        )
        print(f"\n[INFO] Sampling completed successfully!")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Sampling failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())