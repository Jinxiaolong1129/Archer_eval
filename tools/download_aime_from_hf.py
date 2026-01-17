#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download AIME 2024/2025 datasets from HuggingFace and convert to Verl format.

Usage:
    conda activate archer
    cd /scratch/jin509/self_RL/Archer2.0
    python tools/download_aime_from_hf.py

Sources:
    - math-ai/aime24 (AIME 2024, 30 problems)
    - math-ai/aime25 (AIME 2025, 30 problems)

Output:
    - data/test/aime2024.parquet
    - data/test/aime2024.json
    - data/test/aime2025.parquet
    - data/test/aime2025.json
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "./data/test"

# System prompt appended to each problem
MATH_SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""

# Dataset sources on HuggingFace
DATASETS = {
    "aime2024": {
        "hf_repo": "math-ai/aime24",
        "split": "test",
        "problem_field": "problem",
        "answer_field": "solution",  # Contains \boxed{...}
    },
    "aime2025": {
        "hf_repo": "math-ai/aime25",
        "split": "test",
        "problem_field": "problem",
        "answer_field": "answer",  # Direct answer (no boxed)
    },
}


# ============================================================
# Helper Functions
# ============================================================

def extract_answer(solution: str) -> str:
    """
    Extract answer from \\boxed{...} format.
    If no boxed format found, return the original string.
    
    Examples:
        "\\boxed{204}" -> "204"
        "204" -> "204"
    """
    if solution is None:
        return ""
    solution = str(solution)
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    if match:
        return match.group(1).strip()
    return solution.strip()


def create_verl_format(problem: str, answer: str, data_source: str) -> dict:
    """
    Convert a single problem to Verl format.
    
    Args:
        problem: The math problem text
        answer: The correct answer (will be extracted if in boxed format)
        data_source: Dataset identifier (e.g., "aime2024")
    
    Returns:
        Dictionary in Verl format:
        {
            "data_source": "aime2024",
            "prompt": [{"role": "user", "content": "..."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "204"}
        }
    """
    answer_str = extract_answer(answer)
    
    prompt = [
        {"role": "user", "content": f"{problem}\n\n{MATH_SYSTEM_PROMPT}"}
    ]
    
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer_str
        }
    }


def download_and_convert(data_source: str, config: dict, output_dir: str) -> pd.DataFrame:
    """
    Download dataset from HuggingFace and convert to Verl format.
    
    Args:
        data_source: Dataset name (e.g., "aime2024")
        config: Dataset configuration dict
        output_dir: Output directory path
    
    Returns:
        Converted DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing {data_source}")
    print(f"{'='*60}")
    
    # Download from HuggingFace
    print(f"Downloading from {config['hf_repo']}...")
    ds = load_dataset(config['hf_repo'], split=config['split'])
    print(f"  Loaded {len(ds)} samples")
    print(f"  Columns: {ds.column_names}")
    
    # Convert to Verl format
    verl_data = []
    for item in ds:
        problem = item[config['problem_field']]
        answer = item[config['answer_field']]
        
        verl_item = create_verl_format(problem, answer, data_source)
        verl_data.append(verl_item)
    
    # Create DataFrame
    df = pd.DataFrame(verl_data)
    
    # Save as parquet
    parquet_path = os.path.join(output_dir, f"{data_source}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Saved to {parquet_path}")
    
    # Save as JSON (backup)
    json_path = os.path.join(output_dir, f"{data_source}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(verl_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to {json_path}")
    
    return df


def verify_dataset(df: pd.DataFrame, data_source: str) -> bool:
    """
    Verify the dataset has correct Verl format.
    
    Args:
        df: DataFrame to verify
        data_source: Expected data_source value
    
    Returns:
        True if valid, False otherwise
    """
    print(f"\n🔍 Verifying {data_source}...")
    
    # Check columns
    required_cols = ["data_source", "prompt", "ability", "reward_model"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  ✗ Missing column: {col}")
            return False
    
    # Check sample
    sample = df.iloc[0]
    
    # Verify data_source
    if sample['data_source'] != data_source:
        print(f"  ✗ Wrong data_source: {sample['data_source']}")
        return False
    
    # Verify prompt format
    prompt = sample['prompt']
    if isinstance(prompt, (list, tuple)):
        prompt = list(prompt)
    if not isinstance(prompt, list) or len(prompt) == 0:
        print(f"  ✗ Invalid prompt format")
        return False
    
    # Verify reward_model
    rm = sample['reward_model']
    if not isinstance(rm, dict) or 'ground_truth' not in rm:
        print(f"  ✗ Invalid reward_model format")
        return False
    
    print(f"  ✓ Format valid!")
    print(f"  ✓ Total samples: {len(df)}")
    print(f"  ✓ Sample ground_truth: {rm['ground_truth']}")
    
    return True


def show_all_answers(df: pd.DataFrame, data_source: str):
    """Display all answers for verification."""
    print(f"\n📊 All answers for {data_source}:")
    for i, row in df.iterrows():
        gt = row['reward_model']['ground_truth']
        print(f"  Problem {i+1:2d}: {gt}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("AIME Dataset Downloader for Verl")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each dataset
    results = {}
    for data_source, config in DATASETS.items():
        try:
            df = download_and_convert(data_source, config, OUTPUT_DIR)
            verify_dataset(df, data_source)
            show_all_answers(df, data_source)
            results[data_source] = len(df)
        except Exception as e:
            print(f"  ✗ Error processing {data_source}: {e}")
            results[data_source] = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for ds, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {ds}: {count} samples")
    
    print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()




