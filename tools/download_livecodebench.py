#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download LiveCodeBench datasets for Verl evaluation.

Usage:
    python tools/download_livecodebench.py

Sources:
    1. ArcherCodeR pre-processed data (recommended)
    2. Convert from official LiveCodeBench

Output:
    - data/test/livecodebench_v5.parquet
    - data/test/livecodebench_v6.parquet (if available)
"""

import os
import json
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download


def download_from_archercoder():
    """Download pre-processed LiveCodeBench from ArcherCodeR dataset."""
    output_dir = "./data/test"
    os.makedirs(output_dir, exist_ok=True)
    
    repo_id = "wizardII/ArcherCodeR-Dataset"
    
    datasets = [
        ("test/livecodebench_v5.json", "livecodebench_v5"),
    ]
    
    for filename, name in datasets:
        try:
            print(f"Downloading {name} from {repo_id}...")
            file_path = hf_hub_download(
                repo_id=repo_id,
                repo_type='dataset',
                filename=filename
            )
            
            # Read JSON and convert to parquet
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Save as both JSON and parquet
            json_path = os.path.join(output_dir, f"{name}.json")
            parquet_path = os.path.join(output_dir, f"{name}.parquet")
            
            shutil.copyfile(file_path, json_path)
            df.to_parquet(parquet_path, index=False)
            
            print(f"  ✓ Saved to {json_path}")
            print(f"  ✓ Saved to {parquet_path}")
            print(f"  ✓ Total samples: {len(df)}")
            
            # Verify format
            verify_format(df, name)
            
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")


def verify_format(df: pd.DataFrame, name: str):
    """Verify the dataset has correct verl format."""
    print(f"\n🔍 Verifying {name} format...")
    
    required_columns = ["data_source", "prompt", "reward_model"]
    for col in required_columns:
        if col not in df.columns:
            print(f"  ✗ Missing column: {col}")
            return False
    
    # Check sample
    sample = df.iloc[0]
    print(f"  data_source: {sample['data_source']}")
    
    # Check prompt format
    prompt = sample['prompt']
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    print(f"  prompt type: {type(prompt)}")
    print(f"  prompt[0] keys: {prompt[0].keys() if isinstance(prompt, list) else 'N/A'}")
    
    # Check reward_model format
    rm = sample['reward_model']
    if isinstance(rm, str):
        rm = json.loads(rm)
    print(f"  reward_model keys: {rm.keys() if isinstance(rm, dict) else 'N/A'}")
    
    # Check ground_truth format
    gt = rm.get('ground_truth', '')
    if isinstance(gt, str):
        try:
            gt_parsed = json.loads(gt)
            print(f"  ground_truth: {len(gt_parsed)} test cases")
            if gt_parsed:
                print(f"  test case keys: {gt_parsed[0].keys()}")
        except:
            print(f"  ground_truth: {gt[:50]}...")
    
    print(f"  ✓ Format verification passed!")
    return True


def download_from_official_lcb():
    """
    Alternative: Download and convert from official LiveCodeBench.
    
    Note: This requires more processing as official LCB has different format.
    """
    print("\n📦 Attempting to download from official LiveCodeBench...")
    
    try:
        from datasets import load_dataset
        
        # Official LiveCodeBench dataset
        # Note: This may need adjustment based on actual HF repo structure
        dataset = load_dataset("livecodebench/livecodebench", split="test")
        
        print(f"  Loaded {len(dataset)} samples from official LCB")
        print(f"  Columns: {dataset.column_names}")
        
        # Convert to verl format
        verl_data = []
        for item in dataset:
            # Adjust field names based on actual LCB structure
            problem = item.get("question", item.get("problem", ""))
            test_cases = item.get("test_cases", item.get("public_tests", []))
            
            verl_item = {
                "data_source": "livecodebench_v5",
                "prompt": [{"role": "user", "content": problem}],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(test_cases) if isinstance(test_cases, list) else test_cases
                }
            }
            verl_data.append(verl_item)
        
        df = pd.DataFrame(verl_data)
        df.to_parquet("./data/test/livecodebench_v5.parquet", index=False)
        print(f"  ✓ Converted and saved {len(df)} samples")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Please use the ArcherCodeR pre-processed data instead.")


def main():
    print("=" * 60)
    print("LiveCodeBench Dataset Downloader for Verl")
    print("=" * 60)
    
    # Method 1: Download from ArcherCodeR (recommended)
    print("\n📦 Method 1: Download from ArcherCodeR (pre-processed)")
    download_from_archercoder()
    
    # Check if files exist
    lcb_v5_path = "./data/test/livecodebench_v5.parquet"
    if os.path.exists(lcb_v5_path):
        print(f"\n✓ {lcb_v5_path} is ready!")
        
        # Show sample
        df = pd.read_parquet(lcb_v5_path)
        print(f"\nSample data:")
        sample = df.iloc[0]
        prompt = sample['prompt']
        if isinstance(prompt, list):
            content = prompt[0].get('content', '')[:100]
        else:
            content = str(prompt)[:100]
        print(f"  prompt: {content}...")
    else:
        print(f"\n⚠️ {lcb_v5_path} not found. Trying official source...")
        download_from_official_lcb()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()




