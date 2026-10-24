"""
Download and prepare the BillSum dataset for fine-tuning and evaluation.
BillSum contains US Congressional bills with human-written summaries.
"""

import json
from pathlib import Path
from datasets import load_dataset


PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are an expert legislative analyst and summarizer. "
    "Provide a concise, accurate, and structured summary of the congressional bill below.<|im_end|>\n"
    "<|im_start|>user\n"
    "Summarize this bill:\n\n{text}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{summary}<|im_end|>"
)


def prepare_billsum(
    train_samples: int = 1000,
    test_samples: int = 100,
    output_dir: str = "data/billsum",
):
    print(f"[*] Downloading BillSum dataset from Hugging Face...")
    ds = load_dataset("billsum")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and sample
    train_subset = ds["train"].shuffle(seed=42).select(range(min(train_samples, len(ds["train"]))))
    test_subset = ds["test"].shuffle(seed=42).select(range(min(test_samples, len(ds["test"]))))
    
    train_records = []
    for item in train_subset:
        formatted_prompt = PROMPT_TEMPLATE.format(
            text=item["text"][:3500],  # Bound text length for standard context
            summary=item["summary"]
        )
        train_records.append({
            "text": item["text"][:3500],
            "title": item.get("title", ""),
            "target_summary": item["summary"],
            "formatted_text": formatted_prompt
        })
        
    test_records = []
    for item in test_subset:
        test_records.append({
            "text": item["text"][:3500],
            "title": item.get("title", ""),
            "target_summary": item["summary"]
        })
        
    train_file = out_path / "train.jsonl"
    test_file = out_path / "test.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
            
    with open(test_file, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"[+] Prepared {len(train_records)} training samples -> {train_file}")
    print(f"[+] Prepared {len(test_records)} evaluation samples -> {test_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download BillSum dataset")
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="data/billsum")
    args = parser.parse_args()
    
    prepare_billsum(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        output_dir=args.out_dir
    )
