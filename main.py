# main.py
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Import from our new modules
from src.utils import load_prompt_data_from_npz, export_split_responses
from src.feature_computation import process_and_compute_features, generate_responses # Assuming you rename the function
from src.evaluation import run_evaluation
from src.config import CONF
def main():
    parser = argparse.ArgumentParser(description="Modular MIA Baselines Script")
    parser = argparse.ArgumentParser(description="Extended MIA Baselines Script (v3.4)")
    gen_group = parser.add_argument_group('Full Run Arguments (Generation & Feature Computation)')
    gen_group.add_argument("--finetuned", help="Path to finetuned model checkpoint")
    gen_group.add_argument("--reference", help="Path to reference model checkpoint")
    gen_group.add_argument("--member_json", type=Path, help="JSONL file of member prompts")
    gen_group.add_argument("--nonmember_json", type=Path, help="JSONL file of non-member prompts")
    gen_group.add_argument("--label_m", help="JSON file of member labels")
    gen_group.add_argument("--label_nm", help="JSON file of non-member labels")
    gen_group.add_argument("--member", help="Parquet for member information")
    gen_group.add_argument("--nonmember", help="Parquet for member information")
    gen_group.add_argument("--temp", help="generation temp", type=float,default=CONF.temperature)
    gen_group.add_argument("--noise", help="dp noise", type=float,default=0.0)
    gen_group.add_argument("--out_member",help="Generated files",type=Path,default="responses_ft_member.jsonl")
    gen_group.add_argument("--out_nonmember",help="Generated files",type=Path,default="responses_ft_nonmember.jsonl")
    cache_group = parser.add_argument_group('Cached Run Arguments')
    cache_group.add_argument("--load_from_npz", type=Path, help="Path to metrics_prompt.npz to load for re-evaluation.")
    
    out_group = parser.add_argument_group('Output & Options')
    out_group.add_argument("--base_response", required=True, type=Path, help='dir for response of base model')
    out_group.add_argument("--out_dir", required=True, type=Path, help="Directory to save outputs and reports")
    out_group.add_argument("--export_split_responses", action='store_true', help="Export 'all-0' and 'all-1' responses to CSV files.")
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.load_from_npz:
        if not args.load_from_npz.exists():
            raise FileNotFoundError(f"Cache file not found: {args.load_from_npz}")
        prompt_data = load_prompt_data_from_npz(args.load_from_npz)
    else:
        # ... (Argument checking logic) ...
        required_for_full_run = ['finetuned', 'reference', 'member', 'nonmember']
        if not all(getattr(args, arg) for arg in required_for_full_run):
            parser.error("For a full run, all generation-related arguments are required.")
        out_dir = args.out_dir
        base_dir = args.base_response
        generate_responses(args.reference, args.member,base_dir/"responses_base_member.jsonl", args.temp, args.noise)
        generate_responses(args.finetuned, args.member, out_dir / args.out_member, args.temp, args.noise)
        generate_responses(args.reference, args.nonmember, base_dir/"responses_base_nonmember.jsonl", args.temp, args.noise)
        generate_responses(args.finetuned, args.nonmember, out_dir / args.out_nonmember, args.temp, args.noise)
        cache_path = out_dir / "computed_features.jsonl"
        if not cache_path.exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            if 'vl' not in args.reference.lower():
                models = {
                    "ft": AutoModelForCausalLM.from_pretrained(args.finetuned, torch_dtype=dtype, trust_remote_code=True).to(device).eval(),
                    "ref": AutoModelForCausalLM.from_pretrained(args.reference, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
                }
                tokenizers = {"ft": AutoTokenizer.from_pretrained(args.finetuned, trust_remote_code=True)}
            else:
                models = {
                    "ft": Qwen2_5_VLForConditionalGeneration.from_pretrained(args.finetuned, torch_dtype=dtype, trust_remote_code=True).to(device).eval(),
                    "ref": Qwen2_5_VLForConditionalGeneration.from_pretrained(args.reference, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
                }
                tokenizers = {"ft": AutoProcessor.from_pretrained(args.finetuned, trust_remote_code=True)}
        else:
            models,tokenizers = None, None
        print("\nInitializing models for feature computation...")
        # ... (Model and tokenizer loading logic) ...
        
        # This single function call runs the whole generation and feature pipeline
        prompt_data = process_and_compute_features(args, models, tokenizers)
        
        del models; gc.collect(); torch.cuda.empty_cache()

        # Save results to cache
        metrics_path = args.out_dir / "metrics_prompt.npz"
        np.savez(
            metrics_path,
            feats=np.array([p.to_feature_array() for p in prompt_data]),
            labels=np.array([p.label for p in prompt_data]),
            avg_base_scores=np.array([p.avg_base_score for p in prompt_data]),
            avg_ft_scores=np.array([p.avg_ft_score for p in prompt_data]),
            prompt_texts=np.array([p.prompt_text for p in prompt_data])
        )
        print(f"\nAggregated prompt-level metrics saved to {metrics_path}")

    # Run evaluation on the loaded or newly computed data
    print('Total data: ',len(prompt_data))

    run_evaluation(prompt_data, args.out_dir)

    # # Conditionally export responses for qualitative analysis
    # if args.export_split_responses:
    #     # Check that the required files for export exist
    #     required_files = [
    #         args.out_dir / "responses_base_member.jsonl",
    #         args.out_dir / "responses_ft_member.jsonl",
    #         args.out_dir / "responses_base_nonmember.jsonl",
    #         args.out_dir / "responses_ft_nonmember.jsonl"
    #     ]
    #     if not all(p.exists() for p in required_files):
    #         print("\nWARNING: Cannot export split responses. One or more response .jsonl files are missing.")
    #         print("Please run the script without --load_from_npz first to generate them.")
    #     else:
    #         response_paths = {
    #             'ft': [required_files[1], required_files[3]],
    #             'base': [required_files[0], required_files[2]]
    #         }
    #         export_split_responses(prompt_data, args.out_dir, response_paths)

    print("\nScript finished successfully.")

if __name__ == "__main__":
    main()