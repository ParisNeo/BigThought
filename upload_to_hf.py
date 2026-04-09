"""
BigThought HuggingFace Upload Utility
Author      : Saifeddine ALOUI
Description : Wrapper script to upload BigThought dataset and models to HuggingFace.
              This is a convenience wrapper around db_creator.py functionality.
"""

import sys
from pathlib import Path

# Import and re-export main functionality from db_creator
try:
    from db_creator import main as db_creator_main
    from db_creator import upload_to_huggingface, create_dataset_card, save_datasets, create_structured_dataset, create_augmented_dataset
except ImportError:
    print("❌ Error: db_creator.py not found in the same directory.")
    print("   Please ensure db_creator.py is available.")
    sys.exit(1)


def print_usage():
    """Print usage information."""
    print("""
BigThought HuggingFace Upload Utility
====================================

This script uploads the BigThought dataset to HuggingFace Hub.

Usage:
    python upload_to_hf.py --repo-id ID [options]

Required:
    --repo-id ID        HuggingFace repository ID
                        Format: username/repo-name
                        Example: ParisNeo/bigthought-dataset

Options:
    --token TOKEN       HuggingFace API token
                        Or set HF_TOKEN environment variable
    
    --private           Create private repository (default: public)
    
    --augment-factor N  Data augmentation factor (default: 10)
    
    --output-dir DIR    Local output directory (default: data/)

Examples:
    # Upload with token from environment
    export HF_TOKEN=your_token_here
    python upload_to_hf.py --repo-id ParisNeo/bigthought-dataset
    
    # Upload with explicit token
    python upload_to_hf.py --repo-id ParisNeo/bigthought-dataset --token hf_...
    
    # Create private dataset
    python upload_to_hf.py --repo-id ParisNeo/bigthought-dataset --private

For full control, use db_creator.py directly:
    python db_creator.py --upload --repo-id ParisNeo/bigthought-dataset
""")


def main():
    """Main entry point - delegates to db_creator with upload flag."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload BigThought dataset to HuggingFace Hub",
        add_help=False  # We'll handle help ourselves for cleaner output
    )
    
    parser.add_argument('--repo-id', type=str, default=None)
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--augment-factor', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--help', '-h', action='store_true')
    
    args, remaining = parser.parse_known_args()
    
    if args.help:
        print_usage()
        return 0
    
    # Check if repo-id is provided for upload mode
    if not args.repo_id:
        print("❌ Error: --repo-id is required for uploading to HuggingFace")
        print("   Example: --repo-id username/bigthought-dataset")
        print("\n" + "=" * 50)
        print("Creating local dataset files only (no upload)...")
        print("=" * 50)
        # Reconstruct args for db_creator without upload
        sys.argv = [sys.argv[0]] + remaining
        return db_creator_main()
    
    # Force upload mode with all required arguments
    new_argv = [
        sys.argv[0],
        '--upload',
        '--repo-id', args.repo_id,
        '--augment-factor', str(args.augment_factor),
        '--output-dir', str(args.output_dir)
    ]
    
    if args.token:
        new_argv.extend(['--token', args.token])
    if args.private:
        new_argv.append('--private')
    
    # Add any remaining arguments
    new_argv.extend(remaining)
    
    # Replace sys.argv and call db_creator
    sys.argv = new_argv
    
    print(f"🚀 Uploading to HuggingFace: {args.repo_id}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Augmentation factor: {args.augment_factor}")
    if args.private:
        print(f"   Visibility: private")
    else:
        print(f"   Visibility: public")
    print()
    
    return db_creator_main()


if __name__ == "__main__":
    sys.exit(main())