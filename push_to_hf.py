"""
BigThought Push to HuggingFace
Author      : Saifeddine ALOUI
Description : Simple, explicit script to push the BigThought dataset to HuggingFace Hub.
              This is the recommended way to upload your database.

Usage:
    python push_to_hf.py --repo-id YOUR_USERNAME/bigthought-dataset [--token TOKEN]

Environment:
    Set HF_TOKEN environment variable to avoid passing --token
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from db_creator import (
        create_structured_dataset,
        create_augmented_dataset,
        save_datasets,
        upload_to_huggingface,
        create_dataset_card,
        HF_AVAILABLE
    )
except ImportError as e:
    print(f"❌ Error: Could not import from db_creator.py: {e}")
    sys.exit(1)


def main():
    """Push dataset to HuggingFace Hub."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Push BigThought dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using environment variable (recommended)
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
    python push_to_hf.py --repo-id ParisNeo/bigthought-dataset
    
    # With explicit token
    python push_to_hf.py --repo-id ParisNeo/bigthought-dataset --token hf_xxx
    
    # Private repository
    python push_to_hf.py --repo-id ParisNeo/bigthought-dataset --private
        """
    )
    
    parser.add_argument(
        '--repo-id', '-r',
        type=str,
        required=True,
        help='HuggingFace repository ID (e.g., username/bigthought-dataset)'
    )
    
    parser.add_argument(
        '--token', '-t',
        type=str,
        default=None,
        help='HuggingFace API token (or set HF_TOKEN env var)'
    )
    
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repository'
    )
    
    parser.add_argument(
        '--augment-factor', '-a',
        type=int,
        default=10,
        help='Data augmentation factor (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('data'),
        help='Local output directory (default: data/)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output for debugging'
    )
    
    args = parser.parse_args()
    
    # Check HF availability
    if not HF_AVAILABLE:
        print("❌ huggingface-hub not installed.")
        print("   Run: pip install huggingface-hub")
        sys.exit(1)
    
    # Validate token
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("❌ No HuggingFace token found.")
        print("   Options:")
        print("   1. Set HF_TOKEN environment variable")
        print("   2. Pass --token argument")
        print("   Get your token: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Validate repo_id format
    if '/' not in args.repo_id:
        print(f"❌ Invalid repo_id: '{args.repo_id}'")
        print("   Expected format: 'username/repo-name'")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 BigThought Dataset Push to HuggingFace")
    print("=" * 60)
    print(f"Repository: {args.repo_id}")
    print(f"Private: {'Yes' if args.private else 'No'}")
    print(f"Augmentation factor: {args.augment_factor}")
    print(f"Output directory: {args.output_dir}")
    if args.verbose:
        print(f"Token: {token[:10]}...{token[-4:]}")
    print()
    
    # Create datasets
    print("📚 Creating structured dataset...")
    base_dataset = create_structured_dataset()
    print(f"   Base entries: {len(base_dataset)}")
    
    print(f"\n🔧 Augmenting data (factor={args.augment_factor})...")
    augmented_dataset = create_augmented_dataset(base_dataset, factor=args.augment_factor)
    print(f"   Augmented entries: {len(augmented_dataset)}")
    
    # Show statistics
    languages = {}
    categories = {}
    for entry in base_dataset:
        languages[entry['language']] = languages.get(entry['language'], 0) + 1
        categories[entry['category']] = categories.get(entry['category'], 0) + 1
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Languages: {languages}")
    print(f"   Categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")
    
    # Save locally
    print(f"\n💾 Saving datasets to: {args.output_dir}")
    file_paths = save_datasets(args.output_dir, base_dataset, augmented_dataset)
    
    # Create dataset card
    card_path = create_dataset_card(args.output_dir, args.repo_id, base_dataset, augmented_dataset)
    file_paths.append(card_path)
    
    # Verify files exist
    print(f"\n🔍 Verifying files to upload:")
    for fp in file_paths:
        status = "✅" if fp.exists() else "❌ MISSING"
        print(f"   {status} {fp.name} ({fp.stat().st_size if fp.exists() else 0} bytes)")
    
    # Upload to HuggingFace
    print(f"\n📤 Uploading to HuggingFace Hub...")
    print(f"   Target: https://huggingface.co/datasets/{args.repo_id}")
    
    success = upload_to_huggingface(
        file_paths=file_paths,
        repo_id=args.repo_id,
        token=token,
        private=args.private
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✨ SUCCESS! Your dataset is now on HuggingFace Hub.")
        print(f"   URL: https://huggingface.co/datasets/{args.repo_id}")
        print(f"\nUse it in your code:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{args.repo_id}', split='train')")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ Upload failed. Check errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())