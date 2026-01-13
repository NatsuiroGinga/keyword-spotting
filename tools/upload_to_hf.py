#!/usr/bin/env python3
"""
Upload KWS system files to HuggingFace Hub.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import HfApi, get_token
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def upload_to_hf(
    repo_id: str,
    local_dir: str,
    patterns: Optional[List[str]] = None,
    dry_run: bool = False
) -> bool:
    """
    Upload KWS files to HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "username/streaming-kws")
        local_dir: Local directory to upload
        patterns: File patterns to include (default: all)
        dry_run: Just show what would be uploaded
    
    Returns:
        Success status
    """
    token = get_token()
    if not token:
        print("Error: HuggingFace token not found.")
        print("Please login with: huggingface-cli login")
        return False
    
    api = HfApi()
    local_path = Path(local_dir)
    
    if not local_path.exists():
        print(f"Error: Directory not found: {local_dir}")
        return False
    
    try:
        print(f"Uploading to repository: {repo_id}")
        print(f"Local directory: {local_path}")
        
        if dry_run:
            print("\n[DRY RUN] Files that would be uploaded:")
            for file_path in sorted(local_path.rglob("*")):
                if file_path.is_file() and not file_path.name.startswith("."):
                    rel_path = file_path.relative_to(local_path)
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  {rel_path} ({size_mb:.1f} MB)")
            return True
        
        print("\nUploading files...")
        repo_url = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            repo_type="model",
            token=token
        )
        
        print(f"\n✓ Upload completed successfully!")
        print(f"  Repository: {repo_url}")
        print(f"  Visit: https://huggingface.co/{repo_id}")
        
        return True
    
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload KWS system files to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (e.g., username/streaming-kws)"
    )
    parser.add_argument(
        "--local-dir",
        default="hf_upload",
        help="Local directory to upload (default: hf_upload)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    success = upload_to_hf(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
