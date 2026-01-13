#!/usr/bin/env python3
"""
Create HuggingFace private model repository for streaming KWS system.
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import create_repo, get_token
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def create_kws_repo(username: str, repo_name: str = "streaming-kws"):
    """
    Create a private HuggingFace Model repository for KWS system.
    
    Args:
        username: HuggingFace username
        repo_name: Repository name (default: streaming-kws)
    
    Returns:
        Repository URL
    """
    repo_id = f"{username}/{repo_name}"
    
    try:
        token = get_token()
        if not token:
            print("Error: HuggingFace token not found.")
            print("Please login with: huggingface-cli login")
            sys.exit(1)
        
        print(f"Creating private repository: {repo_id}")
        
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,  # Don't fail if repo already exists
            token=token
        )
        
        print(f"✓ Repository created successfully!")
        print(f"  Repository URL: {repo_url}")
        print(f"  Clone URL: https://huggingface.co/{repo_id}")
        
        return repo_url
    
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create HuggingFace private model repository for KWS system"
    )
    parser.add_argument("--username", required=True, help="HuggingFace username")
    parser.add_argument("--repo-name", default="streaming-kws", help="Repository name")
    
    args = parser.parse_args()
    
    create_kws_repo(args.username, args.repo_name)
