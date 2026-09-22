"""Publish only site/ to a user-owned Hugging Face Static Space after HF login."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True,help='Your HF username/space-name')
    args=parser.parse_args()
    api=HfApi()
    api.whoami()
    api.create_repo(repo_id=args.repo,repo_type='space',space_sdk='static',exist_ok=True)
    api.upload_folder(repo_id=args.repo,repo_type='space',folder_path=Path(__file__).resolve().parents[1]/'site',commit_message='Publish STAT6207 assignment results')
    print(f'https://huggingface.co/spaces/{args.repo}')
