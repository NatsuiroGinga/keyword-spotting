#!/usr/bin/env python3
"""
可复用的KWS模型部署到HuggingFace技能脚本。
可用于自动化和标准化KWS系统的HuggingFace部署流程。
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess

try:
    from huggingface_hub import create_repo, get_token, snapshot_download, HfApi
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


class KWSDeploymentSkill:
    """KWS系统到HuggingFace的部署技能"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化部署技能"""
        self.config: Dict[str, Any] = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
    
    def validate_environment(self) -> bool:
        """验证环境和HuggingFace访问权限"""
        print("=" * 60)
        print("验证部署环境...")
        print("=" * 60)
        
        token = get_token()
        if not token:
            print("❌ HuggingFace令牌未找到")
            print("   运行: huggingface-cli login")
            return False
        
        print("✓ HuggingFace令牌有效")
        
        # 验证git lfs
        try:
            result = subprocess.run(
                ["git", "lfs", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✓ Git LFS已安装")
            else:
                print("⚠ Git LFS未安装（可选，但推荐）")
        except FileNotFoundError:
            print("⚠ Git未找到（可选，但推荐）")
        
        return True
    
    def prepare_upload_directory(
        self,
        source_models_dir: str,
        source_code_dir: str,
        keywords_file: str,
        tokens_file: str,
        output_dir: str = "hf_upload"
    ) -> bool:
        """准备上传目录"""
        print("\n" + "=" * 60)
        print("准备上传目录...")
        print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # 创建子目录
            (output_path / "kws_finetune_v3").mkdir(exist_ok=True)
            (output_path / "models").mkdir(exist_ok=True)
            (output_path / "src").mkdir(exist_ok=True)
            
            # 复制模型文件
            self._copy_files(
                source_dir=Path(source_models_dir),
                dest_dir=output_path / "kws_finetune_v3",
                patterns=["*.onnx", "*.txt"]
            )
            
            # 复制推理代码
            if Path(source_code_dir).exists():
                self._copy_directory(
                    source_dir=Path(source_code_dir),
                    dest_dir=output_path / "src"
                )
            
            print(f"✓ 文件已准备在: {output_dir}")
            self._list_directory(output_path)
            return True
        
        except Exception as e:
            print(f"❌ 准备目录失败: {e}")
            return False
    
    def create_repository(
        self,
        username: str,
        repo_name: str = "streaming-kws"
    ) -> Optional[str]:
        """创建HuggingFace私有仓库"""
        print("\n" + "=" * 60)
        print("创建HuggingFace仓库...")
        print("=" * 60)
        
        repo_id = f"{username}/{repo_name}"
        token = get_token()
        
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
                token=token
            )
            print(f"✓ 仓库已创建/存在: {repo_id}")
            print(f"  URL: {repo_url}")
            return repo_id
        
        except Exception as e:
            print(f"❌ 创建仓库失败: {e}")
            return None
    
    def upload_files(
        self,
        repo_id: str,
        local_dir: str = "hf_upload",
        dry_run: bool = False
    ) -> bool:
        """上传文件到HuggingFace"""
        print("\n" + "=" * 60)
        print("上传文件到HuggingFace...")
        print("=" * 60)
        
        api = HfApi()
        token = get_token()
        local_path = Path(local_dir)
        
        if not local_path.exists():
            print(f"❌ 目录不存在: {local_dir}")
            return False
        
        try:
            if dry_run:
                print("[DRY RUN] 将上传以下文件:")
                self._list_files(local_path)
                return True
            
            print(f"上传到: {repo_id}")
            repo_url = api.upload_folder(
                repo_id=repo_id,
                folder_path=str(local_path),
                repo_type="model",
                token=token
            )
            
            print(f"✓ 上传完成！")
            print(f"  仓库地址: https://huggingface.co/{repo_id}")
            return True
        
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return False
    
    def verify_upload(self, repo_id: str) -> bool:
        """验证上传的文件"""
        print("\n" + "=" * 60)
        print("验证上传...")
        print("=" * 60)
        
        api = HfApi()
        
        try:
            info = api.model_info(repo_id=repo_id, repo_type="model")
            print(f"✓ 仓库访问成功: {repo_id}")
            print(f"  最后更新: {info.last_modified}")
            print(f"  私有: {info.private}")
            
            # 列出文件
            files = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"\n  包含 {len(list(files))} 个文件:")
            for item in files:
                if hasattr(item, 'name'):
                    print(f"    - {item.path}")
            
            return True
        
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False
    
    def download_and_test(
        self,
        repo_id: str,
        local_dir: str = "./downloaded_kws"
    ) -> bool:
        """下载并测试模型"""
        print("\n" + "=" * 60)
        print("下载并测试模型...")
        print("=" * 60)
        
        try:
            print(f"下载模型从: {repo_id}")
            download_dir = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir
            )
            
            print(f"✓ 模型已下载到: {download_dir}")
            
            # 基本检查
            required_files = [
                "kws_finetune_v3/encoder.int8.onnx",
                "kws_finetune_v3/decoder.int8.onnx",
                "kws_finetune_v3/joiner.int8.onnx",
                "models/mlp_verifier.onnx",
                "README.md"
            ]
            
            print("\n检查必需文件:")
            missing = []
            for file in required_files:
                file_path = Path(download_dir) / file
                if file_path.exists():
                    size = file_path.stat().st_size
                    size_mb = size / (1024 * 1024)
                    print(f"  ✓ {file} ({size_mb:.1f} MB)")
                else:
                    print(f"  ✗ {file} (缺失)")
                    missing.append(file)
            
            if missing:
                print(f"\n⚠ 缺失 {len(missing)} 个文件")
                return False
            
            print("\n✓ 所有文件完整！")
            return True
        
        except Exception as e:
            print(f"❌ 下载/测试失败: {e}")
            return False
    
    def full_deployment(
        self,
        username: str,
        repo_name: str = "streaming-kws",
        source_models_dir: str = "exp/kws_finetune_v3",
        source_code_dir: str = "src",
        keywords_file: str = "exp/kws_finetune_v3/keywords.txt",
        tokens_file: str = "exp/kws_finetune_v3/tokens.txt",
        dry_run: bool = False
    ) -> bool:
        """执行完整部署流程"""
        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║" + "  KWS系统 HuggingFace部署技能".center(58) + "║")
        print("║" + " " * 58 + "║")
        print("╚" + "=" * 58 + "╝")
        
        # 验证环境
        if not self.validate_environment():
            return False
        
        # 准备目录
        if not self.prepare_upload_directory(
            source_models_dir=source_models_dir,
            source_code_dir=source_code_dir,
            keywords_file=keywords_file,
            tokens_file=tokens_file
        ):
            return False
        
        # 创建仓库
        repo_id = self.create_repository(username, repo_name)
        if not repo_id:
            return False
        
        # 上传文件
        if not self.upload_files(repo_id, dry_run=dry_run):
            return False
        
        # 验证
        if not dry_run and not self.verify_upload(repo_id):
            return False
        
        print("\n" + "=" * 60)
        print("✓ 部署完成！")
        print("=" * 60)
        print(f"\n仓库地址: https://huggingface.co/{repo_id}")
        print(f"私有状态: 🔐 私有")
        
        return True
    
    @staticmethod
    def _copy_files(source_dir: Path, dest_dir: Path, patterns: list):
        """复制匹配模式的文件"""
        for pattern in patterns:
            for file in source_dir.glob(pattern):
                if file.is_file():
                    import shutil
                    shutil.copy2(file, dest_dir / file.name)
    
    @staticmethod
    def _copy_directory(source_dir: Path, dest_dir: Path):
        """递归复制目录"""
        import shutil
        if source_dir.exists():
            for item in source_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(
                        item,
                        dest_dir / item.name,
                        dirs_exist_ok=True
                    )
                elif item.is_file():
                    shutil.copy2(item, dest_dir / item.name)
    
    @staticmethod
    def _list_directory(path: Path, indent: int = 0):
        """列出目录结构"""
        for item in sorted(path.iterdir()):
            prefix = "  " * indent + "├─ "
            if item.is_dir():
                print(f"{prefix}{item.name}/")
                KWSDeploymentSkill._list_directory(item, indent + 1)
            else:
                size = item.stat().st_size
                size_kb = size / 1024
                print(f"{prefix}{item.name} ({size_kb:.0f}KB)")
    
    @staticmethod
    def _list_files(path: Path, indent: int = 0):
        """列出所有文件"""
        for item in sorted(path.rglob("*")):
            if item.is_file() and not item.name.startswith("."):
                rel_path = item.relative_to(path)
                size = item.stat().st_size / (1024 * 1024)
                print(f"  {rel_path} ({size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="KWS系统HuggingFace部署技能"
    )
    
    parser.add_argument(
        "--action",
        choices=["full", "prepare", "create", "upload", "verify", "download"],
        default="full",
        help="执行的操作"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="HuggingFace用户名"
    )
    parser.add_argument(
        "--repo-name",
        default="streaming-kws",
        help="仓库名称"
    )
    parser.add_argument(
        "--source-models",
        default="exp/kws_finetune_v3",
        help="源模型目录"
    )
    parser.add_argument(
        "--source-code",
        default="src",
        help="源代码目录"
    )
    parser.add_argument(
        "--output-dir",
        default="hf_upload",
        help="上传目录"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式"
    )
    parser.add_argument(
        "--download-dir",
        default="./downloaded_kws",
        help="下载目录"
    )
    
    args = parser.parse_args()
    
    skill = KWSDeploymentSkill()
    repo_id = f"{args.username}/{args.repo_name}"
    
    actions = {
        "full": lambda: skill.full_deployment(
            username=args.username,
            repo_name=args.repo_name,
            source_models_dir=args.source_models,
            source_code_dir=args.source_code,
            dry_run=args.dry_run
        ),
        "prepare": lambda: skill.prepare_upload_directory(
            source_models_dir=args.source_models,
            source_code_dir=args.source_code,
            keywords_file=f"{args.source_models}/keywords.txt",
            tokens_file=f"{args.source_models}/tokens.txt",
            output_dir=args.output_dir
        ),
        "create": lambda: skill.create_repository(
            username=args.username,
            repo_name=args.repo_name
        ) is not None,
        "upload": lambda: skill.upload_files(
            repo_id=repo_id,
            local_dir=args.output_dir,
            dry_run=args.dry_run
        ),
        "verify": lambda: skill.verify_upload(repo_id),
        "download": lambda: skill.download_and_test(
            repo_id=repo_id,
            local_dir=args.download_dir
        )
    }
    
    success = actions[args.action]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
