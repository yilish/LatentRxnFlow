# flow_nerf_mvp/utils/experiment.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import datetime
import sys
import yaml
import wandb
import torch.nn as nn
import logging
import shutil


def setup_experiment(cfg: Dict[str, Any]) -> Path:
    """
    创建一个实验目录：experiments/<timestamp>_<name>/
    并把当前 config + 命令行保存进去。
    """
    base_dir = Path(cfg.get("experiment", {}).get("output_dir", "experiments"))
    base_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = cfg.get("experiment", {}).get("name", "unnamed")
    exp_dir = base_dir / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存当前 config
    with (exp_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 保存命令行
    with (exp_dir / "cmd.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    return exp_dir


def setup_logging(exp_dir: Path) -> None:
    """
    设置 logging：同时输出到控制台和 exp_dir/train.log
    """
    log_file = exp_dir / "train.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清掉已有 handler，避免重复
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 控制台日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


def save_model_summary(exp_dir: Path, model: nn.Module) -> None:
    with (exp_dir / "model.txt").open("w", encoding="utf-8") as f:
        f.write(str(model) + "\n")


def save_wandb_info(exp_dir: Path) -> None:
    run = wandb.run
    if run is None:
        return
    info = {
        "project": run.project,
        "entity": run.entity,
        "name": run.name,
        "id": run.id,
        "url": run.url,
    }
    text = "\n".join(f"{k}: {v}" for k, v in info.items())
    with (exp_dir / "wandb_run.txt").open("w", encoding="utf-8") as f:
        f.write(text + "\n")

def copytree_safe(src: Path, dst: Path):
    """Copy directory tree, compatible with Python 3.7."""
    if not dst.exists():
        shutil.copytree(src, dst)
        return

    # merge
    for item in src.iterdir():
        s = src / item.name
        d = dst / item.name
        if s.is_dir():
            copytree_safe(s, d)
        else:
            shutil.copy2(s, d)


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def snapshot_code(exp_dir: Path, root_dir: Path) -> None:
    """
    把 train_flow_nerf.py 和 models/* 拷贝到 EXP_DIR/code 下
    """
    code_dir = exp_dir / "code"
    code_dir.mkdir(exist_ok=True)

    # 1) 复制 train_flow_nerf.py
    src_train = root_dir / "train_flow_nerf.py"
    if src_train.is_file():
        shutil.copy2(src_train, code_dir / "train_flow_nerf.py")

    # 2) 复制 models 目录
    src_models = root_dir / "models"
    dst_models = code_dir / "models"
    if src_models.is_dir():
        copytree_safe(src_models, dst_models)

    # 3) 你想的话也可以顺便把 utils 备份
    src_utils = root_dir / "utils"
    dst_utils = code_dir / "utils"
    if src_utils.is_dir():
        copytree_safe(src_utils, dst_utils)