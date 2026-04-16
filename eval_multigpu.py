from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader
import pickle
from utils.encoder_utils import load_checkpoint
from utils.experiment import (
    setup_experiment,
    setup_logging,
    load_config,
)
import logging
import wandb
from utils.data_utils import evaluate_smiles
import warnings
warnings.filterwarnings("ignore")
FLOW_ROOT = Path(__file__).resolve().parent
import sys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

if str(FLOW_ROOT) not in sys.path:
    sys.path.append(str(FLOW_ROOT))

from data.uspto_main_product import USPTOReact2MainProduct, collate_fn
from models.flow_nerf_model import FlowNERFModel, SimpleArgs, DecoderConfig
from utils.seed import set_seed

from contextlib import contextmanager
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def unwrap_model(m):
    return m.module if hasattr(m, "module") else m

def init_distributed():
    """
    用 torchrun 启动时，环境变量会自动给：
      RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" not in os.environ:
        return False, 0, 1, 0  # not distributed

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)
    dist.barrier()
    return True, rank, world_size, local_rank

def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

# DecoderConfig is imported from models.flow_nerf_model

def build_eval_dataloader(
    pickle_path: str,
    batch_size: int,
    ddp_on: bool,
    num_workers: int = 4,
) -> DataLoader:
    """
    从pickle文件加载数据并创建评估DataLoader
    """
    rank = get_rank()
    world_size = get_world_size()
    
    # 加载pickle数据
    with open(pickle_path, "rb") as f:
        data_list = pickle.load(f)
    
    if rank == 0:
        logging.info(f"[Data] Loaded {len(data_list)} reactions from {pickle_path}")
    
    # 创建数据集
    eval_dataset = USPTOReact2MainProduct(
        data_list=data_list,
        if_shuffle=False,
    )
    
    # 创建sampler（如果使用DDP）
    eval_sampler = None
    if ddp_on:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    
    # 创建DataLoader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    if rank == 0:
        logging.info(f"[Data] Eval dataset size = {len(eval_dataset)}, "
                    f"DataLoader batches = {len(eval_loader)}")
    
    return eval_loader

def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-NERF 多GPU评估脚本")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="配置文件路径（yaml格式，需包含checkpoint_path和eval.pickle_path）",
    )
    parser.add_argument("--ddp", action="store_true", help="use DDP (torchrun)")
    args = parser.parse_args()
    
    # 初始化分布式
    ddp_on, rank, world_size, local_rank = init_distributed()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 验证必需的配置项
    if "eval" not in cfg:
        raise ValueError("配置文件中必须包含 'eval' 部分")
    if "checkpoint_path" not in cfg["eval"]:
        raise ValueError("配置文件中 eval.checkpoint_path 必须指定checkpoint路径")
    if "pickle_path" not in cfg["eval"]:
        raise ValueError("配置文件中 eval.pickle_path 必须指定数据集pickle路径")
    
    # 设置实验目录和日志（仅在主进程）
    if is_main_process():
        set_seed(cfg.get("train", {}).get("seed", 42))
        
        # 创建实验目录
        exp_dir = setup_experiment(cfg)
        
        # 设置日志
        setup_logging(exp_dir)
        logging.info(f"Evaluation experiment dir: {exp_dir}")
        logging.info(f"Checkpoint path: {cfg['eval']['checkpoint_path']}")
        logging.info(f"Dataset pickle path: {cfg['eval']['pickle_path']}")
    else:
        exp_dir = None
    
    if ddp_on:
        dist.barrier()
    
    # 初始化wandb（所有进程都需要初始化，因为evaluate_smiles中的可视化函数会调用wandb.log）
    # 主进程使用正常模式，非主进程使用disabled模式
    if is_main_process():
        wandb_mode = "online" if cfg.get("eval", {}).get("use_wandb", True) else "disabled"
    else:
        wandb_mode = "disabled"  # 非主进程禁用wandb，避免重复记录
    
    wandb.init(
        project=cfg.get("experiment", {}).get("project", "flow-nerf-eval"),
        name=cfg.get("experiment", {}).get("name", "flow-nerf-eval"),
        config=cfg,
        mode=wandb_mode,
    )
    
    # 设备设置
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 构建数据加载器
    eval_batch_size = cfg.get("eval", {}).get("batch_size", 128)
    num_workers = cfg.get("data", {}).get("num_workers", 4)
    eval_loader = build_eval_dataloader(
        pickle_path=cfg["eval"]["pickle_path"],
        batch_size=eval_batch_size,
        ddp_on=ddp_on,
        num_workers=num_workers,
    )
    
    # 构建模型
    args_nn = SimpleArgs()
    
    # Set model config attributes for flow head selection
    class ModelConfig:
        pass
    args_nn.model = ModelConfig()
    model_cfg = cfg.get("model", {})
    args_nn.model.flow_cond_head = model_cfg.get("flow_cond_head", "controlnet")
    args_nn.model.film_hidden_dim = model_cfg.get("film_hidden_dim", model_cfg.get("latent_dim", 256) * 2)
    args_nn.model.film_init_zero = model_cfg.get("film_init_zero", True)
    
    dec = cfg.get("model", {}).get("decoder", {}) or {}
    
    decoder_cfg = DecoderConfig(
        delta_source=dec.get("delta_source", "tf"),
        input_mode=dec.get("input_mode", "fuse"),
        ode_method=dec.get("ode_method", "heun"),
    )
    
    model = FlowNERFModel(
        latent_dim=cfg["model"]["latent_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
        ntoken=cfg["model"].get("ntoken", 128),
        args=args_nn,
        flow_weight=cfg["model"].get("flow_weight", 1e-2),
        detach_encoder_for_flow=cfg["model"].get("detach_encoder_for_flow", True),
        flow_sampling_cfg=cfg["model"].get("flow_sampling_cfg", None),
        fm_sigma=cfg["model"].get("fm_sigma", 0.0),
        decoder_cfg=decoder_cfg,
        use_conditional_flow=cfg["model"].get("use_conditional_flow", False),
        nfe=dec.get("nfe", 20),
    ).to(device)
    
    # 包装为DDP（如果启用）
    if ddp_on:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    if is_main_process():
        base = unwrap_model(model)
        logging.info(f"[Init] decoder_cfg = delta_source={base.decoder_cfg.delta_source}, "
                    f"input_mode={base.decoder_cfg.input_mode}")
        logging.info(f"[Init] flow_sampling_cfg = {base.flow_sampling_cfg}")
    
    # 加载checkpoint
    checkpoint_cfg = {"checkpoint_path": cfg["eval"]["checkpoint_path"]}
    if is_main_process():
        logging.info(f"[Checkpoint] Loading from {checkpoint_cfg['checkpoint_path']}")
    
    # 加载checkpoint到模型
    load_checkpoint(unwrap_model(model), checkpoint_cfg)
    
    if ddp_on:
        dist.barrier()
    
    # 设置评估模式
    model.eval()
    
    # 获取评估配置
    eval_modes = cfg.get("eval", {}).get("decoder_eval_modes", ["tf"])
    max_eval_batches = cfg.get("eval", {}).get("max_batches", None)
    temperature = cfg.get("eval", {}).get("temperature", 0.7)
    num_examples = cfg.get("eval", {}).get("num_examples", 8)
    nfe = cfg.get("eval", {}).get("nfe", 20)  # ODE integration steps
    ode_method = cfg.get("eval", {}).get("ode_method", "heun")  # ODE solver method
    
    # 确保 atol 和 rtol 是浮点数（YAML 可能读取为字符串）
    atol_val = cfg.get("eval", {}).get("atol", 1e-4)
    atol = float(atol_val) if not isinstance(atol_val, (int, float)) else atol_val
    
    rtol_val = cfg.get("eval", {}).get("rtol", 1e-4)
    rtol = float(rtol_val) if not isinstance(rtol_val, (int, float)) else rtol_val
    
    ode_options = cfg.get("eval", {}).get("ode_options", None)  # Additional options for torchdiffeq
    
    if is_main_process():
        logging.info(f"[Eval] Eval modes: {eval_modes}")
        logging.info(f"[Eval] Max batches: {max_eval_batches}")
        logging.info(f"[Eval] Temperature: {temperature}")
        logging.info(f"[Eval] NFE (ODE n_steps): {nfe}")
        logging.info(f"[Eval] ODE method: {ode_method}")
        logging.info(f"[Eval] ODE atol: {atol}, rtol: {rtol}")
    
    # 执行评估（仅在主进程，或所有进程并行评估）
    if ddp_on:
        # 多GPU评估：每个进程评估一部分数据
        if isinstance(eval_loader.sampler, DistributedSampler):
            eval_loader.sampler.set_epoch(0)
    
    # 执行评估（所有进程都参与）
    # 只在主进程显示进度条，避免多GPU时输出混乱
    disable_tqdm = not is_main_process()
    eval_results = evaluate_smiles(
        model=model,
        dataloader=eval_loader,
        device=device,
        eval_modes=eval_modes,
        temperature=temperature,
        max_batches=max_eval_batches,
        log_prefix="eval",
        num_examples=num_examples,
        epoch=0,  # 评估时epoch设为0
        tag="eval",
        disable_tqdm=disable_tqdm,
        nfe=nfe,  # 传递nfe参数
        ode_method=ode_method,  # 传递ODE方法参数
        atol=atol,  # 传递绝对容差
        rtol=rtol,  # 传递相对容差
        options=ode_options,  # 传递其他选项
    )
    
    # 在多GPU情况下，汇总所有进程的结果
    if ddp_on:
        # 汇总每个模式的准确率
        for mode_name, results in eval_results.items():
            if isinstance(results, dict) and "acc" in results:
                # 获取当前进程的准确率和样本数
                acc_tensor = torch.tensor(results["acc"], device=device)
                # 使用all_reduce汇总（这里假设每个进程评估的样本数相同）
                # 如果需要更精确的汇总，需要同时汇总correct数和总数
                dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
                acc_tensor = acc_tensor / world_size
                eval_results[mode_name]["acc"] = acc_tensor.item()
        
        dist.barrier()
    
    # 汇总结果（仅在主进程）
    if is_main_process():
        logging.info("[Eval] Evaluation completed!")
        logging.info(f"[Eval] Results: {eval_results}")
        
        # 记录到wandb
        for mode_name, results in eval_results.items():
            if isinstance(results, dict) and "acc" in results:
                wandb.log({f"eval/acc_{mode_name}": results["acc"]})
        
        # 保存结果到文件
        if exp_dir is not None:
            import json
            results_file = exp_dir / "eval_results.json"
            with results_file.open("w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            logging.info(f"[Eval] Results saved to {results_file}")
    
    # 所有进程都需要finish wandb
    wandb.finish()
    
    cleanup_distributed()

if __name__ == "__main__":
    main()

