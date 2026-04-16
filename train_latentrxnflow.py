from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset
import pickle
from utils.save_checkpoint import save_checkpoint
from utils.experiment import (
    setup_experiment,
    setup_logging,
    save_model_summary,
    save_wandb_info,
    snapshot_code,
)
from utils.encoder_utils import load_pretrained_encoder, load_backbone_only
import logging
import wandb
from utils.data_utils import build_dataloaders, evaluate_smiles
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
from utils.experiment import load_config


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
@contextmanager
def override_decoder_cfg(model, delta_source=None, input_mode=None):
    old = deepcopy(getattr(model, "decoder_cfg", None))
    try:
        if not hasattr(model, "decoder_cfg") or model.decoder_cfg is None:
            yield
            return
        if delta_source is not None:
            model.decoder_cfg.delta_source = delta_source
        if input_mode is not None:
            model.decoder_cfg.input_mode = input_mode
        yield
    finally:
        if old is not None:
            model.decoder_cfg = old


def _parse_mode(mode):
    """
    mode 支持:
      - "tf" / "flow" / "ode"
      - {"name": "...", "delta_source": "...", "input_mode": "..."}
    返回: (mode_name, delta_source, input_mode)
    """
    if isinstance(mode, dict):
        name = mode.get("name", "custom")
        return str(name), mode.get("delta_source", None), mode.get("input_mode", None)
    s = str(mode).lower()
    return s, s, None


@torch.no_grad()
def evaluate_loss_modes(
    model,
    dataloader: DataLoader,
    device: torch.device,
    eval_modes=None,
    max_batches: int | None = None,
):
    """
    多模式 eval（loss 版），返回：
      {
        "tf":   {"loss/total":..., "loss/pred":..., ...},
        "flow": {...},
        "ode":  {...},
      }

    使用 model.decoder_cfg (tf/flow/ode + input_mode)
    """
    base = unwrap_model(model)   # ✅ 关键：拿到真实 FlowNERFModel

    model.eval()

    if eval_modes is None:
        eval_modes = ["tf", "flow"]
    # 强制包含 TF
    has_tf = any((m == "tf") or (isinstance(m, dict) and "tf" in m.get("name") ) for m in eval_modes)
    if not has_tf:
        eval_modes = ["tf"] + list(eval_modes)

    all_metrics = {}

    for mode in eval_modes:
        mode_name, delta_source, input_mode = _parse_mode(mode)

        n_batches = 0
        sum_total = 0.0
        sum_pred = 0.0
        sum_flow = 0.0
        sum_bond = 0.0
        sum_aroma = 0.0
        sum_charge = 0.0

        # --- 设置 mode（优先新 decoder_cfg，否则用旧 flag）---
        if getattr(base, "decoder_cfg", None) is not None:
            ctx = override_decoder_cfg(base, delta_source=delta_source, input_mode=input_mode)
        else:
            ctx = contextmanager(lambda: (yield))()
        with ctx:
            for step, batch in enumerate(dataloader):
                if max_batches is not None and step >= max_batches:
                    break

                reactant = batch.reactant.to(device)
                condition = batch.condition.to(device)
                product_latent = batch.product_latent.to(device)
                product_token = batch.product_token.to(device)

                outputs = model(
                    reactant=reactant,
                    condition=condition,
                    product_latent=product_latent,
                    product_token=product_token,
                )

                sum_total += outputs["loss"].item()
                sum_pred  += outputs["pred_loss"].item()
                sum_flow  += outputs["flow_loss"].item()
                sum_bond  += outputs["bond_loss"].item()
                sum_aroma += outputs["aroma_loss"].item()
                sum_charge+= outputs["charge_loss"].item()
                n_batches += 1

        if n_batches == 0:
            all_metrics[mode_name] = {}
        else:
            all_metrics[mode_name] = {
                "loss/total":  sum_total / n_batches,
                "loss/pred":   sum_pred  / n_batches,
                "loss/flow":   sum_flow  / n_batches,
                "loss/bond":   sum_bond  / n_batches,
                "loss/aroma":  sum_aroma / n_batches,
                "loss/charge": sum_charge/ n_batches,
            }

    return all_metrics

def train_one_epoch(
    model: FlowNERFModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> None:
    model.train()
    main = is_main_process()
    if main:
        logging.info(f"[Train][Epoch {epoch}]")
    for step, batch in enumerate(dataloader):
        reactant = batch.reactant.to(device)
        condition = batch.condition.to(device)
        # print(condition)
        # exit()
        product_latent = batch.product_latent.to(device)
        product_token = batch.product_token.to(device)

        outputs = model(
            reactant=reactant,
            condition=condition,
            product_latent=product_latent,
            product_token=product_token,
        )
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Extract metrics for minimal logger
        flow_loss_val = outputs["flow_loss"].item()
        v_cos_val = outputs.get("v_cos", 0.0)
        if isinstance(v_cos_val, torch.Tensor):
            v_cos_val = v_cos_val.item()
        v_pred_norm_mean = outputs.get("v_pred_norm_mean", 0.0)
        if isinstance(v_pred_norm_mean, torch.Tensor):
            v_pred_norm_mean = v_pred_norm_mean.item()
        v_tgt_norm_mean = outputs.get("v_tgt_norm_mean", 0.0)
        if isinstance(v_tgt_norm_mean, torch.Tensor):
            v_tgt_norm_mean = v_tgt_norm_mean.item()
        norm_ratio = v_pred_norm_mean / (v_tgt_norm_mean + 1e-8) if v_tgt_norm_mean > 0 else 0.0
        cond_drop_rate = outputs.get("cond_drop_rate", 0.0)
        if isinstance(cond_drop_rate, torch.Tensor):
            cond_drop_rate = cond_drop_rate.item()
        
        # Minimal logger: every 200 steps
        if main and step % 200 == 0:
            logging.info(
                f"[{epoch:3d}|{step:5d}] "
                f"flow={flow_loss_val:.4f} "
                f"v_cos={v_cos_val:.3f} "
                f"norm_ratio={norm_ratio:.3f} "
                f"cond_drop={cond_drop_rate:.3f}"
            )
        
        bond_loss = outputs.get("bond_loss", torch.tensor(0.0)).item() if "bond_loss" in outputs else 0.0
        aroma_loss = outputs.get("aroma_loss", torch.tensor(0.0)).item() if "aroma_loss" in outputs else 0.0
        charge_loss = outputs.get("charge_loss", torch.tensor(0.0)).item() if "charge_loss" in outputs else 0.0
        pred_loss = outputs.get("pred_loss", torch.tensor(0.0)).item() if "pred_loss" in outputs else 0.0

        if main and step % 10 == 0:
            logging.info(                f"[Epoch {epoch} Step {step}] "
                f"total={loss.item():.4f} "
                f"bond={bond_loss:.4f} aroma={aroma_loss:.4f} charge={charge_loss:.4f} flow={flow_loss_val:.4f}"
            )
        if main:    
            # Collect all metrics for wandb
            wandb_dict = {
                "loss/total": loss.item(),
                "loss/pred": pred_loss,
                "loss/bond": bond_loss,
                "loss/aroma": aroma_loss,
                "loss/charge": charge_loss,
                "loss/flow": flow_loss_val,  # flow_loss_total (包含 norm 约束)
                "lr": optimizer.param_groups[0]["lr"],
                # Minimal logger metrics
                "flow/v_cos": v_cos_val,
                "flow/norm_ratio": norm_ratio,
                "flow/cond_drop_rate": cond_drop_rate,
            }
            
            # Extract flow_loss_raw and loss_norm separately for detailed logging
            flow_loss_raw_val = outputs.get("flow_loss_raw", None)
            if flow_loss_raw_val is None:
                # Try with "flow/" prefix
                flow_loss_raw_val = outputs.get("flow/flow_loss_raw", None)
            if flow_loss_raw_val is not None:
                if isinstance(flow_loss_raw_val, torch.Tensor):
                    wandb_dict["flow/flow_loss_raw"] = flow_loss_raw_val.item()
                else:
                    wandb_dict["flow/flow_loss_raw"] = flow_loss_raw_val
            
            loss_norm_val = outputs.get("loss_norm", None)
            if loss_norm_val is None:
                # Try with "flow/" prefix
                loss_norm_val = outputs.get("flow/loss_norm", None)
            if loss_norm_val is not None:
                if isinstance(loss_norm_val, torch.Tensor):
                    wandb_dict["flow/loss_norm"] = loss_norm_val.item()
                else:
                    wandb_dict["flow/loss_norm"] = loss_norm_val
            
            # Add flow sanity metrics (if available)
            # Check both with and without "flow/" prefix (for layer1_flow_only mode)
            sanity_metric_keys = [
                "v_cos",  # node-level cosine similarity
                "v_cos_graph",  # graph-level cosine similarity
                "dir_loss_node",  # node-level 方向监督损失
                "dir_loss_graph",  # graph-level 方向监督损失
                "dir_loss",  # 混合后的方向监督损失
                "mag_loss",  # 幅度对齐损失（graph-level log norm loss）
                "graph_norm_ratio",  # graph-level 幅度比例
                "vt_norm2",
                "flow_loss_per_dim",
                "cos_mean_graph",
                "norm_ratio_graph_mean",
                "norm_ratio_graph_std",
                "loss_norm",  # 幅度约束项（保留用于兼容性）
                "flow_loss_raw",  # 原始 flow_loss（使用 dir_loss）
                # Decomposition metrics
                "cos_node_total",  # total node-level cosine
                "cos_graph",  # graph vector-level cosine (new)
                "cos_res",  # residual cosine (new)
                "cos_res_node",  # residual node-level cosine
                "cos_graph_node",  # graph broadcast to node cosine
                "res_norm_mean",  # mean residual norm
                "graph_norm_mean",  # mean graph norm
                "flow_loss_total",  # total flow loss (used for training)
                "dir_loss_graph_decomp",  # graph direction loss (decomp mode)
                "dir_loss_res",  # residual direction loss (decomp mode)
                # New graph + residual decomposition metrics
                "mag_loss_graph",  # graph magnitude loss
                "mag_loss_res",  # residual magnitude loss
                "var_loss_res",  # residual variance loss
                # Endpoint consistency loss
                "end_loss",  # endpoint consistency loss
            ]
            for key in sanity_metric_keys:
                # Try with "flow/" prefix first (normal mode)
                flow_key = f"flow/{key}"
                if flow_key in outputs:
                    value = outputs[flow_key]
                    if isinstance(value, torch.Tensor):
                        wandb_dict[flow_key] = value.item()
                    else:
                        wandb_dict[flow_key] = value
                # Try without prefix (layer1_flow_only mode)
                elif key in outputs:
                    value = outputs[key]
                    if isinstance(value, torch.Tensor):
                        wandb_dict[flow_key] = value.item()  # Still use "flow/" prefix in wandb
                    else:
                        wandb_dict[flow_key] = value
            
            # Add other diagnostic metrics (if available)
            diagnostic_keys = [
                "cond_norm",
                "gate_pool_mean",
                "film_gamma_abs",
                "film_beta_abs",
                "control_alpha",
            ]
            for key in diagnostic_keys:
                if key in outputs:
                    value = outputs[key]
                    if isinstance(value, torch.Tensor):
                        wandb_dict[f"diagnostics/{key}"] = value.item()
                    else:
                        wandb_dict[f"diagnostics/{key}"] = value
            
            # Add debug attention and gradient metrics (if available)
            # These are already prefixed with "flow/" in the model outputs
            debug_metric_keys = [
                "flow/attn_mask_true_ratio",
                "flow/attn_max_mean",
                "flow/attn_entropy_mean",
                "flow/attn_eff_tokens_mean",
                "flow/ctx_norm_mean",
                "flow/ctx_ratio",
                "flow/Nc_max",
                "flow/cond_num_mean",
                "flow/cond_num_max",
                "flow/dbg_grad_attn",
                "flow/dbg_grad_fp",
            ]
            for key in debug_metric_keys:
                if key in outputs:
                    value = outputs[key]
                    if isinstance(value, torch.Tensor):
                        wandb_dict[key] = value.item()
                    else:
                        wandb_dict[key] = value
            
            wandb.log(wandb_dict)
            
def load_checkpoint_for_resume(checkpoint_path: Path, model, optimizer, device):
    """
    加载checkpoint用于恢复训练
    返回: (start_epoch, checkpoint_cfg)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if is_main_process():
        logging.info(f"[Resume] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 加载模型权重
    if "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        base_model = unwrap_model(model)
        
        # 处理state_dict的key：总是加载到base_model（unwrap后的模型，没有module.前缀）
        # 所以需要移除checkpoint中可能存在的module.前缀
        model_state_clean = {}
        for k, v in model_state.items():
            if k.startswith("module."):
                # checkpoint保存的是DDP模型（带module.前缀），移除前缀
                model_state_clean[k[7:]] = v
            else:
                # checkpoint保存的是普通模型，直接使用
                model_state_clean[k] = v
        
        # 加载到base_model（unwrap后的模型，没有module.前缀）
        missing_keys, unexpected_keys = base_model.load_state_dict(model_state_clean, strict=False)
        if is_main_process():
            if missing_keys:
                logging.warning(f"[Resume] Missing keys when loading model: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"[Resume] Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"[Resume] Unexpected keys when loading model: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"[Resume] Unexpected keys: {unexpected_keys}")
            logging.info("[Resume] Model weights loaded")
    else:
        if is_main_process():
            logging.warning("[Resume] No model_state found in checkpoint")
    
    # 加载优化器状态（必须恢复，用于继续训练）
    if "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if is_main_process():
                logging.info("[Resume] Optimizer state loaded successfully")
        except Exception as e:
            if is_main_process():
                logging.error(f"[Resume] Failed to load optimizer state: {e}")
                logging.error("[Resume] This may happen if model structure changed. Training will continue with fresh optimizer state.")
                logging.error("[Resume] If you need to resume optimizer state, please ensure model structure matches the checkpoint.")
            # 不抛出异常，允许训练继续（使用新的optimizer状态）
            # 如果用户确实需要optimizer_state，可以检查错误信息并修复
    else:
        if is_main_process():
            logging.warning("[Resume] No optimizer_state found in checkpoint - optimizer will start fresh")
    
    # 获取起始epoch
    start_epoch = checkpoint.get("epoch", 0) + 1  # 从下一个epoch开始
    if is_main_process():
        logging.info(f"[Resume] Resuming from epoch {start_epoch}")
    
    # 返回checkpoint中的config（如果有）
    checkpoint_cfg = checkpoint.get("config", None)
    
    return start_epoch, checkpoint_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-NERF MVP 训练脚本")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("flow_nerf_mvp/configs/base.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--ddp", action="store_true", help="use DDP (torchrun)")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint路径，用于恢复训练",
    )
    args = parser.parse_args()
    ddp_on, rank, world_size, local_rank = init_distributed()

    cfg = load_config(args.config)
    if is_main_process():
        set_seed(cfg["train"]["seed"])
        # ===== 1) 建 experiment 目录，保存 config / cmd =====
        exp_dir = setup_experiment(cfg)

        # ===== 2) 设置 logging（控制台 + train.log） =====
        setup_logging(exp_dir)
        logging.info(f"Experiment dir: {exp_dir}")

        # ===== 3) 代码快照：train_flow_nerf.py + models/* =====
        snapshot_code(exp_dir, FLOW_ROOT)
        logging.info("Code snapshot saved.")
        
        # 保存exp_dir到cfg中，方便后续使用
        cfg["_exp_dir"] = str(exp_dir)
    else:
        exp_dir = None

    if ddp_on:
        dist.barrier()
    # ===== 4) 初始化 wandb（所有进程都需要初始化，因为evaluate_smiles中的可视化函数会调用wandb.log）
    # 主进程使用正常模式，非主进程使用disabled模式
    if is_main_process():
        wandb_mode = "online"  # 可以根据配置调整
    else:
        wandb_mode = "disabled"  # 非主进程禁用wandb，避免重复记录
    
    wandb.init(
        project=cfg.get("experiment", {}).get("project", "flow-nerf"),
        name=cfg.get("experiment", {}).get("name", "flow-nerf-mvp"),
        config=cfg,
        mode=wandb_mode,
    )
    # device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # 这里按你原来的方式加载数据
    train_loader, eval_loader, overfit_loader = build_dataloaders(cfg, args.ddp)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    args_nn = SimpleArgs()
    
    # Set model config attributes for flow head selection
    class ModelConfig:
        pass
    args_nn.model = ModelConfig()
    model_cfg = cfg.get("model", {})
    args_nn.model.flow_cond_head = model_cfg.get("flow_cond_head", "controlnet")
    args_nn.model.film_hidden_dim = model_cfg.get("film_hidden_dim", model_cfg.get("latent_dim", 256) * 2)
    args_nn.model.film_init_zero = model_cfg.get("film_init_zero", True)
    args_nn.model.film_s_gamma = model_cfg.get("film_s_gamma", 1.0)
    args_nn.model.film_s_beta = model_cfg.get("film_s_beta", 0.2)
    args_nn.model.cond_pool = model_cfg.get("cond_pool", "gated")
    args_nn.model.cond_drop_prob = model_cfg.get("cond_drop_prob", 0.2)
    args_nn.model.force_zero_cond = model_cfg.get("force_zero_cond", False)
    # CondAttn head specific configs
    args_nn.model.cond_attn_n_heads = model_cfg.get("cond_attn_n_heads", 4)
    args_nn.model.cond_mol_emb_dim = model_cfg.get("cond_mol_emb_dim", 256)
    # Debug configuration
    args_nn.model.debug_attn = model_cfg.get("debug_attn", False)
    args_nn.model.debug_attn_every = model_cfg.get("debug_attn_every", 200)
    args_nn.model.debug_grad = model_cfg.get("debug_grad", False)
    args_nn.model.debug_grad_every = model_cfg.get("debug_grad_every", 200)
    # Layer-1 flow-only mode configuration
    args_nn.model.layer1_flow_only = model_cfg.get("layer1_flow_only", False)
    args_nn.model.layer1_freeze_backbone = model_cfg.get("layer1_freeze_backbone", False)
    args_nn.model.layer1_debug_shape_asserts = model_cfg.get("layer1_debug_shape_asserts", False)
    # Flow objective and sanity check configuration
    args_nn.model.flow_objective = model_cfg.get("flow_objective", "fm")
    args_nn.model.log_flow_sanity = model_cfg.get("log_flow_sanity", True)
    args_nn.model.flow_loss_reduce = model_cfg.get("flow_loss_reduce", "sum")
    args_nn.model.lambda_norm = model_cfg.get("lambda_norm", 0.02)
    args_nn.model.lambda_end = model_cfg.get("lambda_end", 0.0)
    args_nn.model.w_mag_graph = model_cfg.get("w_mag_graph", 0.1)
    args_nn.model.w_mag_res = model_cfg.get("w_mag_res", 0.1)
    args_nn.model.w_var_res = model_cfg.get("w_var_res", 0.05)
    
    # Set trainable modules configuration
    args_nn.model.freeze_encoder = model_cfg.get("freeze_encoder", True)
    args_nn.model.freeze_decoder = model_cfg.get("freeze_decoder", True)
    args_nn.model.unfreeze_enc_last_n_layers = model_cfg.get("unfreeze_enc_last_n_layers", 0)
    args_nn.model.unfreeze_dec_first_n_layers = model_cfg.get("unfreeze_dec_first_n_layers", 0)
    args_nn.model.unfreeze_layernorm = model_cfg.get("unfreeze_layernorm", True)
    args_nn.model.unfreeze_decoder_head = model_cfg.get("unfreeze_decoder_head", True)
    
    dec = cfg.get("model", {}).get("decoder", {}) or {}

    decoder_cfg = DecoderConfig(
        delta_source=dec.get("delta_source", "tf"),
        input_mode=dec.get("input_mode", "fuse"),
        ode_method=dec.get("ode_method", "heun"),
    )

    model = FlowNERFModel(
        latent_dim=cfg["model"]["latent_dim"],
        # vocab_size=cfg["model"]["vocab_size"],   # 你项目里实际怎么取就怎么填
        cond_dim=cfg["model"]["cond_dim"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
        ntoken=cfg["model"].get("ntoken", 128),
        args=args_nn,
        flow_weight=cfg["model"].get("flow_weight", 1e-2),
        detach_encoder_for_flow=cfg["model"].get("detach_encoder_for_flow", True),
        flow_sampling_cfg=cfg["model"].get("flow_sampling_cfg", None),
        fm_sigma=cfg["model"].get("fm_sigma", 0.0),
        # ✅ 关键：把 decoder_cfg 传进去
        decoder_cfg=decoder_cfg,
        use_conditional_flow=cfg["model"].get("use_conditional_flow", False),
        # ✅ 从 config 读取 nfe 参数
        nfe=dec.get("nfe", 20),
        # cond_dim=cfg["model"].get("cond_dim", 640),
    ).to(device)
    if ddp_on:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = model.to(device)
    if is_main_process():
        base = unwrap_model(model)
        print(f"[Init] decoder_cfg = delta_source={base.decoder_cfg.delta_source}, input_mode={base.decoder_cfg.input_mode}")
        print(f"[Init] flow_sampling_cfg = {base.flow_sampling_cfg}")
        
        # Layer-1 sanity check: print trainable params summary
        if base.layer1_flow_only or base.layer1_freeze_backbone:
            base.print_trainable_params_summary()
        
        save_model_summary(exp_dir, base)
    # else:
        # base = None
    # if is_main_process():
    # print(f"[Init] decoder_cfg = delta_source={model.decoder_cfg.delta_source}, input_mode={model.decoder_cfg.input_mode}")
    # print(f"[Init] flow_sampling_cfg = {model.flow_sampling_cfg}")
    # save_model_summary(exp_dir, model)

    # ===== 加载 backbone/encoder 预训练权重 =====
    # 优先级：backbone_ckpt > encoder_ckpt
    # - backbone_ckpt: 加载整个 backbone (encoder + decoder)，忽略 flow/cond 模块
    # - encoder_ckpt: 只加载 encoder 部分
    backbone_ckpt = cfg.get("model", {}).get("backbone_ckpt")
    use_encoder_ckpt = cfg.get("train", {}).get("use_pretrained_encoder", False)
    
    if is_main_process():
        if backbone_ckpt:
            # 方式1: 使用 load_backbone_only（加载整个 backbone）
            if use_encoder_ckpt:
                logging.warning(
                    "[Checkpoint] Both backbone_ckpt and encoder_ckpt are specified. "
                    "Using backbone_ckpt (ignoring encoder_ckpt)."
                )
            strict_backbone = cfg.get("model", {}).get("strict_backbone", False)
            load_backbone_only(
                model=unwrap_model(model),
                ckpt_path=backbone_ckpt,
                map_location="cpu",
                strict_backbone=strict_backbone,
            )
        elif use_encoder_ckpt:
            # 方式2: 使用 load_pretrained_encoder（只加载 encoder）
            load_pretrained_encoder(
                encoder=unwrap_model(model).backbone.M_encoder,
                cfg=cfg["train"],
            )
        else:
            logging.info("[Checkpoint] No backbone_ckpt or encoder_ckpt specified. Using random initialization.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])

    num_epochs = cfg["train"]["num_epochs"]  # 比如你在 yaml 里写 10
    eval_modes = cfg.get("eval", {}).get("decoder_eval_modes", None) or ["tf", "flow"]
    logging.info(f"Eval modes: {eval_modes}")
    max_eval_batches = cfg.get("eval", {}).get("max_batches", None)
    # logging.info(f"[Debug] eval_modes(raw) = {eval_modes}")
    # exit()
    ddp_on = dist.is_available() and dist.is_initialized()
    
    # ===== 恢复训练：加载checkpoint =====
    start_epoch = 0
    if args.resume is not None:
        # 所有进程都需要加载checkpoint（DDP模式下）
        start_epoch, checkpoint_cfg = load_checkpoint_for_resume(
            args.resume, model, optimizer, device
        )
        
        # 如果checkpoint中有config，可以选择性地更新当前config
        if is_main_process() and checkpoint_cfg is not None:
            logging.info("[Resume] Checkpoint contains config, but using current config file")
        
        # DDP模式下同步所有进程，确保所有进程的start_epoch一致
        if ddp_on:
            dist.barrier()
            # 广播start_epoch到所有进程（确保一致性）
            start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.long, device=device)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
        
        if is_main_process():
            logging.info(f"[Resume] Training will resume from epoch {start_epoch}/{num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        
        if ddp_on and isinstance(train_loader.sampler, DistributedSampler):
            # if is_main_process():
            train_loader.sampler.set_epoch(epoch)
            # train_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, device, epoch)

        if is_main_process() and cfg["train"]["save_last_ckpt"]:
            # 使用exp_dir作为checkpoint保存目录（如果存在），否则使用配置的save_dir
            save_dir = cfg.get("_exp_dir", cfg["train"]["save_dir"])
            # save_checkpoint内部会计算 epoch = num_epochs - 1，所以传入 epoch + 1
            save_checkpoint(unwrap_model(model), optimizer, cfg, epoch + 1, save_dir=save_dir)
        if not cfg["eval"]["enabled"] or epoch % cfg["eval"]["eval_interval"] != 0:
            continue
        
        # ===== 单GPU评估：只在主进程进行 =====
        if not is_main_process():
            continue
        
        # 创建评估子集（如果配置了eval_subset_ratio）
        eval_subset_ratio = cfg.get("eval", {}).get("eval_subset_ratio", None)
        eval_loader_to_use = eval_loader
        if eval_subset_ratio is not None and eval_subset_ratio < 1.0:
            # 创建子集DataLoader
            eval_dataset = eval_loader.dataset
            total_size = len(eval_dataset)
            subset_size = int(total_size * eval_subset_ratio)
            indices = list(range(subset_size))
            subset_dataset = torch.utils.data.Subset(eval_dataset, indices)
            eval_loader_to_use = DataLoader(
                subset_dataset,
                batch_size=cfg["eval"]["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=cfg["data"].get("num_workers", 4),
                pin_memory=True,
            )
            logging.info(f"[Eval] Using {subset_size}/{total_size} samples ({eval_subset_ratio*100:.1f}%) for evaluation")
        
        of_loss_all, of_acc_all = {}, {}
        if overfit_loader is not None:
            # of_loss_all = evaluate_loss_modes(
            #     model=model,
            #     dataloader=overfit_loader,
            #     device=device,
            #     eval_modes=eval_modes,
            #     max_batches=max_eval_batches,
            # )
            of_acc_all = evaluate_smiles(
                model=model,
                dataloader=overfit_loader,
                device=device,
                eval_modes=eval_modes,
                temperature=0.7,
                max_batches=max_eval_batches,
                log_prefix="overfit",
                num_examples=8,
                epoch=epoch,
                tag="overfit",
                disable_tqdm=False,
                ode_method=dec.get("ode_method", "heun"),  # Use config value or default to "heun"
            )

        # ===== val评估 =====
        # val_loss_all = evaluate_loss_modes(
        #     model=model,
        #     dataloader=eval_loader_to_use,
        #     device=device,
        #     eval_modes=eval_modes,
        #     max_batches=max_eval_batches,
        # )
        val_loss_all = {}
        val_acc_all = evaluate_smiles(
            model=model,
            dataloader=eval_loader_to_use,
            device=device,
            eval_modes=eval_modes,
            temperature=0.7,
            max_batches=max_eval_batches,
            log_prefix="val",
            num_examples=8,
            epoch=epoch,
            tag="val",
            disable_tqdm=False,
            ode_method=dec.get("ode_method", "heun"),  # Use config value or default to "heun"
        )

        # ===== logging =====
        logging.info(f"[Eval][Epoch {epoch}]")
        if of_loss_all:
            logging.info(f"  Overfit loss: { {m: of_loss_all[m].get('loss/total', None) for m in of_loss_all} }")
        if of_acc_all:
            logging.info(f"  Overfit acc : { {m: of_acc_all[m].get('acc', None) for m in of_acc_all} }")
        if val_loss_all:
            logging.info(f"  Val loss    : { {m: val_loss_all[m].get('loss/total', None) for m in val_loss_all} }")
        if val_acc_all:
            logging.info(f"  Val acc     : { {m: val_acc_all[m].get('acc', None) for m in val_acc_all} }")

        # ===== wandb log =====
        log_dict = {"epoch": epoch}

        # overfit loss/acc
        for mode_name, md in of_loss_all.items():
            for k, v in md.items():
                log_dict[f"loss/overfit_{mode_name}_{k.split('/')[-1]}"] = v  # e.g. loss/overfit_tf_total
        for mode_name, md in of_acc_all.items():
            if isinstance(md, dict) and "acc" in md:
                log_dict[f"acc/overfit_{mode_name}_acc"] = md["acc"]

        # val loss/acc
        for mode_name, md in val_loss_all.items():
            for k, v in md.items():
                log_dict[f"loss/val_{mode_name}_{k.split('/')[-1]}"] = v
        for mode_name, md in val_acc_all.items():
            if isinstance(md, dict) and "acc" in md:
                log_dict[f"acc/val_{mode_name}_acc"] = md["acc"]

        wandb.log(log_dict)

    # ===== save + finish =====
    if is_main_process():
        if cfg["train"]["save_checkpoint"]:
            # 使用exp_dir作为checkpoint保存目录（如果存在），否则使用配置的save_dir
            save_dir = cfg.get("_exp_dir", cfg["train"]["save_dir"])
            save_checkpoint(unwrap_model(model), optimizer, cfg, num_epochs, save_dir=save_dir)
        save_wandb_info(exp_dir)
    
    # 所有进程都需要finish wandb
    wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
