# flow_nerf_mvp/utils/encoder_utils.py

from __future__ import annotations
import logging
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import random


def load_pretrained_encoder(
    encoder: nn.Module,
    cfg: Dict[str, Any],
) -> None:
    """
    根据配置加载 encoder 预训练权重：
    model:
      use_pretrained_encoder: true
      encoder_ckpt: "checkpoints/M_encoder_last.pt"
      freeze_encoder: true
    """
    
    use_pretrained = cfg.get("use_pretrained_encoder", False)
    if not use_pretrained:
        logging.info("[Encoder] use_pretrained_encoder = False, skip loading.")
        return

    ckpt_path = Path(cfg.get("encoder_ckpt", ""))
    if not ckpt_path.is_file():
        logging.warning(f"[Encoder] encoder_ckpt not found at {ckpt_path}, skip loading.")
        return

    logging.info(f"[Encoder] Loading pretrained encoder from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]

        # remove backbone prefix from state keys
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        state = {k.replace("M_encoder.", ""): v for k, v in state.items()}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"[Encoder] Missing keys when loading: {missing}")
    if unexpected:
        logging.warning(f"[Encoder] Unexpected keys when loading: {unexpected}")

    if cfg.get("freeze_encoder", True):
        logging.info("[Encoder] Freezing encoder parameters")
        for p in encoder.parameters():
            p.requires_grad = False

    logging.info("[Encoder] done.")
    
def load_checkpoint(model, cfg):
    ckpt_path = Path(cfg.get("checkpoint_path", ""))
    if not ckpt_path.is_file():
        logging.warning(f"[Checkpoint] checkpoint_path not found at {ckpt_path}, skip loading.")
        return

    logging.info(f"[Checkpoint] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    logging.info("[Checkpoint] loaded.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"[Checkpoint] Missing keys when loading: {missing}")
    if unexpected:
        logging.warning(f"[Checkpoint] Unexpected keys when loading: {unexpected}")
    logging.info("[Checkpoint] Load + Check done.")


def load_backbone_only(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict_backbone: bool = False,
) -> None:
    """
    只加载 backbone (encoder+decoder) 权重，忽略 flow head / condition 模块。
    
    支持两种 checkpoint 格式：
    1) 旧版 NERF(w/o vae) RR>>PP（只包含 backbone 或者 key 前缀不同）
    2) 新版 FlowNERF（包含 backbone + flow/cond）
    
    Args:
        model: FlowNERFModel 实例
        ckpt_path: checkpoint 文件路径
        map_location: 加载到哪个设备（默认 "cpu"）
        strict_backbone: 如果 True，遇到 backbone shape mismatch 直接 raise；否则跳过并记录
    
    Returns:
        None（直接修改 model 的权重）
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    logging.info(f"[Backbone Loader] Loading backbone-only weights from {ckpt_path}")
    
    # 1. 加载 checkpoint 并探测真实 state_dict
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    
    # 尝试多种可能的字段名
    state_dict = None
    possible_keys = ["state_dict", "model", "model_state", "net", "weights"]
    
    if isinstance(checkpoint, dict):
        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                logging.info(f"[Backbone Loader] Found state_dict under key: '{key}'")
                break
        
        # 如果都没找到，检查是否整个 checkpoint 就是 state_dict
        if state_dict is None:
            # 检查是否所有值都是 torch.Tensor
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
                logging.info("[Backbone Loader] Checkpoint appears to be a direct state_dict")
    else:
        # checkpoint 本身就是 state_dict
        state_dict = checkpoint
        logging.info("[Backbone Loader] Checkpoint is a direct state_dict")
    
    if state_dict is None:
        raise ValueError(
            f"Could not find state_dict in checkpoint. "
            f"Tried keys: {possible_keys}. "
            f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
        )
    
    # 2. 移除 DDP 前缀（module.）
    state_dict_clean = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            state_dict_clean[k[7:]] = v  # 移除 "module." 前缀
        else:
            state_dict_clean[k] = v
    
    # 3. 获取模型的完整 state_dict（用于匹配）
    model_full_state = dict(model.named_parameters())
    model_full_state.update(dict(model.named_buffers()))
    
    # 获取 backbone 的参数名（用于过滤）
    backbone_param_names = set()
    for name, _ in model.backbone.named_parameters():
        backbone_param_names.add(name)
    for name, _ in model.backbone.named_buffers():
        backbone_param_names.add(name)
    
    # 4. 过滤和映射：只处理 backbone 相关的 key
    backbone_keys_to_load = {}  # {model_key: ckpt_tensor}
    ignored_keys = []  # 非 backbone 的 key
    missing_in_ckpt = []  # 模型需要但 checkpoint 没有的
    shape_mismatch = []  # shape 不匹配的
    unexpected_in_ckpt = []  # checkpoint 有但模型不需要的（非 backbone）
    
    # 需要跳过的关键词（非 backbone 模块）
    skip_keywords = [
        "flow_head", "flow_scale", "time_embed", "delta_fuser",
        "cond_fp_mlp", "cond_gate_mlp", "cond_project",
        "film", "control", "drift"
    ]
    
    # 收集所有可能的 backbone key（带和不带 backbone. 前缀）
    model_backbone_keys = set()
    for name in backbone_param_names:
        model_backbone_keys.add(f"backbone.{name}")  # 带前缀
        model_backbone_keys.add(name)  # 不带前缀
    
    # 遍历 checkpoint 的 key
    for ckpt_key, ckpt_tensor in state_dict_clean.items():
        # 检查是否应该跳过（非 backbone 模块）
        should_skip = False
        for keyword in skip_keywords:
            if keyword in ckpt_key.lower():
                should_skip = True
                break
        
        if should_skip:
            ignored_keys.append(ckpt_key)
            continue
        
        # 尝试匹配到模型的 backbone key
        matched_model_key = None
        
        # 情况 1: checkpoint key 直接是 "backbone.xxx"
        if ckpt_key.startswith("backbone."):
            model_key = ckpt_key  # 保持原样
            if model_key in model_backbone_keys:
                matched_model_key = model_key
            else:
                # 尝试去掉 backbone. 前缀
                model_key_no_prefix = ckpt_key[9:]  # 移除 "backbone."
                if f"backbone.{model_key_no_prefix}" in model_backbone_keys:
                    matched_model_key = f"backbone.{model_key_no_prefix}"
        
        # 情况 2: checkpoint key 没有 backbone. 前缀，尝试添加
        elif ckpt_key in model_backbone_keys:
            matched_model_key = ckpt_key
        elif f"backbone.{ckpt_key}" in model_backbone_keys:
            matched_model_key = f"backbone.{ckpt_key}"
        
        # 如果匹配成功，检查 shape
        if matched_model_key:
            # 获取模型中的对应参数（从完整 state_dict 中）
            if matched_model_key in model_full_state:
                model_tensor = model_full_state[matched_model_key]
            else:
                # 尝试去掉 backbone. 前缀
                model_key_no_prefix = matched_model_key.replace("backbone.", "")
                if f"backbone.{model_key_no_prefix}" in model_full_state:
                    model_tensor = model_full_state[f"backbone.{model_key_no_prefix}"]
                    matched_model_key = f"backbone.{model_key_no_prefix}"
                else:
                    unexpected_in_ckpt.append((ckpt_key, "model key not found"))
                    continue
            
            # 检查 shape
            if model_tensor.shape == ckpt_tensor.shape:
                backbone_keys_to_load[matched_model_key] = ckpt_tensor
            else:
                shape_mismatch.append((ckpt_key, model_tensor.shape, ckpt_tensor.shape))
                if strict_backbone:
                    raise RuntimeError(
                        f"Shape mismatch for backbone key '{ckpt_key}': "
                        f"model expects {model_tensor.shape}, checkpoint has {ckpt_tensor.shape}"
                    )
        else:
            # checkpoint 有这个 key，但模型不需要（非 backbone）
            unexpected_in_ckpt.append((ckpt_key, "not a backbone key"))
    
    # 检查模型需要但 checkpoint 没有的 key（只检查 backbone）
    for backbone_name in backbone_param_names:
        # 尝试多种可能的 checkpoint key 格式
        possible_ckpt_keys = [
            f"backbone.{backbone_name}",  # 带前缀
            backbone_name,  # 不带前缀
        ]
        
        found = False
        for ckpt_key in possible_ckpt_keys:
            if ckpt_key in state_dict_clean:
                # 检查是否已经在 backbone_keys_to_load 中（可能因为 shape mismatch 被跳过）
                model_key = f"backbone.{backbone_name}"
                if model_key in backbone_keys_to_load or backbone_name in backbone_keys_to_load:
                    found = True
                    break
                # 或者检查是否因为 shape mismatch 被跳过
                for mismatch_key, _, _ in shape_mismatch:
                    if mismatch_key == ckpt_key:
                        found = True  # 找到了，只是 shape 不匹配
                        break
                if found:
                    break
        
        if not found:
            missing_in_ckpt.append(f"backbone.{backbone_name}")
    
    # 5. 实际加载
    loaded_count = 0
    for model_key, ckpt_tensor in backbone_keys_to_load.items():
        # 从完整 state_dict 中获取参数对象
        if model_key in model_full_state:
            model_param = model_full_state[model_key]
        else:
            logging.warning(f"[Backbone Loader] Could not find parameter for key: {model_key}")
            continue
        
        # 复制权重
        with torch.no_grad():
            model_param.copy_(ckpt_tensor)
        loaded_count += 1
    
    # 6. 打印统计信息
    logging.info("=" * 70)
    logging.info("[Backbone Loader] Loading Statistics:")
    logging.info(f"  Backbone:")
    logging.info(f"    ✅ Loaded: {loaded_count}")
    logging.info(f"    ⚠️  Skipped (shape mismatch): {len(shape_mismatch)}")
    logging.info(f"    ❌ Missing in checkpoint: {len(missing_in_ckpt)}")
    logging.info(f"    ⚠️  Unexpected in checkpoint: {len(unexpected_in_ckpt)}")
    logging.info(f"  Non-backbone:")
    logging.info(f"    🚫 Ignored (filtered out): {len(ignored_keys)}")
    logging.info("=" * 70)
    
    # 7. 随机抽样打印未加载原因（最多 5 个）
    all_issues = []
    all_issues.extend([(k, "shape mismatch", f"{ms} vs {cs}") for k, ms, cs in shape_mismatch])
    all_issues.extend([(k, "missing in checkpoint", "") for k in missing_in_ckpt[:10]])  # 最多显示 10 个
    all_issues.extend([(k, reason, "") for k, reason in unexpected_in_ckpt[:10]])
    
    if all_issues:
        sample_size = min(5, len(all_issues))
        sampled = random.sample(all_issues, sample_size) if len(all_issues) > sample_size else all_issues
        logging.info(f"[Backbone Loader] Sample of unloaded keys (showing {len(sampled)}/{len(all_issues)}):")
        for key, reason, detail in sampled:
            if detail:
                logging.info(f"    - {key}: {reason} ({detail})")
            else:
                logging.info(f"    - {key}: {reason}")
    
    logging.info(f"[Backbone Loader] Done. Loaded {loaded_count} backbone parameters.")
    logging.info("=" * 70)