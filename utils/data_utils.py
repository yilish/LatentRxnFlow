import random
from typing import Dict, Any, Tuple, Optional
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.uspto_main_product import USPTOReact2MainProduct, collate_fn
def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1

def build_dataloaders(
    cfg: Dict[str, Any],
    ddp_on: bool,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    返回:
      - train_loader: full train
      - eval_loader:  eval set
      - overfit_loader: train 的小子集（只用于 eval）
    
    支持两种模式：
    1. 直接指定 train_pickle_path 和 eval_pickle_path（推荐）
    2. 使用 pickle_path + train_ratio/eval_ratio 分割（向后兼容）
    """
    ddp_on, rank, world_size = _dist_info()

    # 优先使用直接指定的训练集和验证集pickle文件
    train_pickle_path = cfg["data"].get("train_pickle_path", None)
    eval_pickle_path = cfg["data"].get("eval_pickle_path", None)
    
    if train_pickle_path is not None and eval_pickle_path is not None:
        # 模式1：直接加载指定的pickle文件
        if rank == 0:
            print(f"[Data] Loading train data from {train_pickle_path}")
        with open(train_pickle_path, "rb") as f:
            train_data = pickle.load(f)
        
        if rank == 0:
            print(f"[Data] Loading eval data from {eval_pickle_path}")
        with open(eval_pickle_path, "rb") as f:
            eval_data = pickle.load(f)
        
        if rank == 0:
            print(f"[Data] Train size = {len(train_data)}, Eval size = {len(eval_data)}")
    else:
        # 模式2：从单个pickle文件分割（向后兼容）
        data_path = cfg["data"].get(
            "pickle_path", "data/all_reactions_data_with_segmentation_mask.pickle"
        )
        with open(data_path, "rb") as f:
            data_list = pickle.load(f)

        if rank == 0:
            print(f"[Data] Loaded {len(data_list)} reactions from {data_path}")

        # ---- 关键：固定 seed，确保每个 rank 的 split 完全一致 ----
        split_seed = int(cfg["data"].get("split_seed", cfg["train"]["seed"]))
        indices = list(range(len(data_list)))
        rng = random.Random(split_seed)
        rng.shuffle(indices)

        train_ratio = cfg["data"].get("train_ratio", 0.9)
        eval_ratio = cfg["data"].get("eval_ratio", 0.1)

        split = int(len(indices) * train_ratio)
        eval_split = split + int(len(indices) * eval_ratio)

        train_idx = indices[:split]
        eval_idx  = indices[split:eval_split]

        train_data = [data_list[i] for i in train_idx]
        eval_data  = [data_list[i] for i in eval_idx]

        if rank == 0:
            print(f"[Data] Train size = {len(train_data)}, Eval size = {len(eval_data)}")

    train_batch_size = cfg["train"]["batch_size"]
    eval_batch_size  = cfg["eval"]["batch_size"]

    num_workers = cfg["data"].get("num_workers", 4)
    pin_memory = True

    # ===== train loader =====
    train_dataset = USPTOReact2MainProduct(
        data_list=train_data,
        if_shuffle=False,  # ✅ DDP 下不要靠 dataset 内部 shuffle（由 sampler 控制）
    )

    train_sampler = None
    if ddp_on:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,   # 常见做法：保持各 rank batch 数一致
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,          # 先关掉验证是否还卡
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    # ===== eval loader =====
    eval_dataset = USPTOReact2MainProduct(
        data_list=eval_data,
        if_shuffle=False,
    )

    # eval 使用 DistributedSampler 支持多卡并行评估
    eval_sampler = None
    if ddp_on:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # ===== overfit loader =====
    overfit_subset_size = cfg["data"].get("overfit_subset_size", 0)
    overfit_loader: Optional[DataLoader] = None

    # ✅ 建议只在 rank0 做 overfit eval（否则重复生成/打印/写 wandb）
    if (not ddp_on or rank == 0) and overfit_subset_size and overfit_subset_size > 0:
        # 从训练集中取前 overfit_subset_size 个样本
        sub_data = train_data[:overfit_subset_size]
        if rank == 0:
            print(f"[Data] Overfit subset size = {len(sub_data)}")

        overfit_dataset = USPTOReact2MainProduct(
            data_list=sub_data,
            if_shuffle=False,
        )
        overfit_loader = DataLoader(
            overfit_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
    if rank == 0:
        try:
            sampler_len = len(train_sampler) if train_sampler is not None else len(train_dataset)
        except Exception:
            sampler_len = None
        print(
            f"[DL DEBUG] ddp_on={ddp_on} world_size={world_size} "
            f"train_len={len(train_dataset)} sampler={type(train_sampler).__name__ if train_sampler else None} "
            f"sampler_len={sampler_len} batch_size(per_gpu)={train_batch_size} len(train_loader)={len(train_loader)}"
        )
    return train_loader, eval_loader, overfit_loader
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import torch
import wandb

# 你自己的工具函数：
# from your_module import result2mol, visualize

from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import os
import json
import pdb

            
def mol2array(mol):
    img = Draw.MolToImage(mol, kekulize=False)
    array = np.array(img)[:, :, 0:3]
    return array

def check(smile):
    smile = smile.split('.')
    smile.sort(key = len)
    try:
        mol = Chem.MolFromSmiles(smile[-1], sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except Exception:
        return False

def mol2file(m, name):
    AllChem.Compute2DCoords(m)
    img = Draw.MolToImage(m)
    Draw.MolToFile(m, os.path.join('./img', name))


def result2mol(args): # for threading
    element, mask, bond, aroma, charge, reactant = args
    # [L], [L], [L, 4], [l], [l]
    mask = mask.ne(1)
    cur_len = sum(mask.long())
    l = element.shape[0]

    mol = Chem.RWMol()
    
    element = element.cpu().numpy().tolist()
    charge = charge.cpu().numpy().tolist()
    bond = bond.cpu().numpy().tolist()    
    
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(l):
        if mask[i] == False:
            continue
        a = Chem.Atom(element[i])
        if not reactant is None and reactant[i]:
            a.SetAtomMapNum(i+1)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for this in range(l):
        if mask[this] == False:
            continue
        lst = bond[this]
        for j in range(len(bond[0])):
            other = bond[this][j]
            # only traverse half the matrix
            if other >= this or other in lst[0:j] or not this in bond[other]:
                continue
            if lst.count(other)==3 or bond[other].count(this) == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type) 
            elif lst.count(other) == 2 or bond[other].count(this) == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)   
            else:
                if aroma[this]==aroma[other] and aroma[this]>0: 
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
                 
    for i, item in enumerate(charge):
        if mask[i] == False:
            continue
        if not item == 0:
            atom = mol.GetAtomWithIdx(node_to_idx[i])
            atom.SetFormalCharge(item)
    # Convert RWMol to Mol object
    mol = mol.GetMol() 
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    smile = Chem.MolToSmiles(mol)
    return mol, smile, check(smile)

def visualize(element, mask, bond, aroma, charge, reactant=None):
    mol, smile, _ = result2mol((element, mask, bond, aroma, charge, reactant))
    array = mol2array(mol)
    return array, smile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import torch
import wandb

from utils.viz import log_tsne_delta
from utils.viz import log_tsne_reaction_field
from utils.viz import log_tsne_by_type
from utils.viz import log_tsne_true_flow_by_type
from utils.viz import log_cos_hist
from utils.viz import (
    # log_reaction_field_by_type_shared_pca,
    log_norm_stats_by_type,
    log_path_similarity_by_type,
    log_delta_diagnostics,
)


from contextlib import contextmanager
from copy import deepcopy


from contextlib import contextmanager
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
import wandb

# ===== 你原来用到的可视化函数（保持不变）=====
from utils.viz import log_tsne_delta
from utils.viz import log_tsne_reaction_field
from utils.viz import log_tsne_by_type
from utils.viz import log_tsne_true_flow_by_type
from utils.viz import log_cos_hist
from utils.viz import (
    log_norm_stats_by_type,
    log_path_similarity_by_type,
)

# 你已有的 result2mol
# from xxx import result2mol


@contextmanager
def override_decoder_cfg(model, delta_source=None, input_mode=None):
    """
    临时覆盖 model.decoder_cfg.delta_source / input_mode（如果存在），退出后恢复。
    """
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
    支持：
      - "tf" / "flow" / "ode"
      - {"name": "...", "delta_source": "...", "input_mode": "..."}
    返回: (mode_name_for_log, delta_source, input_mode)
    """
    if isinstance(mode, dict):
        name = mode.get("name", "custom")
        delta_source = mode.get("delta_source", None)
        input_mode = mode.get("input_mode", None)
        return str(name), delta_source, input_mode

    mode_str = str(mode).lower()
    # 简写：把 mode_str 当 delta_source；input_mode 默认 None（沿用模型配置）
    return mode_str, mode_str, None


def _call_sample_structures(model, tensors_gpu, delta_source, input_mode, temperature, nfe=20, ode_method="heun", atol=1e-4, rtol=1e-4, options=None):
    """
    使用 decoder_cfg 控制 sample_structures
    """
    # 使用 override decoder_cfg
    if hasattr(model, "decoder_cfg") and (getattr(model, "decoder_cfg", None) is not None):
        with override_decoder_cfg(model, delta_source=delta_source, input_mode=input_mode):
            # print( "========================= override decoder_cfg , in _call_sample_structures ========================= ", delta_source, input_mode)
            res =  model.sample_structures(
                tensors=tensors_gpu, 
                temperature=temperature, 
                nfe=nfe, 
                ode_method=ode_method,
                atol=atol,
                rtol=rtol,
                options=options
            )
            # print( "========================= override ended ========================= ", )
            return res
    else:
        # 如果没有 decoder_cfg，直接调用
        # print( "========================= no decoder_cfg, in _call_sample_structures ========================= ")
        return model.sample_structures(
            tensors=tensors_gpu, 
            temperature=temperature, 
            nfe=nfe, 
            ode_method=ode_method,
            atol=atol,
            rtol=rtol,
            options=options
        )


@torch.no_grad()
def evaluate_smiles(
    model,
    dataloader,
    device,
    eval_modes=None,
    temperature: float = 0.7,
    max_batches: int = None,
    log_prefix: str = "val",
    num_examples: int = 8,
    epoch: int = None,
    tag: str = "val",
    disable_tqdm: bool = False,
    nfe: int = 20,
    ode_method: str = "heun",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    options: dict = None,
):
    """
    多模式评估（强制保留 TF）+ 保留你那套可视化 + wandb examples table。
    """
    base = model.module if hasattr(model, "module") else model
    base.eval()
    model.eval()

    # ---- modes ----
    if eval_modes is None:
        eval_modes = ["tf"]
    # 强制保留 TF
    # has_tf = any((m == "tf") or (isinstance(m, dict) and m.get("name") == "tf") for m in eval_modes)
    # if not has_tf:
    #     eval_modes = ["tf"] + list(eval_modes)

    pool = ProcessPoolExecutor(2)
    all_results = {}

    # 只做一次可视化（避免每个 mode 都打一遍，太吵）
    did_viz = False

    for mode in eval_modes:
        mode_name, delta_source, input_mode = _parse_mode(mode)
        # print( "=========================", mode_name, delta_source, input_mode, "=========================")
        true_cnt = 0
        cnt = 0
        example_records = []
        tag = f"{log_prefix}_{mode_name}"
        pbar = tqdm(dataloader, desc=f"SMILES-Eval({log_prefix},{mode_name},{tag})", disable=disable_tqdm)

        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # ===== 合成 tensors dict（CPU）=====
            tensors = {}
            for k, v in batch.reactant.data.items():
                tensors[k] = v
            if hasattr(batch.condition, "data") and isinstance(batch.condition.data, dict):
                for k, v in batch.condition.data.items():
                    tensors[k] = v
            
            element   = tensors["element"]          # [B, L]
            src_bond  = tensors["src_bond"]
            tgt_bond  = tensors["tgt_bond"]
            src_mask  = tensors["src_mask"]
            tgt_mask  = tensors["tgt_mask"]
            src_aroma = tensors["src_aroma"].bool().long()
            src_charge = tensors["src_charge"]
            tgt_aroma = tensors["tgt_aroma"].bool().long()
            tgt_charge = tensors["tgt_charge"]
            condition_embedding = None
            if "condition_embedding" in tensors:
                condition_embedding = tensors["condition_embedding"]
            B, L = element.shape
            cnt += B

            # ===== tensors to GPU（喂给模型）=====
            tensors_gpu = {}
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor):
                    tensors_gpu[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    tensors_gpu[k] = [t.to(device) for t in v]
                else:
                    tensors_gpu[k] = v
            if batch_idx == 0:
                delta_true_g, delta_pred_g, z0_graph, zhat_graph, r_type = base.get_graph_delta_true_pred_z0_zhat(tensors_gpu)
                log_delta_diagnostics(delta_true_g, delta_pred_g, flow_scale=base.flow_scale, tag=f"{tag}", step=epoch) 
            else:
                delta_true_g = None
                delta_pred_g = None
                z0_graph = None
                zhat_graph = None
                r_type = None
            # ===== 你的可视化：只打一次（epoch%10==2 且 batch0）=====
            if (not did_viz) and (batch_idx == 0) and (epoch is not None) and (epoch % 10 == 2):
                # 这里按你原逻辑：用 GPU tensors 调 get_graph_delta...
                
                log_tsne_delta(
                    delta_true=delta_true_g,
                    delta_pred=delta_pred_g,
                    r_type=r_type,
                    tag=f"tsne/{tag}",
                    step=epoch,
                )
                log_tsne_reaction_field(z0_graph=z0_graph, zhat_graph=zhat_graph, r_type=r_type, tag=f"{tag}", step=epoch)
                log_tsne_by_type(delta_true_g, r_type, tag=f"{tag}_true")
                log_tsne_true_flow_by_type(delta_true_g, delta_pred_g, r_type, tag=f"{tag}_both")
                log_cos_hist(delta_true_g, delta_pred_g, tag=f"{tag}")

                log_norm_stats_by_type(
                    delta_true=delta_true_g,
                    delta_pred=delta_pred_g,
                    r_type=r_type,
                    tag=f"{tag}",
                    step=epoch,
                    max_types=12,
                    make_hist_images=True,
                )
                log_path_similarity_by_type(V=delta_pred_g, r_type=r_type, tag=f"{tag}/pred", step=epoch)
                log_path_similarity_by_type(V=delta_true_g, r_type=r_type, tag=f"{tag}/true", step=epoch)

                did_viz = True

            # ===== 1) GT src/tgt SMILES（CPU，复用你原逻辑）=====
            arg_list_src = [
                (element[i], src_mask[i], src_bond[i], src_aroma[i], src_charge[i], None)
                for i in range(B)
            ]
            src_results = list(pool.map(result2mol, arg_list_src, chunksize=16))
            src_smiles = [item[1] for item in src_results]

            arg_list_tgt = [
                (element[i], tgt_mask[i], tgt_bond[i], tgt_aroma[i], tgt_charge[i], None)
                for i in range(B)
            ]
            tgt_results = list(pool.map(result2mol, arg_list_tgt, chunksize=16))
            tgts = [item[1].split(".") for item in tgt_results]

            # ===== 2) 预测结构（关键：按 mode 切换）=====
            sample_out = _call_sample_structures(
                model=base,
                tensors_gpu=tensors_gpu,
                delta_source=delta_source,
                input_mode=input_mode,
                temperature=temperature,
                nfe=nfe,
                ode_method=ode_method,
                atol=atol,
                rtol=rtol,
                options=options,
            )

            pred_bond   = sample_out["bond"].cpu()
            pred_aroma  = sample_out["aroma"].cpu()
            pred_charge = sample_out["charge"].cpu()

            arg_list_pred = [
                (element[j], src_mask[j], pred_bond[j], pred_aroma[j], pred_charge[j], None)
                for j in range(B)
            ]
            pred_results = list(pool.map(result2mol, arg_list_pred, chunksize=16))
            pred_smiles = [item[1].split(".") for item in pred_results]

            # ===== 3) 逐样本计数 + 前 k 个例子 =====
            for j in range(B):
                ok = True
                for frag in tgts[j]:
                    if frag not in pred_smiles[j]:
                        ok = False
                        break
                if ok:
                    true_cnt += 1

                if len(example_records) < num_examples:
                    example_records.append({
                        "global_idx": cnt - B + j,
                        "src": src_smiles[j],
                        "tgt": ".".join(tgts[j]),
                        "pred": ".".join(pred_smiles[j]),
                        "correct": ok,
                    })

            if cnt > 0:
                acc = true_cnt / cnt
                pbar.set_postfix(acc=f"{acc:.4f}")

        # ===== 4) wandb table + acc per mode =====
        if len(example_records) > 0:
            table = wandb.Table(columns=["epoch", "mode", "idx", "correct", "src", "tgt", "pred"])
            for rec in example_records:
                table.add_data(
                    epoch if epoch is not None else -1,
                    mode_name,
                    int(rec["global_idx"]),
                    bool(rec["correct"]),
                    rec["src"],
                    rec["tgt"],
                    rec["pred"],
                )
            wandb.log({f"{log_prefix}/smiles_examples_{mode_name}": table})

        acc = (true_cnt / cnt) if cnt > 0 else 0.0
        wandb.log({f"{log_prefix}/smiles_acc_{mode_name}": acc})
        print(f"[SMILES-Eval][{log_prefix}][{mode_name}] accuracy = {acc:.4f}")

        all_results[mode_name] = {"acc": acc, "mode": mode_name}

    return all_results