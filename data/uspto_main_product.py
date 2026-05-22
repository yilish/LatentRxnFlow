# data/uspto_main_product.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from random import shuffle


# ========= 你原来的 TransformerDataset（略微清理了一下） =========

class TransformerDataset(Dataset):
    """
    原始反应数据集：每个样本是一个 dict，
    包含 element, src_bond, charge, aroma, mask, reactant_tokens, product_tokens 等。
    """

    def __init__(self, if_shuffle: bool, data: List[Dict[str, Any]]):
        self.data = data  # list of dict
        self.shuffle = if_shuffle
        self.cat_indices = self._precompute_indices()

        # 把需要 tensor 化的字段转成 tensor
        condition_fp_count = 0
        condition_num_count = 0
        for idx, feature_dict in enumerate(self.data):
            for key in feature_dict:
                if (
                    "smile" in key
                    or "action" in key
                    or "idx" in key
                    or "reactant_tokens" in key
                    or "product_tokens" in key
                    or "mol_mapping" in key
                    or "temp_mapping" in key
                    or "time_mapping" in key
                    or "condition_smiles_list" in key
                    # or "r_type" in key
                ):
                    # 这些暂时保持原始 python 对象（list / str），后面再用
                    continue
                if isinstance(feature_dict[key], str):
                    continue
                # Handle condition_fp: convert to torch.uint8 tensor
                if key == "condition_fp":
                    val = feature_dict[key]
                    # Handle empty arrays
                    if isinstance(val, np.ndarray):
                        if val.size == 0:
                            feature_dict[key] = torch.zeros((0, 512), dtype=torch.uint8)
                        else:
                            feature_dict[key] = torch.from_numpy(val).to(torch.uint8)
                    elif isinstance(val, (list, tuple)):
                        if len(val) == 0:
                            feature_dict[key] = torch.zeros((0, 512), dtype=torch.uint8)
                        else:
                            feature_dict[key] = torch.tensor(val, dtype=torch.uint8)
                    else:
                        feature_dict[key] = torch.tensor(val, dtype=torch.uint8)
                    condition_fp_count += 1
                    continue
                # Handle condition_num: convert to torch.long scalar tensor
                if key == "condition_num":
                    val = feature_dict[key]
                    if isinstance(val, (int, float, np.integer)):
                        feature_dict[key] = torch.tensor(int(val), dtype=torch.long)
                    elif torch.is_tensor(val):
                        feature_dict[key] = val.to(torch.long)
                    else:
                        feature_dict[key] = torch.tensor(int(val), dtype=torch.long)
                    condition_num_count += 1
                    continue
                tmp = torch.tensor(feature_dict[key])
                if "aroma" in key or "mask" in key:
                    feature_dict[key] = tmp.bool()
                elif "laplacian" in key:
                    feature_dict[key] = tmp.float()
                else:
                    feature_dict[key] = tmp.long()

    def __len__(self) -> int:
        return len(self.data)

    def preprocess_data(self, idx: int) -> Dict[str, Any]:
        data = self.data[idx]

        if self.shuffle:
            length = data["element"].shape[0]
            index = list(range(length))
            shuffle(index)

            index_t = torch.tensor(index, dtype=torch.long)

            # 为 bond 映射准备 reverse index
            reverse = torch.zeros_like(index_t)
            # reverse[new_pos] = old_pos
            for new_pos, old_pos in enumerate(index_t):
                reverse[old_pos] = new_pos
            reverse = reverse.unsqueeze(1).expand(-1, data["src_bond"].shape[1])

            new_data: Dict[str, Any] = {}

            for key, value in data.items():
                # 1) bond 相关：需要用 double permutation
                if "bond" in key:
                    v = value[index_t]                # 先按新顺序重排源 index
                    v = torch.gather(reverse, 0, v)   # 再把 index 映射到新编号
                    new_data[key] = v

                # 2) 反应类型，转 int 即可，不参与 shuffle
                elif key == "r_type":
                    new_data[key] = int(value)

                # 3) 这些是全局/非 per-atom 特征，不要按原子 shuffle
                elif key in (
                    "sfp",
                    "reactant",
                    "reactant_tokens",
                    "product_tokens",
                    "smiles",
                    "idx",
                    "mol_mapping",
                    "temp_mapping",
                    "time_mapping",
                    "actions",
                    "tokenized_actions",
                    "tokenized_actions_only",
                    "solvent",
                    "catalyst",
                    "condition_embedding",
                    "condition_smiles",
                    "condition_smiles_list",
                    "condition_fp",
                    "condition_num",
                ):
                    new_data[key] = value

                # 4) 其它 tensor：只有在 dim0 == length 时才按原子 shuffle
                elif torch.is_tensor(value) and value.dim() > 0 and value.size(0) == length:
                    new_data[key] = value[index_t]

                # 5) 剩下的（标量、长度不匹配的 tensor、字符串…）不动
                else:
                    new_data[key] = value

            data = new_data

        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.preprocess_data(idx)

    @staticmethod
    def collate_dicts(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        原来的 collate_fn：把 list[dict] pad 成 batch dict。
        """
        max_len = max(i["element"].shape[0] for i in data_list)
        batch: Dict[str, Any] = {}

        for key in data_list[0]:
            if key in ("reactant_tokens", "product_tokens"):
                batch[key] = [i[key] for i in data_list]
            elif key == "condition_embedding":
                batch[key] = torch.stack([i[key] for i in data_list])
            elif key == "condition_num":
                # Stack condition_num to [B] long tensor
                lst = []
                for i, item in enumerate(data_list):
                    if key in item:
                        val = item[key]
                        if torch.is_tensor(val):
                            t = val
                        else:
                            t = torch.tensor(val, dtype=torch.long)
                        lst.append(t)
                    else:
                        # Missing key - default to 0
                        lst.append(torch.tensor(0, dtype=torch.long))
                batch[key] = torch.stack(lst)
            elif key == "condition_smiles_list":
                # Keep as List[List[str]] (no tensorization)
                batch[key] = [i[key] for i in data_list]
            elif key == "condition_fp":
                # Pad condition_fp to [B, Nc_max, 512] using pad_sequence
                lst = []
                for i, item in enumerate(data_list):
                    if key in item:
                        val = item[key]
                        if torch.is_tensor(val):
                            t = val
                        else:
                            # Convert numpy array or list to tensor
                            if isinstance(val, np.ndarray):
                                t = torch.from_numpy(val).to(torch.uint8)
                            else:
                                t = torch.tensor(val, dtype=torch.uint8)
                        lst.append(t)
                    else:
                        # Missing key - create empty tensor
                        lst.append(torch.zeros((0, 512), dtype=torch.uint8))
                
                # Use pad_sequence if we have any tensors (even if some are empty)
                # pad_sequence can handle empty tensors correctly
                if len(lst) > 0:
                    # Ensure all tensors have the same number of dimensions (2D: [Nc, 512])
                    # pad_sequence expects 2D tensors
                    lst_2d = []
                    for t in lst:
                        if t.dim() == 0:
                            # Scalar - shouldn't happen, but handle it
                            lst_2d.append(torch.zeros((0, 512), dtype=torch.uint8))
                        elif t.dim() == 1:
                            # 1D tensor [512] - reshape to [1, 512]
                            lst_2d.append(t.unsqueeze(0))
                        elif t.dim() == 2:
                            # Already 2D [Nc, 512] - use as is
                            lst_2d.append(t)
                        else:
                            # Higher dim - shouldn't happen, but flatten
                            lst_2d.append(t.view(-1, 512))
                    
                    batch[key] = pad_sequence(lst_2d, batch_first=True, padding_value=0)
                else:
                    # Handle case where no samples have condition_fp at all
                    B = len(data_list)
                    batch[key] = torch.zeros((B, 0, 512), dtype=torch.uint8)
            elif key in (
                "idx",
                "actions",
                "smiles",
                "mol_mapping",
                "temp_mapping",
                "time_mapping",
            ):
                # 暂时不进 batch
                continue
            elif key == "condition_smiles":
                # Keep as List[str] for debugging
                batch[key] = [i[key] for i in data_list]
            elif key in ("r_type", "temperature"):
                batch[key] = torch.tensor([int(i[key]) for i in data_list])

            elif key in ("tokenized_actions", "tokenized_actions_only"):
                lst = [torch.tensor(i[key]) for i in data_list]
                batch[key] = pad_sequence(lst, batch_first=True, padding_value=0)

            elif key in ("solvent", "catalyst"):
                lst = [i[key][0] for i in data_list]
                batch[key] = pad_sequence(lst, batch_first=True, padding_value=0)
            elif key == "scaled_yield":
                batch[key] = torch.tensor([float(i[key]) for i in data_list])
            else:
                lst = [i[key] for i in data_list]
                batch[key] = pad_sequence(lst, batch_first=True, padding_value=1)

        return batch

    def _precompute_indices(self):
        category_indices: Dict[Any, List[int]] = {}
        self.types: List[Any] = []
        for idx, item in enumerate(self.data):
            if "r_type" not in item:
                return None
            r_type = item["r_type"]
            if r_type not in category_indices:
                category_indices[r_type] = []
            category_indices[r_type].append(idx)
            self.types.append(r_type)
        return category_indices


# ========= 为了配合 train_one_epoch 封装的 batch 类型 =========

@dataclass
class GraphBatch:
    """
    保存图结构相关的张量（原子、键等）。
    内部用一个 dict 装所有张量，并提供 .to(device) 方法。
    """
    data: Dict[str, torch.Tensor]

    def to(self, device: torch.device) -> "GraphBatch":
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device)
        return self


@dataclass
class ConditionBatch:
    """
    保存条件相关特征（actions, solvent, catalyst, temp, r_type 等）。
    """
    data: Dict[str, Any]  # Changed to Any to support List[List[str]] for condition_smiles_list

    def to(self, device: torch.device) -> "ConditionBatch":
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device)
        return self


@dataclass
class FlowBatch:
    """
    最终 DataLoader 返回的 batch，和 train_one_epoch 中的用法一致：
      batch.reactant
      batch.condition
      batch.product_latent
      batch.product_token
    """
    reactant: GraphBatch
    condition: ConditionBatch
    product_latent: torch.Tensor  # 先占位一个张量
    product_token: torch.Tensor   # (B, L)


# ========= 真正给 DataLoader 用的 Dataset 和 collate_fn =========

class USPTOReact2MainProduct(Dataset):
    """
    适配 Cursor 生成的 train_one_epoch 接口的 Dataset。
    底层用 TransformerDataset 存实际反应数据。
    """

    def __init__(self, data_list, if_shuffle: bool = True):
        super().__init__()
        self.inner = TransformerDataset(if_shuffle=if_shuffle, data=data_list)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 返回单个样本的 dict，后面交给 collate_fn 统一打包
        return self.inner[idx]


def collate_fn(samples: List[Dict[str, Any]]) -> FlowBatch:
    """
    将 list[dict] -> FlowBatch，满足 train_one_epoch 里 batch.xxx 的接口。
    """
    # 先用原来的逻辑拼成一个大 dict
    batch_dict = TransformerDataset.collate_dicts(samples)

    # ----- 1) 构造 reactant 图结构 -----
    # 除去 product_tokens 和 condition 相关字段，其余都视为图特征
    cond_keys = {
        "r_type",
        "temperature",
        "tokenized_actions",
        "tokenized_actions_only",
        "solvent",
        "catalyst",
        "condition_embedding",
        "condition_smiles",
        "condition_smiles_list",
        "condition_fp",
        "condition_num",
    }
    graph_data: Dict[str, torch.Tensor] = {}
    for k, v in batch_dict.items():
        if k in cond_keys:
            continue
        if k in ("product_tokens", "reactant_tokens"):
            continue
        graph_data[k] = v
    reactant = GraphBatch(graph_data)

    # ----- 2) 构造 condition -----
    cond_data: Dict[str, Any] = {}  # Changed to Any to support List[List[str]] for condition_smiles_list
    for k in cond_keys:
        if k in batch_dict:
            cond_data[k] = batch_dict[k]
   
    condition = ConditionBatch(cond_data)
    # ----- 3) 构造 product_token（pad 成张量） -----
    # 有 product_tokens 就用 product_tokens，
    # 没有的话就先用 reactant_tokens 顶上，
    # 再没有就造一个长度为 1 的 dummy 序列。
    if "product_tokens" in batch_dict:
        product_token_lists: List[List[int]] = batch_dict["product_tokens"]
    elif "reactant_tokens" in batch_dict:
        product_token_lists = batch_dict["reactant_tokens"]
    else:
        # fallback：全 dummy
        B = len(samples)
        product_token_lists = [[0] for _ in range(B)]

    B = len(product_token_lists)
    lengths = [len(seq) for seq in product_token_lists]
    max_len = max(lengths)

    product_token = torch.zeros((B, max_len), dtype=torch.long)
    for i, seq in enumerate(product_token_lists):
        L = len(seq)
        if L == 0:
            continue
        product_token[i, :L] = torch.tensor(seq, dtype=torch.long)
    # ----- 4) product_latent 先用占位全零张量 -----
    # 这里的维度暂时无所谓，因为后面会在模型里完全忽略 product_latent，
    # 自己用 encoder 计算 latent。
    product_latent = torch.zeros((B, 1), dtype=torch.float32)

    return FlowBatch(
        reactant=reactant,
        condition=condition,
        product_latent=product_latent,
        product_token=product_token,
    )


# 如果之后你还想用工厂函数，也可以保留：
def make_uspto_dataset(data_list, if_shuffle: bool = True) -> USPTOReact2MainProduct:
    return USPTOReact2MainProduct(data_list=data_list, if_shuffle=if_shuffle)