# utils/viz.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb

def log_tsne_delta(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    r_type: np.ndarray,
    tag: str,
    step: int,
    max_points: int = 800,
):
    """
    delta_true: [B, D]  (graph-level Δz_true)
    delta_pred: [B, D]  (graph-level Δz_pred)
    r_type    : [B] or None, reaction type (int or str)
    tag       : 'tsne/train' or 'tsne/val' ...
    step      : global_step
    """

    # --- 下采样，避免太大 ---
    B = delta_true.shape[0]
    if B > max_points:
        idx = np.random.choice(B, max_points, replace=False)
        delta_true = delta_true[idx]
        delta_pred = delta_pred[idx]
        r_type = r_type[idx] if r_type is not None else None
        B = max_points

    # 拼一起做 t-SNE
    all_feats = np.concatenate([delta_true, delta_pred], axis=0)  # [2B, D]
    labels = np.array([0] * B + [1] * B)  # 0=TRUE, 1=PRED

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, B // 10)),
        learning_rate='auto',
        init='random',
        metric='euclidean',
    )
    feats_2d = tsne.fit_transform(all_feats)   # [2B, 2]
    x, y = feats_2d[:, 0], feats_2d[:, 1]

    # --- 画图 ---
    fig, ax = plt.subplots(figsize=(5, 5))

    mask_true = (labels == 0)
    mask_pred = (labels == 1)

    ax.scatter(
        x[mask_true], y[mask_true],
        s=10, alpha=0.7, label='Δz_true'
    )
    ax.scatter(
        x[mask_pred], y[mask_pred],
        s=10, alpha=0.7, marker='x', label='Δz_flow'
    )

    # 可选：按 r_type 分颜色/marker，可以后面再玩

    ax.set_title(f"{tag} (true vs flow)")
    ax.legend(loc='best')
    ax.set_xticks([])
    ax.set_yticks([])

    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)

def log_tsne_reaction_field(
    z0_graph, zhat_graph, r_type, 
    tag="val", step=None, num_classes=None
):
    """
    可视化 reaction latent field:
        每个点 = z0_graph (反应物图 embedding)
        每个箭头 = z0 -> zhat (flow 预测的反应偏移)
        颜色 = reaction type

    z0_graph:   [N, D]
    zhat_graph: [N, D]
    r_type:     [N] int
    """
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # --- 1) 串 z0 和 zhat 一起做一次 t-SNE ---
    X = np.concatenate([z0_graph, zhat_graph], axis=0)  # [2N, D]
    tsne = TSNE(
        n_components=2, perplexity=30,
        learning_rate=200, init="pca", random_state=0
    )
    X_2d = tsne.fit_transform(X)  # [2N, 2]

    N = z0_graph.shape[0]
    z0_2d = X_2d[:N]
    zhat_2d = X_2d[N:]

    # --- 2) 可视化 ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # auto number of classes
    if num_classes is None:
        num_classes = int(np.max(r_type)) + 1
    types = np.unique(r_type)

    # 颜色映射
    colors   = plt.cm.tab10(np.linspace(0, 1, len(types)))

    # --- 点: z0 ---
    for i, c in enumerate(types):
        idx = (r_type == c)
        ax.scatter(
            z0_2d[idx, 0], z0_2d[idx, 1],
            s=20, alpha=0.7,
            color=colors[i],
            label=f"type {c}"
        )

    # --- 箭头: z0 → zhat ---
    for i in range(N):
        ax.arrow(
            z0_2d[i, 0], z0_2d[i, 1],
            zhat_2d[i, 0] - z0_2d[i, 0],
            zhat_2d[i, 1] - z0_2d[i, 1],
            length_includes_head=True,
            head_width=0.5, head_length=0.7,
            alpha=0.4,
            color=colors[int(r_type[i])-1]
        )

    ax.set_title(f"latent-field/{tag}  (z0 → ẑ, colored by reaction type)")
    # ax.legend(markerscale=1.5, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.legend(fontsize=6, markerscale=0.7)

    wandb.log({f"latent_field/{tag}": wandb.Image(fig)})
    plt.close(fig)


       # ---- 5) per-type plots ----
    for t in types:
        idx = np.where(r_type == t)[0]
        if idx.size == 0:
            continue
        c = colors[int(t)-1]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"latent-field/{tag}/type_{t} (n={idx.size})")
        
        all_x = np.concatenate([z0_2d[idx, 0], zhat_2d[idx, 0]])
        all_y = np.concatenate([z0_2d[idx, 1], zhat_2d[idx, 1]])
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()

        padx = 0.05 * (xmax - xmin + 1e-9)
        pady = 0.05 * (ymax - ymin + 1e-9)

        xlim = (xmin - padx, xmax + padx)
        ylim = (ymin - pady, ymax + pady)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        

        ax.scatter(z0_2d[idx, 0], z0_2d[idx, 1], s=14, alpha=0.75, color=c)
        ax.quiver(
            z0_2d[idx, 0], z0_2d[idx, 1],
            zhat_2d[idx, 0] - z0_2d[idx, 0], zhat_2d[idx, 1] - z0_2d[idx, 1],
            angles="xy", scale_units="xy", scale=1.0,
            width=0.003, alpha=0.45, color=c
        )

        # 这类图 legend 没啥意义，别 call legend，避免 warning
        wandb.log({f"latent_field/{tag}/type_{t}": wandb.Image(fig)})
        plt.close(fig)

def log_tsne_by_type(delta_true_g, r_type, tag="val_true"):
    X = delta_true_g
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=200, init="pca", random_state=0)
    X_2d = tsne.fit_transform(X)

    r_type = np.array(r_type)  # [N]
    types = np.unique(r_type)

    fig, ax = plt.subplots(figsize=(5, 5))
    for t in types:
        mask = (r_type == t)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.7, label=str(t))
    ax.set_title(f"tsne/{tag} (Δz_true by type)")
    ax.legend(fontsize=6, markerscale=0.7)
    wandb.log({f"tsne/{tag}_by_type": wandb.Image(fig)})
    plt.close(fig)

def log_tsne_true_flow_by_type(delta_true_g, delta_pred_g, r_type, tag="val_both"):
    X = np.concatenate([delta_true_g, delta_pred_g], axis=0)
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=200, init="pca", random_state=0)
    X_2d = tsne.fit_transform(X)
    N = delta_true_g.shape[0]
    true_2d = X_2d[:N]
    flow_2d = X_2d[N:]
    r_type = np.array(r_type)
    types = np.unique(r_type)
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, t in enumerate(types):
        mask = (r_type == t)
        ax.scatter(true_2d[mask, 0], true_2d[mask, 1], s=10, alpha=0.7, label=f"{t}-true", marker="o", color=colors[i])
        ax.scatter(flow_2d[mask, 0], flow_2d[mask, 1], s=10, alpha=0.7, label=f"{t}-flow", marker="x", color=colors[i])
    ax.set_title(f"tsne/{tag} (true vs flow, colored by type)")
    ax.legend(fontsize=6, markerscale=0.7)
    wandb.log({f"tsne/{tag}_true_flow_by_type": wandb.Image(fig)})
    plt.close(fig)

def log_cos_hist(delta_true_g, delta_pred_g, tag="val"):
    dt = delta_true_g / (np.linalg.norm(delta_true_g, axis=-1, keepdims=True) + 1e-8)  # [N,D]
    dp = delta_pred_g / (np.linalg.norm(delta_pred_g, axis=-1, keepdims=True) + 1e-8)
    cos = np.sum(dt * dp, axis=-1)  # [N]

    fig, ax = plt.subplots()
    ax.hist(cos, bins=30, range=(-1, 1))
    ax.set_title(f"cos(Δz_true, Δz_flow) / {tag}")
    wandb.log({f"hist/cos_{tag}": wandb.Image(fig)})
    plt.close(fig)

    # utils/viz.py
import math
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _unique_types(r_type: torch.Tensor):
    # r_type: [B] int
    rt = r_type.tolist()#.detach().cpu().numpy().tolist()
    return sorted(list(set(int(t) for t in rt)))


def _fit_pca2d_torch(X: torch.Tensor):
    """
    X: [N, D] float
    return mean[D], V[D,2]  (principal axes)
    """
    X = X.detach().float()
    mu = X.mean(dim=0, keepdim=True)  # [1,D]
    Xc = X - mu
    # pca_lowrank is stable and built-in
    # q=2 gives top-2 components
    U, S, V = torch.pca_lowrank(Xc, q=2, center=False)
    # V: [D, q]
    return mu.squeeze(0), V[:, :2]


def _project_pca2d(X: torch.Tensor, mu: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
    """
    X: [N,D], mu:[D], V2:[D,2]  => [N,2]
    """
    return (X.detach().float() - mu[None, :]) @ V2



# utils/viz.py (append)

@torch.no_grad()
def log_norm_stats_by_type(
    delta_true: torch.Tensor,  # [B,D]
    delta_pred: torch.Tensor,  # [B,D]
    r_type: torch.Tensor,      # [B]
    tag: str,
    step: int = None,
    max_types: int = 12,
    make_hist_images: bool = True,
    bins: int = 30,
):
    """
    每个 type 内：
      - ||Δz_true|| / ||Δz_pred|| 分布（hist 可选）
      - 汇总到 wandb.Table：mean/median/std/q10/q90
    """
    rt_np = r_type # .detach().cpu().numpy().astype(int)
    types = _unique_types(r_type)[:max_types]

    true_norm = np.linalg.norm(delta_true, axis=-1)
    pred_norm = np.linalg.norm(delta_pred, axis=-1)

    table = wandb.Table(columns=[
        "type", "n",
        "true_mean", "true_median", "true_std", "true_q10", "true_q90",
        "pred_mean", "pred_median", "pred_std", "pred_q10", "pred_q90",
    ])

    for t in types:
        idx = np.where(rt_np == t)[0]
        if idx.size == 0:
            continue
        tn = true_norm[idx]
        pn = pred_norm[idx]

        def _stats(x):
            return (
                float(np.mean(x)),
                float(np.median(x)),
                float(np.std(x)),
                float(np.quantile(x, 0.10)),
                float(np.quantile(x, 0.90)),
            )

        t_mean, t_med, t_std, t_q10, t_q90 = _stats(tn)
        p_mean, p_med, p_std, p_q10, p_q90 = _stats(pn)

        table.add_data(
            int(t), int(idx.size),
            t_mean, t_med, t_std, t_q10, t_q90,
            p_mean, p_med, p_std, p_q10, p_q90,
        )

        if make_hist_images:
            fig = plt.figure(figsize=(6, 4))
            ax = plt.gca()
            ax.set_title(f"Norm dist | type={t} | n={idx.size}")
            ax.hist(tn, bins=bins, alpha=0.6, label="||Δ_true||")
            ax.hist(pn, bins=bins, alpha=0.6, label="||Δ_pred||")
            ax.set_xlabel("norm")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.2)
            ax.legend()
            wandb.log({f"{tag}/norm_hist/type_{t}": wandb.Image(fig)}, )
            plt.close(fig)

    wandb.log({f"{tag}/norm_stats_table": table}, )

# utils/viz.py (append)

@torch.no_grad()
def log_path_similarity_by_type(
    V: torch.Tensor,          # [B,D] (建议用 graph-level Δz_pred 或 Δz_true；你可以分别跑两次)
    r_type: torch.Tensor,     # [B]
    tag: str,
    step: int = None,
    max_types: int = 12,
    min_n: int = 5,
    max_n_for_heatmap: int = 80,
    eps: float = 1e-8,
):
    """
    对每个 type：
      - pairwise cosine（去对角线）的 mean/std/q10/q90
      - MRL = ||mean(unit vectors)||  ∈ [0,1]
      - 可选：n <= max_n_for_heatmap 时 log cosine heatmap
    """
    rt_np = r_type #.detach().cpu().numpy().astype(int)
    types = _unique_types(r_type)[:max_types]

    table = wandb.Table(columns=[
        "type", "n",
        "mean_cos_offdiag", "std_cos_offdiag", "q10_cos", "q90_cos",
        "MRL",
        "mean_norm", "median_norm",
    ])

    V = V#.detach().float()
    norms = np.linalg.norm(V, axis=-1)  # [B]

    for t in types:
        idx = np.where(rt_np == t)[0]
        n = int(idx.size)
        if n < min_n:
            continue

        Vt = V[idx]  # [n,D]
        nt = norms[idx]
        # unit vectors
        Vn = Vt / (np.linalg.norm(Vt, axis=-1, keepdims=True) + eps)

        # cosine matrix
        Cos = np.clip((Vn @ Vn.T), -1.0, 1.0)  # [n,n]
        mask = ~np.eye(n, dtype=bool)
        off = Cos[mask]

        mean_cos = float(off.mean())
        std_cos  = float(off.std())
        q10 = float(np.quantile(off, 0.10))
        q90 = float(np.quantile(off, 0.90))

        # MRL
        m = Vn.mean(axis=0)
        mrl = float(np.linalg.norm(m, axis=-1, keepdims=True).item())

        table.add_data(
            int(t), n,
            mean_cos, std_cos, q10, q90,
            mrl,
            float(nt.mean()), float(np.median(nt)),
        )

        if n <= max_n_for_heatmap:
            fig = plt.figure(figsize=(5, 4.5))
            ax = plt.gca()
            ax.set_title(f"Cosine heatmap | type={t} | n={n}")
            im = ax.imshow(Cos, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("sample i")
            ax.set_ylabel("sample j")
            wandb.log({f"{tag}/cos_heatmap/type_{t}": wandb.Image(fig)}, )
            plt.close(fig)

    wandb.log({f"{tag}/path_similarity_table": table}, )

import numpy as np
import torch
import wandb

def log_delta_diagnostics(delta_true_g, delta_pred_g, flow_scale, tag: str, step: int):
    """
    delta_true_g, delta_pred_g: numpy arrays [B, D] or torch tensors
    tag: e.g. "val/tf" or "val/flow"
    """
    if isinstance(delta_true_g, torch.Tensor):
        dt = delta_true_g.float()
    else:
        dt = torch.from_numpy(delta_true_g).float()

    if isinstance(delta_pred_g, torch.Tensor):
        dp = delta_pred_g.float()
    else:
        dp = torch.from_numpy(delta_pred_g).float()


    if isinstance(flow_scale, torch.Tensor):
        flow_scale = flow_scale.item()
    else:
        flow_scale = float(flow_scale)
    eps = 1e-8
    dt_norm = dt.norm(dim=-1).clamp_min(eps)
    dp_norm = dp.norm(dim=-1).clamp_min(eps)

    cos = (dt * dp).sum(dim=-1) / (dt_norm * dp_norm)          # [B]
    ratio = (dp_norm / dt_norm)                                 # [B]
    mse = ((dp - dt) ** 2).mean(dim=-1)                          # [B]
    nrmse = (dp - dt).norm(dim=-1) / dt_norm                     # [B]

    # robust stats
    def stats(x):
        return {
            "mean": x.mean().item(),
            "median": x.median().item(),
            "p10": x.kthvalue(max(1, int(0.10 * x.numel()))).values.item(),
            "p90": x.kthvalue(max(1, int(0.90 * x.numel()))).values.item(),
        }

    cos_s = stats(cos)
    ratio_s = stats(ratio)
    mse_s = stats(mse)
    nrmse_s = stats(nrmse)

    
    log = {
        f"diag/{tag}_cos_mean": cos_s["mean"],
        f"diag/{tag}_cos_median": cos_s["median"],
        f"diag/{tag}_cos_p10": cos_s["p10"],
        f"diag/{tag}_cos_p90": cos_s["p90"],

        f"diag/{tag}_ratio_mean": ratio_s["mean"],
        f"diag/{tag}_ratio_median": ratio_s["median"],
        f"diag/{tag}_ratio_p10": ratio_s["p10"],
        f"diag/{tag}_ratio_p90": ratio_s["p90"],

        f"diag/{tag}_mse_mean": mse_s["mean"],
        f"diag/{tag}_nrmse_mean": nrmse_s["mean"],
        f"diag/{tag}_flow_scale": flow_scale,
    }
    # Also log as a table for detailed inspection in wandb
    # import pandas as pd
    # table_data = [{"metric": k, "value": v} for k, v in log.items()]
    # table = wandb.Table(data=table_data, columns=["metric", "value"])
    wandb.log(log)
    # wandb.log({f"diag/{tag}_metrics_table": table})