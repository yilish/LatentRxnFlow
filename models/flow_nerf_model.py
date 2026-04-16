from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import math
import torch
import torch.nn as nn
import logging
from .molecule_nerf import MoleculeVAE

from dataclasses import dataclass

# 尝试导入 torchdiffeq，如果不可用则设为 None
try:
    import torchdiffeq
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    torchdiffeq = None

@dataclass
class DecoderConfig:
    # Δz 来源：tf / flow / ode
    delta_source: str = "tf"
    # decoder 输入：fuse / add / delta / z1
    input_mode: str = "fuse"
    # ODE integration method (for delta_source="ode")
    ode_method: str = "heun"

@dataclass
class FlowSamplingConfig:
    # 采样策略: "uniform" | "edge" | "mixture_edge_uniform" | "beta"
    t_sampling: str = "uniform"

    # edge / mixture_edge_uniform 用
    edge_eps: float = 0.05          # 端点邻域宽度 ε
    edge_sides: str = "left"        # "left" | "both" | "right"
    edge_mix_prob: float = 0.5      # mixture 中 edge 的比例 (0~1)

    # beta 用（更平滑的贴边）
    beta_a: float = 0.5
    beta_b: float = 0.5

    # 数值安全：避免 t=0/1（可选）
    t_min: float = 1e-4
    t_max: float = 1.0 - 1e-4

class SimpleArgs:
    def __init__(self):
        self.local_rank = 0
        self.vae = False
        self.action = False
        self.use_solvent = False
        self.use_catalyst = False
        self.use_temp = False
        self.beta = 0.0


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        args = t[:, None] * self.freq[None, :]
        emb_sin = torch.sin(args)
        emb_cos = torch.cos(args)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        if emb.size(-1) < self.dim:
            pad = torch.zeros(emb.size(0), self.dim - emb.size(-1), device=emb.device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


class FlowHead(nn.Module):
    def __init__(self, latent_dim: int, time_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, h_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h_cond is ignored for unconditional FlowHead
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.mlp(x)

# <<<< MODIFIED: ControlFlowHead 保持 ControlNet 结构，但移除了注释并确保逻辑清晰
class ControlFlowHead(nn.Module):
    def __init__(self, latent_dim: int, time_embed_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        
        # 1. 主干路径 (Base Stream)
        self.base_mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 2. 控制侧支 (Control Stream) 
        # 输入维度是三者之和，因为我们在外部做好了 Broadcast 和 Concat 准备
        self.control_mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim) 
        )
        
        # 3. 关键：零初始化控制门 (Zero-initialized Gate)
        self.zero_conv = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        self.final_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Cache for diagnostic info (detached)
        self._last_alpha = None

    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, h_cond: torch.Tensor) -> torch.Tensor:
        # 此时 z_t, t_emb, h_cond 的第一维 N (有效节点数) 必须完全一致
        alpha = self.alpha.sigmoid()
        # Cache for diagnostic (detached)
        self._last_alpha = alpha.detach()
        
        # 主干输出
        base_feat = self.base_mlp(torch.cat([z_t, t_emb], dim=-1))
        
        # 控制分支输出
        control_input = torch.cat([z_t, t_emb, h_cond], dim=-1)
        control_feat = self.control_mlp(control_input)
        
        # 通过零初始化层进行残差注入
        hidden_feat = alpha * base_feat + (1 - alpha) * self.zero_conv(control_feat)
        
        return self.final_layer(hidden_feat)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Generates affine transformation parameters (gamma, beta) from condition.
    Supports bounded output via tanh scaling.
    """
    def __init__(
        self, 
        cond_dim: int, 
        latent_dim: int, 
        hidden_dim: int, 
        init_zero: bool = True,
        s_gamma: float = 1.0,
        s_beta: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Output: [gamma, beta]
        )
        if init_zero:
            # Zero-initialize the last layer to start with identity transformation
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        
        # Bounded FiLM scaling factors
        self.s_gamma = s_gamma
        self.s_beta = s_beta
    
    def forward(self, h_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_cond: [N, C] condition embedding
        Returns:
            gamma: [N, D] scaling parameter (bounded by tanh)
            beta: [N, D] shifting parameter (bounded by tanh)
        """
        params = self.mlp(h_cond)  # [N, 2*D]
        gamma_raw, beta_raw = torch.chunk(params, 2, dim=-1)  # Each: [N, D]
        
        # Apply bounded tanh scaling
        gamma = self.s_gamma * torch.tanh(gamma_raw)
        beta = self.s_beta * torch.tanh(beta_raw)
        
        return gamma, beta


class FiLMResidualFlowHead(nn.Module):
    """
    Residual-FiLM Flow Head: applies FiLM modulation to base flow output.
    v = base_mlp([z_t, t_emb]) * (1 + gamma) + beta
    """
    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int,
        cond_dim: int,
        hidden_dim: int,
        film_hidden_dim: int = None,
        film_init_zero: bool = True,
        film_s_gamma: float = 1.0,
        film_s_beta: float = 0.2,
    ):
        super().__init__()
        # Base MLP (equivalent to FlowHead)
        self.base_mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # FiLM module (with bounded output)
        if film_hidden_dim is None:
            film_hidden_dim = hidden_dim
        self.film = FiLM(
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dim=film_hidden_dim,
            init_zero=film_init_zero,
            s_gamma=film_s_gamma,
            s_beta=film_s_beta,
        )
        
        # Cache for diagnostic info (detached)
        self._last_gamma = None
        self._last_beta = None
    
    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, h_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_t: [N, D] latent at time t
            t_emb: [N, T] time embedding
            h_cond: [N, C] condition embedding (optional)
        Returns:
            v: [N, D] velocity field
        """
        # Base velocity from unconditional flow
        base_input = torch.cat([z_t, t_emb], dim=-1)
        v = self.base_mlp(base_input)  # [N, D]
        
        # Apply FiLM modulation if condition is provided
        if h_cond is not None:
            h_cond = h_cond.float()
            gamma, beta = self.film(h_cond)  # Each: [N, D]
            # Cache for diagnostic (detached)
            self._last_gamma = gamma.detach()
            self._last_beta = beta.detach()
            v = v * (1 + gamma) + beta  # Residual-style modulation
        else:
            self._last_gamma = None
            self._last_beta = None
        
        return v


class FiLMHiddenFlowHead(nn.Module):
    """
    FiLM-in-Hidden Flow Head: applies FiLM modulation to hidden layer instead of output.
    Structure: h = MLP_hidden([z_t, t_emb]) -> FiLM(h) -> Linear(h2 -> v)
    """
    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int,
        cond_dim: int,
        hidden_dim: int,
        film_hidden_dim: int = None,
        film_init_zero: bool = True,
        film_s_gamma: float = 1.0,
        film_s_beta: float = 0.2,
    ):
        super().__init__()
        # Hidden layer MLP
        self.hidden_mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # FiLM module for hidden layer (gamma_h/beta_h dimension is hidden_dim)
        if film_hidden_dim is None:
            film_hidden_dim = hidden_dim
        self.film = FiLM(
            cond_dim=cond_dim,
            latent_dim=hidden_dim,  # FiLM operates on hidden dimension
            hidden_dim=film_hidden_dim,
            init_zero=film_init_zero,
            s_gamma=film_s_gamma,
            s_beta=film_s_beta,
        )
        
        # Final projection to velocity
        self.final_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Cache for diagnostic info (detached)
        self._last_gamma = None
        self._last_beta = None
    
    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, h_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_t: [N, D] latent at time t
            t_emb: [N, T] time embedding
            h_cond: [N, C] condition embedding (optional)
        Returns:
            v: [N, D] velocity field
        """
        # Compute hidden representation
        base_input = torch.cat([z_t, t_emb], dim=-1)
        h = self.hidden_mlp(base_input)  # [N, H]
        
        # Apply FiLM modulation to hidden layer if condition is provided
        if h_cond is not None:
            h_cond = h_cond.float()
            gamma_h, beta_h = self.film(h_cond)  # Each: [N, H]
            # Cache for diagnostic (detached)
            self._last_gamma = gamma_h.detach()
            self._last_beta = beta_h.detach()
            # Apply FiLM to hidden
            h2 = h * (1 + gamma_h) + beta_h  # [N, H]
        else:
            self._last_gamma = None
            self._last_beta = None
            h2 = h
        
        # Project to velocity
        v = self.final_proj(h2)  # [N, D]
        
        return v


class ResidualAddFlowHead(nn.Module):
    """
    Residual-Add Flow Head: adds condition-dependent residual to base flow.
    v = base_mlp([z_t, t_emb]) + g(cond)
    where g(cond) maps cond to velocity dimension.
    """
    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int,
        cond_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        # Base MLP (equivalent to FlowHead)
        self.base_mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Condition-to-velocity mapping: g(cond) -> v_residual
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, h_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_t: [N, D] latent at time t
            t_emb: [N, T] time embedding
            h_cond: [N, C] condition embedding (optional)
        Returns:
            v: [N, D] velocity field
        """
        # Base velocity from unconditional flow
        base_input = torch.cat([z_t, t_emb], dim=-1)
        v_base = self.base_mlp(base_input)  # [N, D]
        
        # Add condition-dependent residual if condition is provided
        if h_cond is not None:
            h_cond = h_cond.float()
            v_residual = self.cond_mlp(h_cond)  # [N, D]
            v = v_base + v_residual
        else:
            v = v_base
        
        return v


class ConcatFlowHead(nn.Module):
    """
    Simple concatenation-based conditional flow head.
    Directly concatenates [z_t, t_emb, h_cond] and passes through MLP.
    """
    def __init__(self, latent_dim: int, time_embed_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, h_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_t: [N, D] latent at time t
            t_emb: [N, T] time embedding
            h_cond: [N, C] condition embedding (required for this head)
        Returns:
            v: [N, D] velocity field
        """
        if h_cond is None:
            raise ValueError("ConcatFlowHead requires h_cond to be provided")
        x = torch.cat([z_t, t_emb, h_cond], dim=-1)
        return self.mlp(x)


class CondCrossAttnFlowHead(nn.Module):
    """
    Conditional Cross-Attention Flow Head.
    Uses condition as per-molecule virtual tokens and applies cross-attention to inject condition info.
    
    Structure:
    - fp_mlp: Linear(512->E)->SiLU->Linear(E->E) to embed fingerprints
    - fp_to_d: Linear(E->latent_dim) to project tokens to latent dimension
    - cross_attn: MultiheadAttention for cross-attention between nodes and condition tokens
    - base_mlp: MLP([z_t, t_emb]) -> v_base
    - merge: v = v_base + proj(ctx)
    """
    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int,
        fp_dim: int = 512,
        cond_mol_emb_dim: int = 256,
        n_heads: int = 4,
        hidden_dim: int = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim * 2
        
        # fp_mlp: Linear(512->E)->SiLU->Linear(E->E)
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_dim, cond_mol_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_mol_emb_dim, cond_mol_emb_dim),
        )
        
        # fp_to_d: Linear(E->latent_dim) to project tokens to D
        self.fp_to_d = nn.Linear(cond_mol_emb_dim, latent_dim)
        
        # cross_attn: MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # base_mlp: similar to FlowHead, input concat([z_t, t_emb]) output v_base [B,L,D]
        self.base_mlp = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # merge: proj for context (can be Linear or Identity)
        # Zero-initialize proj to start with identity (v = v_base + 0)
        self.proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(
        self,
        z_t: torch.Tensor,  # [B, L, D] or [N, D] (batch-major preferred)
        t_emb: torch.Tensor,  # [B, L, T] or [N, T]
        h_cond: Optional[torch.Tensor] = None,  # Legacy: [N, C] (ignored for this head)
        cond_tokens: Optional[torch.Tensor] = None,  # [B, Nc_max, D]
        cond_token_mask: Optional[torch.Tensor] = None,  # [B, Nc_max] True=padding
        B: int = None,  # Batch size (for shape inference)
        L: int = None,  # Sequence length (for shape inference)
        valid_flat: Optional[torch.Tensor] = None,  # [B*L] or [N] valid mask (optional)
        debug_attn: bool = False,  # Whether to compute and return attention stats
        debug_step: int = 0,  # Current step for debug logging
        debug_attn_every: int = 200,  # Log every N steps
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            z_t: [B, L, D] or [N, D] - latent at time t (batch-major preferred)
            t_emb: [B, L, T] or [N, T] - time embedding
            h_cond: Legacy parameter (ignored)
            cond_tokens: [B, Nc_max, D] - condition tokens (virtual tokens from fingerprints)
            cond_token_mask: [B, Nc_max] - True indicates padding (for key_padding_mask)
            B: Batch size (for shape inference if z_t is flattened)
            L: Sequence length (for shape inference if z_t is flattened)
            valid_flat: Optional valid mask
            debug_attn: Whether to compute attention statistics
            debug_step: Current step for debug logging
            debug_attn_every: Log every N steps
        Returns:
            v: [B, L, D] or [N, D] - velocity field (same shape as z_t)
            debug_stats: Optional dict with attention statistics (if debug_attn=True)
        """
        # Handle flattened input: reshape to [B, L, D] if needed
        if z_t.dim() == 2:
            # Flattened input [N, D] or [B*L, D]
            if B is None or L is None:
                raise ValueError("B and L must be provided when z_t is flattened")
            z_t = z_t.view(B, L, -1)  # [B, L, D]
            t_emb = t_emb.view(B, L, -1)  # [B, L, T]
        
        # Base velocity from unconditional flow
        base_input = torch.cat([z_t, t_emb], dim=-1)  # [B, L, D+T]
        v_base = self.base_mlp(base_input)  # [B, L, D]
        
        # Debug stats dict
        debug_stats = None
        
        # Cross-attention with condition tokens
        if cond_tokens is not None and cond_token_mask is not None:
            # Check if all tokens are masked (would cause NaN in attention)
            B_batch, Nc_max = cond_tokens.shape[:2]
            if Nc_max == 0:
                # No tokens available: fallback to unconditional
                v = v_base
            else:
                # Check if all tokens are masked for each batch
                all_masked = cond_token_mask.all(dim=1)  # [B] True if all tokens masked
                if all_masked.all():
                    # All batches have all tokens masked: fallback to unconditional
                    v = v_base
                else:
                    # cond_tokens: [B, Nc_max, D]
                    # z_t: [B, L, D]
                    # Query: z_t, Key/Value: cond_tokens
                    ctx, attn_weights = self.cross_attn(
                        query=z_t,  # [B, L, D]
                        key=cond_tokens,  # [B, Nc_max, D]
                        value=cond_tokens,  # [B, Nc_max, D]
                        key_padding_mask=cond_token_mask,  # [B, Nc_max] True=padding
                        need_weights=True,
                        average_attn_weights=True,
                    )  # ctx: [B, L, D], attn_weights: [B, L, Nc_max] (batch_first=True)
                    
                    # Check for NaN in ctx (safety check)
                    if torch.isnan(ctx).any():
                        # Fallback to unconditional if NaN detected
                        v = v_base
                    else:
                        # Merge: v = v_base + proj(ctx)
                        v = v_base + self.proj(ctx)  # [B, L, D]
                        
                        # Compute debug statistics
                        if debug_attn and self.training and (debug_step % debug_attn_every == 0):
                            # Ensure attn_weights is [B, L, Nc_max]
                            if attn_weights.dim() == 3 and attn_weights.shape[0] == B_batch:
                                # Already correct shape
                                attn_w = attn_weights  # [B, L, Nc_max]
                            else:
                                # Reshape if needed
                                attn_w = attn_weights.view(B_batch, L, Nc_max)
                            
                            # Compute statistics
                            mask_true_ratio = cond_token_mask.float().mean().item()
                            attn_max_mean = attn_w.max(dim=-1).values.mean().item()
                            
                            # Entropy: -sum(p * log(p))
                            attn_w_clamped = attn_w.clamp_min(1e-9)
                            attn_entropy = -(attn_w_clamped * attn_w_clamped.log()).sum(dim=-1)  # [B, L]
                            attn_entropy_mean = attn_entropy.mean().item()
                            attn_eff_tokens_mean = torch.exp(attn_entropy).mean().item()
                            
                            # Context injection strength
                            ctx_norm_mean = ctx.norm(dim=-1).mean().item()
                            v_base_norm_mean = v_base.norm(dim=-1).mean().item()
                            ctx_ratio = ctx_norm_mean / (v_base_norm_mean + 1e-6)
                            
                            debug_stats = {
                                "flow/attn_mask_true_ratio": mask_true_ratio,
                                "flow/attn_max_mean": attn_max_mean,
                                "flow/attn_entropy_mean": attn_entropy_mean,
                                "flow/attn_eff_tokens_mean": attn_eff_tokens_mean,
                                "flow/ctx_norm_mean": ctx_norm_mean,
                                "flow/ctx_ratio": ctx_ratio,
                                "flow/Nc_max": float(Nc_max),
                            }
                            
                            # Check for uniform attention (warning)
                            attn_mean_per_token = attn_w.mean(dim=(0, 1))  # [Nc_max]
                            valid_tokens = ~cond_token_mask[0]  # Use first batch as reference
                            if valid_tokens.sum() > 0:
                                valid_attn_mean = attn_mean_per_token[valid_tokens].mean().item()
                                expected_mean = 1.0 / valid_tokens.sum().item()
                                if abs(valid_attn_mean - expected_mean) < 0.01:
                                    logging.warning(f"Attention weights appear uniform (mean={valid_attn_mean:.4f}, expected={expected_mean:.4f})")
        else:
            # No condition tokens: fallback to unconditional
            v = v_base
        
        # Return in same format as input
        # If input was flattened and valid_flat is provided, we could mask here
        # But for now, we return [B, L, D] and let caller handle masking
        return v, debug_stats


class FlowNERFModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_embed_dim: int,
        ntoken: int = 128,
        args: SimpleArgs | None = None,
        flow_weight: float = 1e-2,
        detach_encoder_for_flow: bool = True,
        decoder_cfg: DecoderConfig = None,
        flow_sampling_cfg: FlowSamplingConfig = None,
        fm_sigma: float = 0.0,
        use_conditional_flow: bool = False,
        nfe: int = 20,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.ntoken = ntoken
        self.nfe = nfe

        if args is None:
            args = SimpleArgs()
        self.args = args

        # ====== NERF backbone ======
        self.backbone = MoleculeVAE(
            args=self.args,
            ntoken=self.ntoken,
            dim=self.latent_dim,
            nlayer=8,
            nhead=8,
            dropout=0.1,
        )
        self.use_conditional_flow = use_conditional_flow
        
        # ====== Flow matching 模块 ======
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Read flow_cond_head configuration from args or use default
        if hasattr(args, "model") and hasattr(args.model, "flow_cond_head"):
            self.flow_cond_head = str(args.model.flow_cond_head).lower()
        else:
            # Default: "controlnet" for backward compatibility
            self.flow_cond_head = "controlnet"
        
        # Read FiLM-specific configs
        if hasattr(args, "model") and hasattr(args.model, "film_hidden_dim"):
            self.film_hidden_dim = int(args.model.film_hidden_dim)
        else:
            self.film_hidden_dim = latent_dim * 2  # Default to same as hidden_dim
        
        if hasattr(args, "model") and hasattr(args.model, "film_init_zero"):
            self.film_init_zero = bool(args.model.film_init_zero)
        else:
            self.film_init_zero = True  # Default: zero initialization
        
        # Read bounded FiLM scaling factors
        if hasattr(args, "model") and hasattr(args.model, "film_s_gamma"):
            self.film_s_gamma = float(args.model.film_s_gamma)
        else:
            self.film_s_gamma = 1.0  # Default: s_gamma = 1.0
        
        if hasattr(args, "model") and hasattr(args.model, "film_s_beta"):
            self.film_s_beta = float(args.model.film_s_beta)
        else:
            self.film_s_beta = 0.2  # Default: s_beta = 0.2
        
        # Read condition dropout probability
        if hasattr(args, "model") and hasattr(args.model, "cond_drop_prob"):
            self.cond_drop_prob = float(args.model.cond_drop_prob)
        else:
            self.cond_drop_prob = 0.2  # Default: p_cond_drop = 0.2
        
        # ====== Debug configuration (read early, before flow_head creation) ======
        # Read debug flags from args.model or use defaults (all False)
        if hasattr(args, "model") and hasattr(args.model, "debug_attn"):
            self.debug_attn = bool(args.model.debug_attn)
        else:
            self.debug_attn = False
        
        if hasattr(args, "model") and hasattr(args.model, "debug_attn_every"):
            self.debug_attn_every = int(args.model.debug_attn_every)
        else:
            self.debug_attn_every = 200
        
        
        if hasattr(args, "model") and hasattr(args.model, "debug_grad"):
            self.debug_grad = bool(args.model.debug_grad)
        else:
            self.debug_grad = False
        
        if hasattr(args, "model") and hasattr(args.model, "debug_grad_every"):
            self.debug_grad_every = int(args.model.debug_grad_every)
        else:
            self.debug_grad_every = 200
        
        # Step counter for debug logging
        self._debug_step = 0
        
        # Debug buffers for gradient norms (will be set by hooks)
        if self.debug_grad:
            self.register_buffer("_dbg_grad_attn", torch.tensor(0.0))
            self.register_buffer("_dbg_grad_fp", torch.tensor(0.0))
        
        # Build flow head based on configuration
        if not self.use_conditional_flow:
            # Unconditional flow head
            self.flow_head = FlowHead(
                latent_dim=latent_dim,
                time_embed_dim=time_embed_dim,
                hidden_dim=latent_dim * 2,
            )
        else:
            # Conditional flow head: select type based on flow_cond_head
            if self.flow_cond_head == "controlnet":
                self.flow_head = ControlFlowHead(
                    latent_dim=latent_dim,
                    time_embed_dim=time_embed_dim,
                    cond_dim=cond_dim,
                    hidden_dim=latent_dim * 2,
                )
            elif self.flow_cond_head == "film_residual":
                self.flow_head = FiLMResidualFlowHead(
                    latent_dim=latent_dim,
                    time_embed_dim=time_embed_dim,
                    cond_dim=cond_dim,
                    hidden_dim=latent_dim * 2,
                    film_hidden_dim=self.film_hidden_dim,
                    film_init_zero=self.film_init_zero,
                    film_s_gamma=self.film_s_gamma,
                    film_s_beta=self.film_s_beta,
                )
            elif self.flow_cond_head == "film_hidden":
                self.flow_head = FiLMHiddenFlowHead(
                    latent_dim=latent_dim,
                    time_embed_dim=time_embed_dim,
                    cond_dim=cond_dim,
                    hidden_dim=latent_dim * 2,
                    film_hidden_dim=self.film_hidden_dim,
                    film_init_zero=self.film_init_zero,
                    film_s_gamma=self.film_s_gamma,
                    film_s_beta=self.film_s_beta,
                )
            elif self.flow_cond_head == "residual_add":
                self.flow_head = ResidualAddFlowHead(
                    latent_dim=latent_dim,
                    time_embed_dim=time_embed_dim,
                    cond_dim=cond_dim,
                    hidden_dim=latent_dim * 2,
                )
            elif self.flow_cond_head == "concat":
                self.flow_head = ConcatFlowHead(
                    latent_dim=latent_dim,
                    time_embed_dim=time_embed_dim,
                    cond_dim=cond_dim,
                    hidden_dim=latent_dim * 2,
                )
            elif self.flow_cond_head == "cond_attn":
                # Read cond_attn-specific configs
                fp_dim = 512  # Fixed fingerprint dimension
                if hasattr(args, "model") and hasattr(args.model, "cond_mol_emb_dim"):
                    cond_mol_emb_dim = int(args.model.cond_mol_emb_dim)
                else:
                    cond_mol_emb_dim = 256  # Default
                if hasattr(args, "model") and hasattr(args.model, "cond_attn_n_heads"):
                    n_heads = int(args.model.cond_attn_n_heads)
                else:
                    n_heads = 4  # Default
                self.flow_head = CondCrossAttnFlowHead(
                    latent_dim=latent_dim,
                    time_embed_dim=time_embed_dim,
                    fp_dim=fp_dim,
                    cond_mol_emb_dim=cond_mol_emb_dim,
                    n_heads=n_heads,
                    hidden_dim=latent_dim * 2,
                )
                # Register gradient hooks after head is created
                if self.debug_grad:
                    self._register_grad_hooks()
            else:
                raise ValueError(
                    f"Unknown flow_cond_head='{self.flow_cond_head}'. "
                    f"Expected one of: 'controlnet', 'film_residual', 'film_hidden', 'residual_add', 'concat', 'cond_attn'"
                )
        
        # Log configuration
        logging.info(
            f"[Flow] use_conditional_flow={self.use_conditional_flow}, "
            f"flow_cond_head={self.flow_cond_head}"
        )
        if self.use_conditional_flow and self.flow_cond_head == "film_residual":
            logging.info(
                f"[Flow] FiLM config: film_hidden_dim={self.film_hidden_dim}, "
                f"film_init_zero={self.film_init_zero}"
            )
            # Verify zero initialization if enabled
            if self.film_init_zero and hasattr(self.flow_head, 'film'):
                film_last_layer = self.flow_head.film.mlp[-1]
                weight_norm = film_last_layer.weight.norm().item()
                bias_norm = film_last_layer.bias.norm().item()
                logging.info(
                    f"[Flow] FiLM zero-init verification: weight_norm={weight_norm:.6f}, "
                    f"bias_norm={bias_norm:.6f} (should be ~0.0)"
                )
                if weight_norm >= 1e-6 or bias_norm >= 1e-6:
                    logging.warning(
                        f"FiLM zero initialization may not be perfect: "
                        f"weight_norm={weight_norm}, bias_norm={bias_norm}"
                    )
            
        if isinstance(flow_weight, torch.Tensor):
            flow_weight = flow_weight.item()
        self.flow_weight = float(flow_weight)
        self.detach_encoder_for_flow = detach_encoder_for_flow
        if flow_sampling_cfg is None:
            flow_sampling_cfg = FlowSamplingConfig()
        elif isinstance(flow_sampling_cfg, dict):
            # Convert dict to FlowSamplingConfig dataclass
            flow_sampling_cfg = FlowSamplingConfig(**flow_sampling_cfg)
        self.flow_sampling_cfg = flow_sampling_cfg
        self.fm_sigma = float(fm_sigma)
        
        # ====== ΔZ → decoder 的融合 MLP ======
        self.delta_fuser = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )  
         
        if decoder_cfg is None:
            decoder_cfg = DecoderConfig()

        self.decoder_cfg = decoder_cfg
        self.flow_scale = nn.Parameter(torch.tensor(1.0))
        if hasattr(args, "model") and hasattr(args.model, "fm_sigma"):
            self.fm_sigma = float(args.model.fm_sigma)
            print(f"Using fm_sigma: {self.fm_sigma}")
        else:
            self.fm_sigma = 0.05
            print(f"Using default fm_sigma: {self.fm_sigma}")
        
        # ====== Condition FP embedding modules ======
        # Default hyperparameters
        self.cond_fp_dim = 512  # Fixed: fingerprint dimension
        self.cond_mol_emb_dim = 256  # Per-molecule embedding dimension
        
        # Read cond_pool from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "cond_pool"):
            self.cond_pool = str(args.model.cond_pool).lower()
        else:
            self.cond_pool = "gated"  # Default: gated pooling (A2)
        
        # Read force_zero_cond from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "force_zero_cond"):
            self.force_zero_cond = bool(args.model.force_zero_cond)
        else:
            self.force_zero_cond = False  # Default: use normal condition
        
        # Cache for gated pooling diagnostics
        self._last_gate_mean = None
        # Cache for condition dropout rate
        self._last_cond_drop_rate = 0.0
        
        # Per-molecule FP embedder (MLP)
        self.cond_fp_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        # Gated pooling: gate MLP for attention weights (A2)
        if self.cond_pool == "gated":
            # MLP_gate: Linear(E->E/2)->SiLU->Linear(E/2->1)
            self.cond_gate_mlp = nn.Sequential(
                nn.Linear(256, 128),  # E/2 = 256/2 = 128
                nn.SiLU(),
                nn.Linear(128, 1),
            )
        else:
            self.cond_gate_mlp = None
        
        # Projection layers to ensure final cond_dim consistency
        # Legacy condition_embedding: 640 -> cond_dim
        if cond_dim != 640:
            self.cond_project_legacy = nn.Linear(640, cond_dim)
        else:
            self.cond_project_legacy = None
        
        # Pooled FP embedding: 256 -> cond_dim
        if cond_dim != 256:
            self.cond_project_fp = nn.Linear(256, cond_dim)
        else:
            self.cond_project_fp = None
        
        # ====== Layer-1 flow-only 排雷模式配置 ======
        # Read layer1_flow_only from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "layer1_flow_only"):
            self.layer1_flow_only = bool(args.model.layer1_flow_only)
        else:
            self.layer1_flow_only = False  # Default: False
        
        # Read layer1_freeze_backbone from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "layer1_freeze_backbone"):
            self.layer1_freeze_backbone = bool(args.model.layer1_freeze_backbone)
        else:
            self.layer1_freeze_backbone = False  # Default: False
        
        # If layer1_freeze_backbone is enabled, freeze backbone and related modules
        if self.layer1_freeze_backbone:
            self.freeze_for_layer1()
        
        # Read layer1_debug_shape_asserts from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "layer1_debug_shape_asserts"):
            self.layer1_debug_shape_asserts = bool(args.model.layer1_debug_shape_asserts)
        else:
            self.layer1_debug_shape_asserts = False  # Default: False (disable asserts in production)
        
        # ====== Flow objective and sanity check configuration ======
        # Read flow_objective from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "flow_objective"):
            flow_obj = str(args.model.flow_objective).lower()
            if flow_obj not in ["fm", "dz"]:
                raise ValueError(f"flow_objective must be 'fm' or 'dz', got '{flow_obj}'")
            self.flow_objective = flow_obj
        else:
            self.flow_objective = "fm"  # Default: flow matching
        
        # Read log_flow_sanity from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "log_flow_sanity"):
            self.log_flow_sanity = bool(args.model.log_flow_sanity)
        else:
            self.log_flow_sanity = True  # Default: True
        
        # Read flow_loss_reduce from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "flow_loss_reduce"):
            reduce_mode = str(args.model.flow_loss_reduce).lower()
            if reduce_mode not in ["sum", "mean"]:
                raise ValueError(f"flow_loss_reduce must be 'sum' or 'mean', got '{reduce_mode}'")
            self.flow_loss_reduce = reduce_mode
        else:
            self.flow_loss_reduce = "sum"  # Default: sum (保持现有行为)
        
        # Read lambda_norm from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "lambda_norm"):
            self.lambda_norm = float(args.model.lambda_norm)
            print(f"Using lambda_norm: {self.lambda_norm}")
        else:
            self.lambda_norm = 10  # Default: 0.02
            print(f"Using default lambda_norm: {self.lambda_norm}")
        
        # Read flow_dir_loss from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "flow_dir_loss"):
            self.flow_dir_loss = bool(args.model.flow_dir_loss)
        else:
            self.flow_dir_loss = True  # Default: True
        
        # Read lambda_mag from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "lambda_mag"):
            self.lambda_mag = float(args.model.lambda_mag)
        else:
            self.lambda_mag = 0.5  # Default: 0.5
        
        # Read lambda_end from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "lambda_end"):
            self.lambda_end = float(args.model.lambda_end)
            logging.info(f"Using lambda_end: {self.lambda_end}")
        else:
            self.lambda_end = 0.0  # Default: 0.0 (disabled by default)
            logging.info(f"Using default lambda_end: {self.lambda_end}")
        
        # Initialize auxiliary losses dictionary
        self._aux_losses = {}
        
        # Read flow loss decomposition weights from args.model or use defaults
        if hasattr(args, "model") and hasattr(args.model, "w_mag_graph"):
            self.w_mag_graph = float(args.model.w_mag_graph)
        else:
            self.w_mag_graph = 0.1  # Default: 0.1
        logging.info(f"Using w_mag_graph: {self.w_mag_graph}")
        
        if hasattr(args, "model") and hasattr(args.model, "w_mag_res"):
            self.w_mag_res = float(args.model.w_mag_res)
        else:
            self.w_mag_res = 0.1  # Default: 0.1
        logging.info(f"Using w_mag_res: {self.w_mag_res}")
        
        if hasattr(args, "model") and hasattr(args.model, "w_var_res"):
            self.w_var_res = float(args.model.w_var_res)
        else:
            self.w_var_res = 0.05  # Default: 0.05
        logging.info(f"Using w_var_res: {self.w_var_res}")
        
        # Read flow_eps from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "flow_eps"):
            self.flow_eps = float(args.model.flow_eps)
        else:
            self.flow_eps = 1e-6  # Default: 1e-6
        
        # Read dir_alpha_node from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "dir_alpha_node"):
            self.dir_alpha_node = float(args.model.dir_alpha_node)
        else:
            self.dir_alpha_node = 0.3  # Default: 0.3
        
        # Read dir_graph_tau from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "dir_graph_tau"):
            self.dir_graph_tau = float(args.model.dir_graph_tau)
        else:
            self.dir_graph_tau = 1e-3  # Default: 1e-3
        
        # Read flow_loss_mode from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "flow_loss_mode"):
            self.flow_loss_mode = str(args.model.flow_loss_mode).lower()
        else:
            self.flow_loss_mode = "decomp_dir"  # Default: decomp_dir
        
        # Read res_thr from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "res_thr"):
            self.res_thr = float(args.model.res_thr)
        else:
            self.res_thr = 1e-3  # Default: 1e-3
        
        # Read alpha_res from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "alpha_res"):
            self.alpha_res = float(args.model.alpha_res)
        else:
            self.alpha_res = 0.2  # Default: 0.2
        
        # Read w_max for residual reweight from args.model or use default
        if hasattr(args, "model") and hasattr(args.model, "res_w_max"):
            self.res_w_max = float(args.model.res_w_max)
        else:
            self.res_w_max = 5.0  # Default: 5.0
        
        # Read freeze/unfreeze configuration
        if hasattr(args, "model") and hasattr(args.model, "freeze_encoder"):
            self.freeze_encoder = bool(args.model.freeze_encoder)
        else:
            self.freeze_encoder = True  # Default: True
        
        if hasattr(args, "model") and hasattr(args.model, "freeze_decoder"):
            self.freeze_decoder = bool(args.model.freeze_decoder)
        else:
            self.freeze_decoder = True  # Default: True
        
        if hasattr(args, "model") and hasattr(args.model, "unfreeze_enc_last_n_layers"):
            self.unfreeze_enc_last_n_layers = int(args.model.unfreeze_enc_last_n_layers)
        else:
            self.unfreeze_enc_last_n_layers = 0  # Default: 0
        
        if hasattr(args, "model") and hasattr(args.model, "unfreeze_dec_first_n_layers"):
            self.unfreeze_dec_first_n_layers = int(args.model.unfreeze_dec_first_n_layers)
        else:
            self.unfreeze_dec_first_n_layers = 0  # Default: 0
        
        if hasattr(args, "model") and hasattr(args.model, "unfreeze_layernorm"):
            self.unfreeze_layernorm = bool(args.model.unfreeze_layernorm)
        else:
            self.unfreeze_layernorm = True  # Default: True
        
        if hasattr(args, "model") and hasattr(args.model, "unfreeze_decoder_head"):
            self.unfreeze_decoder_head = bool(args.model.unfreeze_decoder_head)
        else:
            self.unfreeze_decoder_head = True  # Default: True
        
        # Call set_trainable_modules at the end of __init__
        self.set_trainable_modules()

    # ---------- Set trainable modules ----------
    
    def set_trainable_modules(self):
        """
        冻结全部参数，然后根据配置选择性解冻
        encoder / decoder / flow / condition 模块。
        """
        # 1) 先冻结全部参数
        for p in self.parameters():
            p.requires_grad = False
        
        # 2) 始终解冻 flow head 与 condition 相关模块
        if hasattr(self, "flow_head"):
            for p in self.flow_head.parameters():
                p.requires_grad = True
        
        # Condition 相关模块
        if hasattr(self, "time_embed"):
            for p in self.time_embed.parameters():
                p.requires_grad = True
        
        if hasattr(self, "cond_fp_mlp"):
            for p in self.cond_fp_mlp.parameters():
                p.requires_grad = True
        
        if hasattr(self, "cond_gate_mlp") and self.cond_gate_mlp is not None:
            for p in self.cond_gate_mlp.parameters():
                p.requires_grad = True
        
        if hasattr(self, "cond_project_legacy") and self.cond_project_legacy is not None:
            for p in self.cond_project_legacy.parameters():
                p.requires_grad = True
        
        if hasattr(self, "cond_project_fp") and self.cond_project_fp is not None:
            for p in self.cond_project_fp.parameters():
                p.requires_grad = True
        
        # CondCrossAttnFlowHead 中的 condition 相关模块
        if hasattr(self, "flow_head") and hasattr(self.flow_head, "fp_mlp"):
            for p in self.flow_head.fp_mlp.parameters():
                p.requires_grad = True
        
        if hasattr(self, "flow_head") and hasattr(self.flow_head, "fp_to_d"):
            for p in self.flow_head.fp_to_d.parameters():
                p.requires_grad = True
        
        # 3) 解冻 encoder 最后 N 层（如果 N > 0 且 freeze_encoder=False）
        # 注意：如果 layer1_freeze_backbone=True，则跳过此步骤以保持兼容性
        if (not self.freeze_encoder and self.unfreeze_enc_last_n_layers > 0 and 
            not (hasattr(self, "layer1_freeze_backbone") and self.layer1_freeze_backbone)):
            if hasattr(self.backbone, "M_encoder") and hasattr(self.backbone.M_encoder, "transformer_encoder"):
                encoder_layers = self.backbone.M_encoder.transformer_encoder.layers
                N = min(self.unfreeze_enc_last_n_layers, len(encoder_layers))
                for layer in encoder_layers[-N:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        
        # 4) 解冻 decoder 最前 N 层（如果 N > 0 且 freeze_decoder=False）
        # 注意：如果 layer1_freeze_backbone=True，则跳过此步骤以保持兼容性
        if (not self.freeze_decoder and self.unfreeze_dec_first_n_layers > 0 and 
            not (hasattr(self, "layer1_freeze_backbone") and self.layer1_freeze_backbone)):
            if hasattr(self.backbone, "M_decoder") and hasattr(self.backbone.M_decoder, "transformer_encoder"):
                decoder_layers = self.backbone.M_decoder.transformer_encoder.layers
                N = min(self.unfreeze_dec_first_n_layers, len(decoder_layers))
                for layer in decoder_layers[:N]:
                    for p in layer.parameters():
                        p.requires_grad = True
        
        # 5) 如果 unfreeze_layernorm=True，解冻所有 LayerNorm
        if self.unfreeze_layernorm:
            for m in self.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True
        
        # 6) 如果 unfreeze_decoder_head=True，解冻 decoder 输出 head
        if self.unfreeze_decoder_head:
            if hasattr(self.backbone, "M_decoder"):
                decoder = self.backbone.M_decoder
                if hasattr(decoder, "bond_decoder"):
                    for p in decoder.bond_decoder.parameters():
                        p.requires_grad = True
                if hasattr(decoder, "charge_head"):
                    for p in decoder.charge_head.parameters():
                        p.requires_grad = True
                if hasattr(decoder, "aroma_head"):
                    for p in decoder.aroma_head.parameters():
                        p.requires_grad = True
                if hasattr(decoder, "latent_head"):
                    for p in decoder.latent_head.parameters():
                        p.requires_grad = True

    # ---------- Debug gradient hooks ----------
    
    def _register_grad_hooks(self):
        """Register backward hooks to track gradient norms for debug purposes."""
        if not hasattr(self, "flow_head") or not hasattr(self.flow_head, "cross_attn"):
            return
        
        def make_grad_hook(buffer_name):
            def hook(grad):
                if grad is not None:
                    norm = grad.norm().item()
                    setattr(self, buffer_name, torch.tensor(norm))
            return hook
        
        # Hook for cross_attn out_proj weight
        if hasattr(self.flow_head.cross_attn, "out_proj"):
            self.flow_head.cross_attn.out_proj.weight.register_hook(
                make_grad_hook("_dbg_grad_attn")
            )
        
        # Hook for cond_fp_mlp last layer weight
        if hasattr(self, "cond_fp_mlp") and len(self.cond_fp_mlp) > 0:
            last_layer = self.cond_fp_mlp[-1]
            if hasattr(last_layer, "weight"):
                last_layer.weight.register_hook(
                    make_grad_hook("_dbg_grad_fp")
                )
    
    # ---------- Layer-1 freeze method ----------
    
    def freeze_for_layer1(self):
        """
        冻结 backbone + delta_fuser + flow_scale，只让 condition 模块与 flow head 可训练。
        可训练模块：
        - time_embed
        - flow_head
        - cond_fp_mlp
        - cond_gate_mlp (if exists)
        - cond_project_legacy (if exists)
        - cond_project_fp (if exists)
        """
        # Freeze backbone (encoder + decoder)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Freeze delta_fuser
        for param in self.delta_fuser.parameters():
            param.requires_grad = False
        
        # Freeze flow_scale (optional, but typically frozen in layer1 mode)
        self.flow_scale.requires_grad = False
        
        # Ensure condition modules and flow_head are trainable
        for param in self.time_embed.parameters():
            param.requires_grad = True
        
        for param in self.flow_head.parameters():
            param.requires_grad = True
        
        for param in self.cond_fp_mlp.parameters():
            param.requires_grad = True
        
        if self.cond_gate_mlp is not None:
            for param in self.cond_gate_mlp.parameters():
                param.requires_grad = True
        
        if self.cond_project_legacy is not None:
            for param in self.cond_project_legacy.parameters():
                param.requires_grad = True
        
        if self.cond_project_fp is not None:
            for param in self.cond_project_fp.parameters():
                param.requires_grad = True
        
        logging.info("[Layer1] Frozen backbone, delta_fuser, and flow_scale. Only condition modules and flow_head are trainable.")
    
    def print_trainable_params_summary(self):
        """
        打印可训练参数统计，用于 layer1_flow_only 模式的 sanity check。
        打印 requires_grad=True 的参数名数量与总 trainable params，并确认 backbone 为 0。
        """
        trainable_params = []
        trainable_param_count = 0
        backbone_trainable_count = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                trainable_param_count += param.numel()
                if "backbone" in name:
                    backbone_trainable_count += param.numel()
        
        logging.info("=" * 60)
        logging.info("[Layer1 Sanity Check] Trainable Parameters Summary:")
        logging.info(f"  Total trainable parameter names: {len(trainable_params)}")
        logging.info(f"  Total trainable parameters: {trainable_param_count:,}")
        logging.info(f"  Backbone trainable parameters: {backbone_trainable_count:,}")
        
        if self.layer1_freeze_backbone:
            if backbone_trainable_count > 0:
                logging.warning(f"  ⚠️  WARNING: Backbone has {backbone_trainable_count:,} trainable params (should be 0)!")
            else:
                logging.info("  ✅ Backbone correctly frozen (0 trainable params)")
        
        # Print trainable module names (first 20)
        if len(trainable_params) > 0:
            logging.info(f"  Trainable parameter names (showing first 20):")
            for name in trainable_params[:20]:
                param = dict(self.named_parameters())[name]
                logging.info(f"    - {name}: {param.shape} ({param.numel():,} params)")
            if len(trainable_params) > 20:
                logging.info(f"    ... and {len(trainable_params) - 20} more")
        logging.info("=" * 60)

    # ---------- Condition vector builder ----------
    
    def build_cond_tokens(self, tensors: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build condition tokens from condition_fp for cond_attn head.
        
        Args:
            tensors: Dictionary containing condition_fp and condition_num
        Returns:
            cond_tokens: [B, Nc_max, D] or None if condition_fp not available
            cond_token_mask: [B, Nc_max] True=padding, or None
        """
        if "condition_fp" not in tensors or "condition_num" not in tensors:
            return None, None
        
        condition_fp = tensors["condition_fp"]  # [B, Nc_max, 512] uint8
        condition_num = tensors["condition_num"]  # [B]
        B = condition_fp.shape[0]
        Nc_max = condition_fp.shape[1]
        device = condition_fp.device
        
        # Handle Nc_max == 0 or condition_num == 0 case
        if Nc_max == 0:
            # Return zero tokens with shape [B, 0, D]
            cond_tokens = torch.zeros(B, 0, self.latent_dim, device=device, dtype=torch.float32)
            cond_token_mask = torch.ones(B, 0, device=device, dtype=torch.bool)
            return cond_tokens, cond_token_mask
        
        # Convert uint8 to float32 (bit vector 0/1, no division by 255)
        fp_float = condition_fp.to(torch.float32)  # [B, Nc_max, 512]
        
        # Embed per-molecule fingerprints using flow_head's fp_mlp (only for cond_attn)
        if self.flow_cond_head == "cond_attn" and hasattr(self.flow_head, "fp_mlp"):
            fp_emb = self.flow_head.fp_mlp(fp_float)  # [B, Nc_max, cond_mol_emb_dim]
            # Project to latent_dim
            cond_tokens = self.flow_head.fp_to_d(fp_emb)  # [B, Nc_max, D]
        else:
            # This should not happen if called correctly, but provide fallback
            raise RuntimeError("build_cond_tokens should only be called when flow_cond_head == 'cond_attn'")
        
        # Build token mask: True for padding positions
        # token_mask[b, j] = True when j >= condition_num[b]
        indices = torch.arange(Nc_max, device=device).unsqueeze(0).expand(B, -1)  # [B, Nc_max]
        cond_token_mask = indices >= condition_num.unsqueeze(-1)  # [B, Nc_max] True=padding
        
        return cond_tokens, cond_token_mask
    
    def build_condition_vector(self, tensors: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Build condition vector from available condition fields.
        Priority:
        1) condition_fp + condition_num (new fields)
        2) condition_embedding (legacy field)
        3) None if neither available
        
        Returns:
            cond_flat: [B, cond_dim] or None
        """
        cond_flat = None
        
        # Priority 1: Use condition_fp + condition_num if available
        if "condition_fp" in tensors and "condition_num" in tensors:
            condition_fp = tensors["condition_fp"]  # [B, Nc_max, 512] uint8
            condition_num = tensors["condition_num"]  # [B]
            B = condition_fp.shape[0]
            
            # Handle Nc_max == 0 or condition_num == 0 case
            if condition_fp.shape[1] == 0:
                # Return zero pooled embedding
                pooled = torch.zeros(B, 256, device=condition_fp.device, dtype=torch.float32)
            else:
                # Convert uint8 to float32 (bit vector 0/1, no division by 255)
                fp_float = condition_fp.to(torch.float32)
                
                # Generate mask: mask[b, j] = 1 when j < condition_num[b]
                Nc_max = condition_fp.shape[1]
                device = condition_fp.device
                indices = torch.arange(Nc_max, device=device).unsqueeze(0).expand(B, -1)  # [B, Nc_max]
                mask = (indices < condition_num.unsqueeze(-1)).float()  # [B, Nc_max]
                
                # Embed per-molecule fingerprints
                fp_emb = self.cond_fp_mlp(fp_float)  # [B, Nc_max, 256]
                
                # Pooling: mean or gated (A2)
                if self.cond_pool == "gated":
                    # Gated pooling: sigmoid gate weights + weighted sum
                    gate_w = torch.sigmoid(self.cond_gate_mlp(fp_emb)).squeeze(-1)  # [B, Nc_max]
                    # Apply mask: only valid molecules contribute
                    masked_w = gate_w * mask  # [B, Nc_max]
                    # Weighted sum with normalization
                    w_sum = masked_w.sum(dim=1, keepdim=True)  # [B, 1]
                    eps = 1e-6
                    # Normalize weights
                    masked_w_norm = masked_w / (w_sum + eps)  # [B, Nc_max]
                    # Weighted sum of embeddings
                    pooled = (fp_emb * masked_w_norm.unsqueeze(-1)).sum(dim=1)  # [B, 256]
                    # For condition_num==0 rows: return zeros
                    mask_sum = mask.sum(dim=1, keepdim=True)  # [B, 1]
                    pooled = torch.where(mask_sum > 0, pooled, torch.zeros_like(pooled))
                    # Cache gate mean for diagnostics (detached)
                    self._last_gate_mean = masked_w_norm[mask > 0].mean().detach() if (mask > 0).any() else None
                else:
                    # Masked mean pooling (DeepSet)
                    masked_sum = (fp_emb * mask.unsqueeze(-1)).sum(dim=1)  # [B, 256]
                    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
                    pooled = masked_sum / denom  # [B, 256]
                    # For condition_num==0 rows, masked_sum==0 and denom==1, so pooled==0 (correct)
            
            # Project to cond_dim if needed
            if self.cond_project_fp is not None:
                cond_flat = self.cond_project_fp(pooled)  # [B, cond_dim]
            else:
                cond_flat = pooled  # [B, 256] == [B, cond_dim]
        
        # Priority 2: Fall back to legacy condition_embedding
        elif "condition_embedding" in tensors:
            cond_emb = tensors["condition_embedding"]  # [B, 640]
            
            # Project to cond_dim if needed
            if self.cond_project_legacy is not None:
                cond_flat = self.cond_project_legacy(cond_emb)  # [B, cond_dim]
            else:
                cond_flat = cond_emb  # [B, 640] == [B, cond_dim]
        
        # Priority 3: No condition available
        # cond_flat remains None
        
        # Apply force_zero_cond if enabled (only if condition exists)
        if self.force_zero_cond and cond_flat is not None:
            # Return zero tensor with correct shape
            cond_flat = torch.zeros_like(cond_flat)
        
        # Apply condition dropout (A3) - only in training mode
        if cond_flat is not None:
            cond_flat = self.maybe_drop_condition(cond_flat)
        
        return cond_flat
    
    def maybe_drop_condition(self, cond_flat: torch.Tensor) -> torch.Tensor:
        """
        Apply condition dropout during training (A3).
        Randomly zeros out entire condition vectors with probability cond_drop_prob.
        
        Args:
            cond_flat: [B, cond_dim] condition vector
        Returns:
            cond_flat: [B, cond_dim] (possibly dropped out)
        """
        if not self.training or self.cond_drop_prob <= 0.0:
            # Cache actual drop rate (0.0 when disabled)
            self._last_cond_drop_rate = 0.0
            return cond_flat
        
        B = cond_flat.shape[0]
        # Generate dropout mask: [B] of booleans
        drop_mask = torch.rand(B, device=cond_flat.device) < self.cond_drop_prob
        # Cache actual drop rate for monitoring (detached)
        self._last_cond_drop_rate = drop_mask.float().mean().detach().item()
        # Apply dropout: zero out entire rows
        cond_flat = cond_flat * (~drop_mask).float().unsqueeze(-1)
        
        return cond_flat

    # ---------- node-level encode 工具 ----------

    @torch.no_grad()
    def decode_from_latent(
        self,
        z_t: torch.Tensor,          # [B,L,D]
        src_bond: torch.Tensor,
        src_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
    # TODO: ADD logics with cond_flat
        out = self.backbone.M_decoder.forward_logits(
            src_embedding=z_t,
            src_bond=src_bond,
            padding_mask=src_mask,
            temperature=temperature,
            action_embedding=None,
            temps=None,
            solvent_embedding=None,
            catalyst_embedding=None,
        )
        return out

    def _sample_t(self, N: int, device: torch.device) -> torch.Tensor:
        cfg = self.flow_sampling_cfg
        # Support both dict and FlowSamplingConfig dataclass
        if cfg is None:
            mode = "uniform"
        elif isinstance(cfg, dict):
            mode = (cfg.get("t_sampling") or "uniform").lower()
        else:
            # FlowSamplingConfig dataclass
            mode = (cfg.t_sampling or "uniform").lower()

        if mode == "uniform":
            t = torch.rand(N, device=device)

        elif mode == "edge":
            # 只在端点邻域采样：left/right/both
            eps = float(cfg.get("edge_eps") if isinstance(cfg, dict) else cfg.edge_eps)
            edge_sides = cfg.get("edge_sides") if isinstance(cfg, dict) else cfg.edge_sides
            if edge_sides == "left":
                t = eps * torch.rand(N, device=device)
            elif edge_sides == "right":
                t = 1.0 - eps * torch.rand(N, device=device)
            elif edge_sides == "both":
                u = torch.rand(N, device=device)
                left = eps * torch.rand(N, device=device)
                right = 1.0 - eps * torch.rand(N, device=device)
                t = torch.where(u < 0.5, left, right)
            else:
                raise ValueError(f"Unknown edge_sides={edge_sides}")

        elif mode == "mixture_edge_uniform":
            # 以 p 在 edge 采样，(1-p) 在全区间采样
            p = float(cfg.get("edge_mix_prob") if isinstance(cfg, dict) else cfg.edge_mix_prob)
            eps = float(cfg.get("edge_eps") if isinstance(cfg, dict) else cfg.edge_eps)
            edge_sides = cfg.get("edge_sides") if isinstance(cfg, dict) else cfg.edge_sides
            u = torch.rand(N, device=device)
            t_uni = torch.rand(N, device=device)

            if edge_sides == "left":
                t_edge = eps * torch.rand(N, device=device)
            elif edge_sides == "right":
                t_edge = 1.0 - eps * torch.rand(N, device=device)
            elif edge_sides == "both":
                u2 = torch.rand(N, device=device)
                left = eps * torch.rand(N, device=device)
                right = 1.0 - eps * torch.rand(N, device=device)
                t_edge = torch.where(u2 < 0.5, left, right)
            else:
                raise ValueError(f"Unknown edge_sides={edge_sides}")

            t = torch.where(u < p, t_edge, t_uni)

        elif mode == "beta":
            # Beta(a,b)；a,b<1 会贴两端
            a = float(cfg.get("beta_a") if isinstance(cfg, dict) else cfg.beta_a)
            b = float(cfg.get("beta_b") if isinstance(cfg, dict) else cfg.beta_b)
            dist = torch.distributions.Beta(a, b)
            t = dist.sample((N,)).to(device)

        else:
            t_sampling_val = cfg.get("t_sampling") if isinstance(cfg, dict) else (cfg.t_sampling if cfg else None)
            raise ValueError(f"Unknown t_sampling={t_sampling_val}")

        # 可选：clamp，避免精确 0/1
        if cfg:
            # t_min = cfg.get("t_min") if isinstance(cfg, dict) else cfg.t_min
            # t_max = cfg.get("t_max") if isinstance(cfg, dict) else cfg.t_max
            # t = t.clamp(min=float(t_min), max=float(t_max))
            pass
        return t

    def _encode_src_tgt_nodes(
        self, tensors: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_enc, src_mask = self.backbone.encode(tensors, which="src")
        tgt_enc, tgt_mask = self.backbone.encode(tensors, which="tgt")
        return src_enc, tgt_enc, src_mask, tgt_mask

    def sample_structures(
        self,
        tensors: Dict[str, Any],
        temperature: float = 1.0,
        nfe: int = 20,
        ode_method: str = "heun",
        atol: float = 1e-4,
        rtol: float = 1e-4,
        options: Optional[Dict] = None,
    ):
        src_enc, tgt_enc, src_mask, tgt_mask = self._encode_src_tgt_nodes(tensors)  # [L,B,D]
        
        cond_flat = self.build_condition_vector(tensors)
        
        delta_z_true = tgt_enc - src_enc
        _, delta_z_flow, _ = self._flow_forward(src_enc, tgt_enc, src_mask, cond_flat=cond_flat, tensors=tensors)

        z1_hat = self._build_decoder_embedding(
            src_enc=src_enc, 
            tgt_enc=tgt_enc, 
            src_mask=src_mask, 
            delta_z_true=delta_z_true, 
            delta_z_flow=delta_z_flow, 
            nfe=nfe, 
            ode_method=ode_method,
            atol=atol,
            rtol=rtol,
            options=options,
            cond_flat=cond_flat
        )
        # 调用 MoleculeDecoder.sample，注意 src_bond / src_mask 仍然是 reactant 的
        result = self.backbone.M_decoder.sample(
            src_embedding=z1_hat,
            src_bond=tensors["src_bond"],
            padding_mask=tensors["src_mask"],
            temperature=temperature,
            action_embedding=None,
            temps=None,
            solvent_embedding=None,
            catalyst_embedding=None,
        )
        return result

    def flow_v(self, z_t, t_emb, cond_flat=None, cond_tokens=None, cond_token_mask=None, B=None, L=None, valid_flat=None, debug_attn=False, debug_step=0, debug_attn_every=200):
        """
        Unified interface for flow velocity computation.
        All flow heads now support optional h_cond parameter.
        For cond_attn head, also supports cond_tokens and cond_token_mask.
        """
        if not self.use_conditional_flow:
            # Unconditional: h_cond is ignored
            result = self.flow_head(z_t=z_t, t_emb=t_emb, h_cond=None)
            if isinstance(result, tuple):
                v, debug_stats = result
                return self.flow_scale * v, debug_stats
            else:
                return self.flow_scale * result, None
        else:
            # Conditional: pass cond_flat to head
            # For cond_attn head, also pass cond_tokens and cond_token_mask
            if self.flow_cond_head == "cond_attn":
                result = self.flow_head(
                    z_t=z_t,
                    t_emb=t_emb,
                    h_cond=cond_flat,  # Legacy parameter (ignored)
                    cond_tokens=cond_tokens,
                    cond_token_mask=cond_token_mask,
                    B=B,
                    L=L,
                    valid_flat=valid_flat,
                    debug_attn=debug_attn,
                    debug_step=debug_step,
                    debug_attn_every=debug_attn_every,
                )
                if isinstance(result, tuple):
                    v, debug_stats = result
                    return self.flow_scale * v, debug_stats
                else:
                    return self.flow_scale * result, None
            else:
                # Other conditional heads: pass cond_flat to head
                # cond_flat is already aligned with z_t: [N, C] or [B*L, C]
                result = self.flow_head(z_t=z_t, t_emb=t_emb, h_cond=cond_flat)
                if isinstance(result, tuple):
                    v, debug_stats = result
                    return self.flow_scale * v, debug_stats
                else:
                    return self.flow_scale * result, None

    @torch.no_grad()
    def sample_intermediate_structures(self, tensors: Dict[str, Any], temperature: float = 1.0, n_steps: int = 10):
        src_enc, tgt_enc, src_mask, tgt_mask = self._encode_src_tgt_nodes(tensors)  # [L,B,D]
        z0 = src_enc
        src_mask = tensors["src_mask"]
        L, B, D = z0.shape
        device = z0.device
        z = z0.clone()
        
        cond_flat = self.build_condition_vector(tensors)  # [B, C] or None
        valid = (~src_mask).bool()              # [B,L]
        valid_LBD = valid.t().unsqueeze(-1)     # [L,B,1]

        # <<<< MODIFIED: 预先处理 condition 的广播，避免在循环里重复计算
        # h_cond [B, C] -> [B, 1, C] -> [B, L, C] -> [B*L, C]
        # 注意顺序：Flow forward 里的 z 是 z.permute(1,0,2) 即 [Batch-Major]，所以 cond 也要 batch-major 展开
        if cond_flat is not None:
            cond_expanded_flat = cond_flat.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)
        else:
            cond_expanded_flat = None

        dt = 1.0 / n_steps
        heatmaps = []
        for i in range(n_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=device)
            t_emb = self.time_embed(t)  # [B,T]
            # Time 也要对齐到 [B*L, T]
            t_flat = t_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)

            # z: [L,B,D] -> [B,L,D] -> [B*L,D]
            z_flat = z.permute(1, 0, 2).reshape(B * L, D)
            
            # 使用对齐好的 cond_expanded_flat
            v0_flat, _ = self.flow_v(z_t=z_flat, t_emb=t_flat, cond_flat=cond_expanded_flat, B=B, L=L)
            v0 = v0_flat.view(B, L, D).permute(1, 0, 2) * valid_LBD
            z_euler = z + dt * v0

            z_euler_flat = z_euler.permute(1, 0, 2).reshape(B * L, D)
            v1_flat, _ = self.flow_v(z_t=z_euler_flat, t_emb=t_flat, cond_flat=cond_expanded_flat, B=B, L=L)
            v1 = v1_flat.view(B, L, D).permute(1, 0, 2) * valid_LBD

            z = z + 0.5 * dt * (v0 + v1)
            z_fuse = self.delta_fuser(torch.cat([z0, z], dim=-1))
            zt_logits = self.decode_from_latent(z_fuse, tensors["src_bond"], tensors["src_mask"], temperature)
            heatmaps.append(zt_logits)
        return torch.stack(heatmaps, dim=1)


    def _flow_forward(
        self,
        src_enc: torch.Tensor,
        tgt_enc: torch.Tensor,
        src_mask: torch.Tensor,
        cond_flat: torch.Tensor=None, # [B, C]
        override_cond_valid: Optional[torch.Tensor] = None,  # [N, C] for debug
        return_metrics: bool = False,  # Whether to return alignment metrics
        tensors: Optional[Dict[str, Any]] = None,  # For building cond_tokens (cond_attn head)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        node-level Flow Matching
        
        Args:
            override_cond_valid: If provided, use this directly as cond_valid [N, C]
                (for debug/ablation purposes)
            return_metrics: If True, compute and return alignment metrics (detached)
        
        Returns:
            flow_loss: Flow matching loss
            delta_full: Predicted delta [L, B, D]
            metrics: Optional dict of alignment metrics (if return_metrics=True)
        """
        L, B, D = src_enc.shape
        device = src_enc.device
        
        # Shape asserts (if enabled)
        if self.layer1_debug_shape_asserts:
            assert src_enc.dim() == 3, f"src_enc should be [L, B, D], got {src_enc.shape}"
            assert tgt_enc.dim() == 3, f"tgt_enc should be [L, B, D], got {tgt_enc.shape}"
            assert src_enc.shape == tgt_enc.shape, f"src_enc {src_enc.shape} != tgt_enc {tgt_enc.shape}"
            assert src_mask.dim() == 2, f"src_mask should be [B, L], got {src_mask.shape}"
            assert src_mask.shape[0] == B, f"src_mask batch size {src_mask.shape[0]} != B {B}"
            assert src_mask.shape[1] == L, f"src_mask length {src_mask.shape[1]} != L {L}"
            if cond_flat is not None:
                assert cond_flat.dim() == 2, f"cond_flat should be [B, C], got {cond_flat.shape}"
                assert cond_flat.shape[0] == B, f"cond_flat batch size {cond_flat.shape[0]} != B {B}"

        if self.detach_encoder_for_flow:
            src_fm = src_enc.detach()
            tgt_fm = tgt_enc.detach()
        else:
            src_fm = src_enc
            tgt_fm = tgt_enc

        # 1. 转置为 Batch-Major [B, L, D]
        z0 = src_fm.permute(1, 0, 2)
        z1 = tgt_fm.permute(1, 0, 2)

        valid_mask = (~src_mask).bool()  # [B, L] True = 有效节点
        
        # Check if using cond_attn head (requires batch-major processing)
        use_cond_attn = (self.use_conditional_flow and 
                        self.flow_cond_head == "cond_attn" and 
                        tensors is not None and 
                        "condition_fp" in tensors)
        
        # Initialize debug_stats for collection
        debug_stats_from_flow = None
        
        if use_cond_attn:
            # ====== CondAttn path: batch-major processing ======
            # Build cond_tokens and cond_token_mask
            cond_tokens, cond_token_mask = self.build_cond_tokens(tensors)  # [B, Nc_max, D], [B, Nc_max]
            
            # Safety check: if cond_tokens is None or empty, fallback to legacy path
            if cond_tokens is None or cond_tokens.shape[1] == 0:
                use_cond_attn = False
            
            # Sample t for all positions [B, L] (padding positions will be masked out later)
            t_full = self._sample_t(B * L, device=device).view(B, L)  # [B, L]
            t_emb_full = self.time_embed(t_full.view(B * L)).view(B, L, -1)  # [B, L, T]
            
            # Interpolate zt for all positions [B, L, D]
            t_full_ = t_full.unsqueeze(-1)  # [B, L, 1]
            zt_full = (1.0 - t_full_) * z0 + t_full_ * z1 + self.fm_sigma * torch.randn_like(z0)  # [B, L, D]
            v_target_full = z1 - z0  # [B, L, D]
            
            # Call flow_head with batch-major inputs
            debug_attn_enabled = (self.debug_attn and self.training and 
                                 (self._debug_step % self.debug_attn_every == 0))
            v_pred_full, debug_stats = self.flow_v(
                z_t=zt_full,  # [B, L, D]
                t_emb=t_emb_full,  # [B, L, T]
                cond_flat=None,  # Not used for cond_attn
                cond_tokens=cond_tokens,  # [B, Nc_max, D]
                cond_token_mask=cond_token_mask,  # [B, Nc_max]
                B=B,
                L=L,
                valid_flat=valid_mask.reshape(-1),  # [B*L]
                debug_attn=debug_attn_enabled,
                debug_step=self._debug_step,
                debug_attn_every=self.debug_attn_every,
            )  # [B, L, D], debug_stats: Optional[Dict]
            
            # Add cond_num to debug stats
            if debug_stats is None:
                debug_stats = {}
            if self.training:
                if "condition_num" in tensors:
                    cond_num = tensors["condition_num"]
                    debug_stats["flow/cond_num_mean"] = cond_num.float().mean().item()
                    debug_stats["flow/cond_num_max"] = cond_num.float().max().item()
            
            # Store for later use in metrics
            debug_stats_from_flow = debug_stats
            
            # Extract valid nodes for loss computation
            valid_flat = valid_mask.reshape(-1)  # [B*L]
            N = valid_flat.sum().item()
            if N == 0:
                zero = torch.zeros([], device=device)
                delta_zero = torch.zeros_like(src_enc)
                return zero, delta_zero, None
            
            v_pred = v_pred_full.reshape(B * L, D)[valid_flat]  # [N, D]
            v_target = v_target_full.reshape(B * L, D)[valid_flat]  # [N, D]
            
            # Build delta_full from v_pred_full [B, L, D] -> [L, B, D]
            delta_full = v_pred_full.permute(1, 0, 2)  # [L, B, D]
            
        else:
            # ====== Legacy path: flatten and extract valid nodes ======
            # 2. 展平为 [B*L, D]
            z0_flat = z0.reshape(-1, D)      
            z1_flat = z1.reshape(-1, D)      
            valid_flat = valid_mask.reshape(-1)  # [B*L]

            if valid_flat.sum() == 0:
                zero = torch.zeros([], device=device)
                delta_zero = torch.zeros_like(src_enc)
                return zero, delta_zero, None

            # 3. 提取有效节点
            z0_valid = z0_flat[valid_flat]   # [N, D]
            z1_valid = z1_flat[valid_flat]   # [N, D]
            N = z0_valid.size(0)
            
            # Shape asserts for valid nodes
            if self.layer1_debug_shape_asserts:
                assert z0_valid.shape == (N, D), f"z0_valid should be [N, D], got {z0_valid.shape}"
                assert z1_valid.shape == (N, D), f"z1_valid should be [N, D], got {z1_valid.shape}"

            # 4. <<<< MODIFIED: 处理 Condition 的广播和掩码 >>>>
            # 目标：构建一个 [N, C] 的 tensor，与 z0_valid 的节点一一对应
            cond_valid = None
            if override_cond_valid is not None:
                # Use override (for debug/ablation)
                cond_valid = override_cond_valid  # [N, C]
                if self.layer1_debug_shape_asserts:
                    assert cond_valid.shape == (N, self.cond_dim), f"override_cond_valid should be [N, C], got {cond_valid.shape}"
            elif self.use_conditional_flow and cond_flat is not None:
                # cond_flat: [B, C]
                # 先扩展到与 z0_flat 形状对应: [B, 1, C] -> [B, L, C] -> [B*L, C]
                cond_expanded = cond_flat.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)
                # 再提取有效位
                cond_valid = cond_expanded[valid_flat] # [N, C]
                if self.layer1_debug_shape_asserts:
                    assert cond_expanded.shape == (B * L, self.cond_dim), f"cond_expanded should be [B*L, C], got {cond_expanded.shape}"
                    assert cond_valid.shape == (N, self.cond_dim), f"cond_valid should be [N, C], got {cond_valid.shape}"

            # t ~ U(0,1)
            t = self._sample_t(N, device=device)   # [N]
            t_emb = self.time_embed(t)             # [N, T]
            
            # Shape asserts for time embeddings
            if self.layer1_debug_shape_asserts:
                assert t.shape == (N,), f"t should be [N], got {t.shape}"
                assert t_emb.shape == (N, self.time_embed_dim), f"t_emb should be [N, T], got {t_emb.shape}"

            t_ = t.unsqueeze(-1)                   # [N, 1]
            zt = (1.0 - t_) * z0_valid + t_ * z1_valid + self.fm_sigma * torch.randn_like(z0_valid)   # [N, D]
            v_target = z1_valid - z0_valid             # [N, D]
            
            # Shape asserts for zt and v_target
            if self.layer1_debug_shape_asserts:
                assert zt.shape == (N, D), f"zt should be [N, D], got {zt.shape}"
                assert v_target.shape == (N, D), f"v_target should be [N, D], got {v_target.shape}"
            
            # 5. 调用 FlowHead
            # 此时 zt: [N, D], t_emb: [N, T], cond_valid: [N, C]
            # 维度完全对齐，可以安全 concat
            v_pred, _ = self.flow_v(z_t=zt, t_emb=t_emb, cond_flat=cond_valid)          # [N, D]
            
            # Shape assert for v_pred
            if self.layer1_debug_shape_asserts:
                assert v_pred.shape == (N, D), f"v_pred should be [N, D], got {v_pred.shape}"

        # Check for NaN/Inf in predictions (safety check)
        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            # Replace NaN/Inf with zeros (fallback)
            v_pred = torch.where(torch.isnan(v_pred) | torch.isinf(v_pred), torch.zeros_like(v_pred), v_pred)
        
        if torch.isnan(v_target).any() or torch.isinf(v_target).any():
            # Replace NaN/Inf with zeros (fallback)
            v_target = torch.where(torch.isnan(v_target) | torch.isinf(v_target), torch.zeros_like(v_target), v_target)
        
        # 7. 计算 flow_loss（graph + residual 分解）
        eps = self.flow_eps
        
        # (1) 定义 dz_true = v_target (already z1_valid - z0_valid) [N, D]
        dz_true = v_target  # [N, D]
        
        # (2) 计算 batch_idx [N] - 每个 node 对应的 graph id
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(B * L)  # [B*L]
        batch_idx = batch_ids[valid_flat]  # [N]
        
        # (3) 计算 dz_graph [B, D] = scatter_mean(dz_true, batch_idx)
        ones_valid = torch.ones(N, 1, device=device, dtype=dz_true.dtype)  # [N, 1]
        denom_graph = torch.zeros(B, 1, device=device, dtype=dz_true.dtype)  # [B, 1]
        denom_graph.index_add_(0, batch_idx, ones_valid)  # [B, 1]
        denom_graph = denom_graph.clamp_min(1.0)  # 避免除零
        
        dz_graph_sum = torch.zeros(B, D, device=device, dtype=dz_true.dtype)  # [B, D]
        dz_graph_sum.index_add_(0, batch_idx, dz_true)  # [B, D]
        dz_graph = dz_graph_sum / denom_graph  # [B, D]
        
        # dz_graph_node = dz_graph[batch_idx] [N, D]
        dz_graph_node = dz_graph[batch_idx]  # [N, D]
        
        # (4) dz_res = dz_true - dz_graph_node
        dz_res = dz_true - dz_graph_node  # [N, D]
        
        # (5) 对 v_pred 同样做分解
        # v_graph [B, D] = scatter_mean(v_pred, batch_idx)
        v_graph_sum = torch.zeros(B, D, device=device, dtype=v_pred.dtype)  # [B, D]
        v_graph_sum.index_add_(0, batch_idx, v_pred)  # [B, D]
        v_graph = v_graph_sum / denom_graph  # [B, D]
        
        # v_graph_node = v_graph[batch_idx] [N, D]
        v_graph_node = v_graph[batch_idx]  # [N, D]
        
        # v_res = v_pred - v_graph_node
        v_res = v_pred - v_graph_node  # [N, D]
        
        # (6) Graph loss
        # dir_loss_graph = mean(1 - cos(v_graph, dz_graph))
        v_graph_norm = v_graph.norm(dim=-1, keepdim=True).clamp_min(eps)  # [B, 1]
        dz_graph_norm = dz_graph.norm(dim=-1, keepdim=True).clamp_min(eps)  # [B, 1]
        v_graph_dir = v_graph / v_graph_norm  # [B, D]
        dz_graph_dir = dz_graph / dz_graph_norm  # [B, D]
        cos_graph = (v_graph_dir * dz_graph_dir).sum(dim=-1).mean()  # scalar
        dir_loss_graph = (1.0 - cos_graph)
        
        # mag_loss_graph = mean((log||v_graph|| - log||dz_graph||)^2)
        v_graph_norm_vals = v_graph.norm(dim=-1)  # [B]
        dz_graph_norm_vals = dz_graph.norm(dim=-1)  # [B]
        mag_loss_graph = (torch.log(v_graph_norm_vals.clamp_min(eps)) - torch.log(dz_graph_norm_vals.clamp_min(eps))).pow(2).mean()
        
        # (7) Residual loss
        # mag_loss_res = mean((log||v_res|| - log||dz_res||)^2)
        v_res_norm = v_res.norm(dim=-1).clamp_min(eps)  # [N]
        dz_res_norm = dz_res.norm(dim=-1).clamp_min(eps)  # [N]
        mag_loss_res = (torch.log(v_res_norm) - torch.log(dz_res_norm)).pow(2).mean()
        
        # Optional: residual variance loss
        # 对每个图分别算 Var(v_res) 和 Var(dz_res)（对节点维度），做 L2/MSE
        var_loss_res = torch.tensor(0.0, device=device, dtype=v_pred.dtype)
        if self.w_var_res > 0.0:
            # 计算每个图的 variance
            var_v_res_list = []
            var_dz_res_list = []
            for b in range(B):
                mask_b = (batch_idx == b)  # [N]
                if mask_b.sum() > 1:  # 至少需要2个节点才能计算variance
                    v_res_b = v_res[mask_b]  # [N_b, D]
                    dz_res_b = dz_res[mask_b]  # [N_b, D]
                    # 对节点维度求variance: Var(v_res_b) = mean((v_res_b - mean(v_res_b, dim=0))^2)
                    v_res_b_mean = v_res_b.mean(dim=0, keepdim=True)  # [1, D]
                    dz_res_b_mean = dz_res_b.mean(dim=0, keepdim=True)  # [1, D]
                    var_v_res_b = ((v_res_b - v_res_b_mean) ** 2).mean()  # scalar
                    var_dz_res_b = ((dz_res_b - dz_res_b_mean) ** 2).mean()  # scalar
                    var_v_res_list.append(var_v_res_b)
                    var_dz_res_list.append(var_dz_res_b)
            
            if len(var_v_res_list) > 0:
                var_v_res = torch.stack(var_v_res_list)  # [B_valid]
                var_dz_res = torch.stack(var_dz_res_list)  # [B_valid]
                var_loss_res = ((var_v_res - var_dz_res) ** 2).mean()  # MSE
        
        # (8) 总 flow_loss_total
        flow_loss_total = dir_loss_graph + self.w_mag_graph * mag_loss_graph + self.w_mag_res * mag_loss_res + self.w_var_res * var_loss_res
        
        # 计算 cos_res 用于日志（可选）
        v_res_norm_for_cos = v_res.norm(dim=-1).clamp_min(eps)  # [N]
        dz_res_norm_for_cos = dz_res.norm(dim=-1).clamp_min(eps)  # [N]
        v_res_dir = v_res / v_res_norm_for_cos.unsqueeze(-1)  # [N, D]
        dz_res_dir = dz_res / dz_res_norm_for_cos.unsqueeze(-1)  # [N, D]
        cos_res = (v_res_dir * dz_res_dir).sum(dim=-1).mean()  # scalar
        
        # 保留一些兼容性指标（用于日志）
        v_pred_norm = v_pred.norm(dim=-1)  # [N]
        v_target_norm = v_target.norm(dim=-1)  # [N]
        loss_norm = (
            torch.log(v_pred_norm + 1e-6)
            - torch.log(v_target_norm + 1e-6)
        ).pow(2).mean()
        
        # Final safety check
        if torch.isnan(flow_loss_total) or torch.isinf(flow_loss_total):
            # Fallback to a small positive value
            flow_loss_total = torch.tensor(1e-6, device=v_pred.device, dtype=v_pred.dtype)

        # Build delta_full (if not already built in cond_attn path)
        if not use_cond_attn:
            delta_z_hat_valid = v_pred
            # 把 ΔZ_hat 填回到 [L,B,D] 形状
            z0_flat = z0.reshape(-1, D)  # Recompute for legacy path
            delta_flat = torch.zeros_like(z0_flat)      # [B*L, D]
            delta_flat[valid_flat] = delta_z_hat_valid
            delta_full = delta_flat.view(B, L, D).permute(1, 0, 2)  # [L, B, D]
        
        # 9. 计算 sanity metrics（总是计算，用于 logging）
        sanity_metrics = {}
        eps = 1e-8
        
        # Add new loss components to sanity metrics for logging
        sanity_metrics["cos_graph"] = cos_graph.detach()  # graph-level cosine
        sanity_metrics["cos_res"] = cos_res.detach()  # residual cosine (optional)
        sanity_metrics["mag_loss_graph"] = mag_loss_graph.detach()
        sanity_metrics["mag_loss_res"] = mag_loss_res.detach()
        sanity_metrics["var_loss_res"] = var_loss_res.detach()
        sanity_metrics["flow_loss_total"] = flow_loss_total.detach()
        
        # Keep some compatibility metrics
        sanity_metrics["loss_norm"] = loss_norm.detach()  # 保留用于兼容性
        
        # (1) vt_norm2: mean(||v_target||^2)
        vt_norm2 = v_target.pow(2).sum(dim=-1).mean()
        sanity_metrics["vt_norm2"] = vt_norm2.detach()
        
        # (2) flow_loss_per_dim (using flow_loss_total as base)
        if self.flow_loss_reduce == "sum":
            flow_loss_per_dim = flow_loss_total / D
        else:  # "mean"
            flow_loss_per_dim = flow_loss_total  # 已经是 per-dim mean 了
        sanity_metrics["flow_loss_per_dim"] = flow_loss_per_dim.detach()
        
        # (3-5) Graph-level metrics: cos_mean_graph, norm_ratio_graph_mean/std
        # 需要计算 delta_z_true 的 graph-level pooling
        delta_z_true = tgt_enc - src_enc  # [L, B, D]
        dz_true_g = self._pool_graph(delta_z_true, src_mask)  # [B, D]
        dz_pred_g = self._pool_graph(delta_full, src_mask)    # [B, D]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(dz_true_g, dz_pred_g, dim=-1)  # [B]
        cos_mean_graph = cos_sim.mean()
        sanity_metrics["cos_mean_graph"] = cos_mean_graph.detach()
        
        # Norm ratio
        dz_true_norm = dz_true_g.norm(dim=-1)  # [B]
        dz_pred_norm = dz_pred_g.norm(dim=-1)  # [B]
        norm_ratio_graph = dz_pred_norm / (dz_true_norm + eps)  # [B]
        norm_ratio_graph_mean = norm_ratio_graph.mean()
        norm_ratio_graph_std = norm_ratio_graph.std()
        sanity_metrics["norm_ratio_graph_mean"] = norm_ratio_graph_mean.detach()
        sanity_metrics["norm_ratio_graph_std"] = norm_ratio_graph_std.detach()
        
        # debug_stats_from_flow was set in the cond_attn path above if applicable
        
        # Compute alignment metrics if requested
        metrics = None
        if return_metrics:
            # v_cos: mean cosine similarity between v_pred and v_target
            v_pred_norm = v_pred / (v_pred.norm(dim=-1, keepdim=True) + 1e-8)
            v_tgt_norm = v_target / (v_target.norm(dim=-1, keepdim=True) + 1e-8)
            v_cos = (v_pred_norm * v_tgt_norm).sum(dim=-1).mean()
            
            # v_pred_norm_mean/std
            v_pred_norm_vals = v_pred.norm(dim=-1)  # [N]
            v_pred_norm_mean = v_pred_norm_vals.mean()
            v_pred_norm_std = v_pred_norm_vals.std()
            
            # v_tgt_norm_mean/std
            v_tgt_norm_vals = v_target.norm(dim=-1)  # [N]
            v_tgt_norm_mean = v_tgt_norm_vals.mean()
            v_tgt_norm_std = v_tgt_norm_vals.std()
            
            # All metrics are detached to avoid affecting gradients
            metrics = {
                "v_cos": v_cos.detach(),
                "v_pred_norm_mean": v_pred_norm_mean.detach(),
                "v_pred_norm_std": v_pred_norm_std.detach(),
                "v_tgt_norm_mean": v_tgt_norm_mean.detach(),
                "v_tgt_norm_std": v_tgt_norm_std.detach(),
            }
            
            # Add debug stats if available
            if debug_stats_from_flow is not None:
                for k, v in debug_stats_from_flow.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = torch.tensor(float(v), device=v_pred.device)
                    else:
                        metrics[k] = v.detach() if isinstance(v, torch.Tensor) else v
            
            # Add gradient norms if debug_grad is enabled
            if self.debug_grad and self.training and (self._debug_step % self.debug_grad_every == 0):
                if hasattr(self, "_dbg_grad_attn"):
                    metrics["flow/dbg_grad_attn"] = self._dbg_grad_attn.clone()
                if hasattr(self, "_dbg_grad_fp"):
                    metrics["flow/dbg_grad_fp"] = self._dbg_grad_fp.clone()
        
        # Merge sanity_metrics into metrics if return_metrics=True, otherwise attach separately
        if return_metrics:
            if metrics is None:
                metrics = {}
            metrics.update(sanity_metrics)
        else:
            # Store sanity_metrics for later retrieval in forward()
            if not hasattr(self, '_last_flow_sanity_metrics'):
                self._last_flow_sanity_metrics = {}
            self._last_flow_sanity_metrics = sanity_metrics
        
        # Always return 3 values for consistency (metrics is None when return_metrics=False)
        # Return flow_loss_total instead of flow_loss (includes norm constraint)
        return flow_loss_total, delta_full, metrics

    # ---------- 主 forward ----------

    def forward(
        self,
        reactant,
        condition,
        product_latent,
        product_token,
    ) -> Dict[str, torch.Tensor]:
        # Clear auxiliary losses at the start of each forward pass
        self._aux_losses.clear()
        
        # Update debug step counter
        self._debug_step += 1

        # ... (数据组装部分不变) ...
        tensors: Dict[str, Any] = {}
        for k, v in reactant.data.items():
            tensors[k] = v
        if hasattr(condition, "data") and isinstance(condition.data, dict):
            for k, v in condition.data.items():
                tensors[k] = v
        
        cond_flat = self.build_condition_vector(tensors)  # [B, C] or None

        src_enc, tgt_enc, src_mask, tgt_mask = self._encode_src_tgt_nodes(tensors)
        delta_z_true = tgt_enc - src_enc
        
        # 调用 Flow (内部已处理广播)
        # If layer1_flow_only, request metrics for fast screening
        if self.layer1_flow_only:
            flow_loss, delta_z_hat_full, metrics = self._flow_forward(
                src_enc, tgt_enc, src_mask, cond_flat=cond_flat, return_metrics=True, tensors=tensors
            )
            # Add cond_drop_rate to metrics if available
            if metrics is None:
                metrics = {}
            if hasattr(self, "_last_cond_drop_rate"):
                device = flow_loss.device
                metrics["cond_drop_rate"] = torch.tensor(self._last_cond_drop_rate, device=device)
            # Layer-1 flow-only mode: only return flow_loss and metrics, skip decoder
            # Sanity check: ensure decoder is not executed
            assert self.layer1_flow_only, "This branch should only be executed when layer1_flow_only=True"
            
            # Extract sanity metrics from metrics
            result = {
                "loss": self.flow_weight * flow_loss,
                "flow_loss": flow_loss,
                **metrics,  # Add alignment metrics (includes sanity metrics if return_metrics=True)
            }
            
            # Ensure sanity metrics have "flow/" prefix for consistency
            if self.log_flow_sanity:
                sanity_keys = ["v_cos", "v_cos_graph", "dir_loss_node", "dir_loss_graph", "dir_loss", "mag_loss", "graph_norm_ratio", "vt_norm2", "flow_loss_per_dim", "cos_mean_graph", "norm_ratio_graph_mean", "norm_ratio_graph_std", "loss_norm", "flow_loss_raw",
                              "cos_node_total", "cos_graph", "cos_res_node", "cos_graph_node", "res_norm_mean", "graph_norm_mean", "flow_loss_total", "dir_loss_graph_decomp", "dir_loss_res"]
                for key in sanity_keys:
                    # If key exists without prefix, add it with prefix
                    if key in result and f"flow/{key}" not in result:
                        result[f"flow/{key}"] = result.pop(key)
                    # If key doesn't exist, try to get from _last_flow_sanity_metrics
                    elif f"flow/{key}" not in result:
                        if hasattr(self, '_last_flow_sanity_metrics') and key in self._last_flow_sanity_metrics:
                            result[f"flow/{key}"] = self._last_flow_sanity_metrics[key]
            
            return result
        
        # Normal mode: continue with decoder logic
        # Sanity check: ensure decoder is executed only when layer1_flow_only=False
        assert not self.layer1_flow_only, "Decoder should not be executed when layer1_flow_only=True"
        # Always request metrics for monitoring (even in normal mode)
        flow_loss, delta_z_hat_full, flow_metrics = self._flow_forward(
            src_enc, tgt_enc, src_mask, cond_flat=cond_flat, return_metrics=True, tensors=tensors
        )

        # ... (Decoder 部分保持不变) ...
        # Get ode_method from decoder_cfg
        ode_method = self.decoder_cfg.ode_method if hasattr(self.decoder_cfg, 'ode_method') else 'heun'
        src_for_decoder = self._build_decoder_embedding(
            src_enc=src_enc,
            tgt_enc=tgt_enc,
            src_mask=src_mask,
            delta_z_true=delta_z_true,
            delta_z_flow=delta_z_hat_full,
            cond_flat=cond_flat,
            nfe=self.nfe,
            ode_method=ode_method,
        )

        result = self.backbone.M_decoder(
            src_for_decoder,
            tensors["src_bond"],
            tensors["src_mask"],
            latent=None,
            tgt_bond=tensors["tgt_bond"],
            tgt_aroma=tensors["tgt_aroma"],
            tgt_charge=tensors["tgt_charge"],
            tgt_mask=tensors["tgt_mask"],
            action_embedding=None,
            temps=None,
            solvent_embedding=None,
            catalyst_embedding=None,
        )

        pred_loss = result["pred_loss"].mean()
        bond_loss = result["bond_loss"].mean()
        aroma_loss = result["aroma_loss"].mean()
        charge_loss = result["charge_loss"].mean()
        total_flow_loss = self.flow_weight * flow_loss
        total_loss = pred_loss + total_flow_loss
        
        # Add endpoint consistency loss if available
        if "end_loss" in self._aux_losses:
            end_loss = self._aux_losses["end_loss"]
            total_loss = total_loss + self.lambda_end * end_loss
        
        # Collect diagnostic info (detached, for monitoring only)
        diagnostics = {}
        if cond_flat is not None:
            diagnostics["cond_norm"] = cond_flat.norm(dim=-1).mean().detach()
        
        # Condition dropout rate
        if hasattr(self, "_last_cond_drop_rate"):
            device = cond_flat.device if cond_flat is not None else flow_loss.device
            diagnostics["cond_drop_rate"] = torch.tensor(self._last_cond_drop_rate, device=device)
        
        # Gated pooling diagnostics (A2)
        if hasattr(self, "_last_gate_mean") and self._last_gate_mean is not None:
            diagnostics["gate_pool_mean"] = self._last_gate_mean
        
        # FlowHead-specific diagnostics
        if self.use_conditional_flow and hasattr(self.flow_head, "_last_gamma"):
            # FiLMResidualFlowHead or FiLMHiddenFlowHead
            if self.flow_head._last_gamma is not None:
                diagnostics["film_gamma_abs"] = self.flow_head._last_gamma.abs().mean().detach()
            if self.flow_head._last_beta is not None:
                diagnostics["film_beta_abs"] = self.flow_head._last_beta.abs().mean().detach()
        elif self.use_conditional_flow and hasattr(self.flow_head, "_last_alpha"):
            # ControlFlowHead
            if self.flow_head._last_alpha is not None:
                diagnostics["control_alpha"] = self.flow_head._last_alpha.mean().detach()
        
        # Add flow metrics to diagnostics (if available)
        if flow_metrics is not None:
            diagnostics.update(flow_metrics)
        
        # Add endpoint consistency loss to diagnostics (if available)
        if "end_loss" in self._aux_losses:
            diagnostics["flow/end_loss"] = self._aux_losses["end_loss"].detach()
        
        # Add sanity metrics to result (if log_flow_sanity=True)
        if self.log_flow_sanity:
            # Extract sanity metrics from flow_metrics
            sanity_metrics_dict = {}
            if flow_metrics is not None:
                for key in ["v_cos", "v_cos_graph", "dir_loss_node", "dir_loss_graph", "dir_loss", "mag_loss", "graph_norm_ratio", "vt_norm2", "flow_loss_per_dim", "cos_mean_graph", "norm_ratio_graph_mean", "norm_ratio_graph_std", "loss_norm", "flow_loss_raw", 
                           "cos_node_total", "cos_graph", "cos_res_node", "cos_graph_node", "res_norm_mean", "graph_norm_mean", "flow_loss_total", 
                           "dir_loss_graph_decomp", "dir_loss_res"]:
                    if key in flow_metrics:
                        sanity_metrics_dict[f"flow/{key}"] = flow_metrics[key]
            diagnostics.update(sanity_metrics_dict)
        
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "bond_loss": bond_loss,
            "aroma_loss": aroma_loss,
            "charge_loss": charge_loss,
            "flow_loss": flow_loss,
            **diagnostics,  # Add diagnostic metrics (including flow_metrics and sanity_metrics)
        }

    def debug_step(self, batch, device: torch.device) -> Dict[str, float]:
        """
        Debug function for layer1_flow_only mode sanity check.
        运行一次 forward -> backward，打印 flow_loss/v_cos/norm_ratio，以及 backbone 任意参数 grad 是否为 None。
        
        Args:
            batch: 数据批次（需要包含 reactant, condition, product_latent, product_token）
            device: 设备
        
        Returns:
            Dict containing debug metrics
        """
        self.train()  # 确保在训练模式
        
        # 准备输入
        reactant = batch.reactant.to(device)
        condition = batch.condition.to(device)
        product_latent = batch.product_latent.to(device)
        product_token = batch.product_token.to(device)
        
        # Forward
        outputs = self(
            reactant=reactant,
            condition=condition,
            product_latent=product_latent,
            product_token=product_token,
        )
        
        loss = outputs["loss"]
        
        # Backward
        loss.backward()
        
        # 提取指标
        flow_loss = outputs["flow_loss"].item()
        v_cos = outputs.get("v_cos", torch.tensor(0.0))
        if isinstance(v_cos, torch.Tensor):
            v_cos = v_cos.item()
        v_pred_norm_mean = outputs.get("v_pred_norm_mean", torch.tensor(0.0))
        if isinstance(v_pred_norm_mean, torch.Tensor):
            v_pred_norm_mean = v_pred_norm_mean.item()
        v_tgt_norm_mean = outputs.get("v_tgt_norm_mean", torch.tensor(0.0))
        if isinstance(v_tgt_norm_mean, torch.Tensor):
            v_tgt_norm_mean = v_tgt_norm_mean.item()
        norm_ratio = v_pred_norm_mean / (v_tgt_norm_mean + 1e-8) if v_tgt_norm_mean > 0 else 0.0
        
        # 检查 backbone 参数的 grad
        backbone_has_grad = False
        backbone_param_name = None
        for name, param in self.backbone.named_parameters():
            if param.grad is not None:
                backbone_has_grad = True
                backbone_param_name = name
                break
        
        # 打印结果
        logging.info("=" * 60)
        logging.info("[Layer1 Debug Step] Forward + Backward Results:")
        logging.info(f"  flow_loss: {flow_loss:.6f}")
        logging.info(f"  v_cos: {v_cos:.6f}")
        logging.info(f"  norm_ratio: {norm_ratio:.6f}")
        logging.info(f"  v_pred_norm_mean: {v_pred_norm_mean:.6f}")
        logging.info(f"  v_tgt_norm_mean: {v_tgt_norm_mean:.6f}")
        if backbone_has_grad:
            logging.warning(f"  ⚠️  WARNING: Backbone parameter '{backbone_param_name}' has gradient (should be None)!")
        else:
            logging.info("  ✅ Backbone parameters correctly have no gradients")
        logging.info("=" * 60)
        
        # 清理梯度（避免影响后续训练）
        self.zero_grad()
        
        return {
            "flow_loss": flow_loss,
            "v_cos": v_cos,
            "norm_ratio": norm_ratio,
            "backbone_has_grad": backbone_has_grad,
        }
    
    @torch.no_grad()
    def debug_cond_effect(self, tensors: Dict[str, Any]) -> Dict[str, float]:
        """
        Debug method to compute the effect of condition on velocity.
        For the same batch, computes:
        - v_cond: velocity with normal cond_flat
        - v_zero: velocity with cond_flat = 0
        Returns:
            {
                "v_delta_norm_mean": mean(||v_cond - v_zero||)
            }
        """
        if not self.use_conditional_flow:
            return {"v_delta_norm_mean": 0.0}
        
        # Encode source and target
        src_enc, tgt_enc, src_mask, tgt_mask = self._encode_src_tgt_nodes(tensors)
        
        # Build normal condition vector
        cond_flat = self.build_condition_vector(tensors)  # [B, C] or None
        
        if cond_flat is None:
            return {"v_delta_norm_mean": 0.0}
        
        L, B, D = src_enc.shape
        device = src_enc.device
        
        # Prepare for _flow_forward: extract valid nodes and build cond_valid
        if self.detach_encoder_for_flow:
            src_fm = src_enc.detach()
            tgt_fm = tgt_enc.detach()
        else:
            src_fm = src_enc
            tgt_fm = tgt_enc
        
        z0 = src_fm.permute(1, 0, 2)  # [B, L, D]
        z1 = tgt_fm.permute(1, 0, 2)  # [B, L, D]
        valid_mask = (~src_mask).bool()  # [B, L]
        
        z0_flat = z0.reshape(-1, D)  # [B*L, D]
        z1_flat = z1.reshape(-1, D)  # [B*L, D]
        valid_flat = valid_mask.reshape(-1)  # [B*L]
        
        if valid_flat.sum() == 0:
            return {"v_delta_norm_mean": 0.0}
        
        z0_valid = z0_flat[valid_flat]  # [N, D]
        z1_valid = z1_flat[valid_flat]  # [N, D]
        N = z0_valid.size(0)
        
        # Build cond_valid for normal condition
        cond_expanded = cond_flat.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)  # [B*L, C]
        cond_valid = cond_expanded[valid_flat]  # [N, C]
        
        # Build cond_valid_zero (all zeros)
        cond_valid_zero = torch.zeros_like(cond_valid)  # [N, C]
        
        # Sample t (same for both)
        t = self._sample_t(N, device=device)  # [N]
        t_emb = self.time_embed(t)  # [N, T]
        t_ = t.unsqueeze(-1)  # [N, 1]
        zt = (1.0 - t_) * z0_valid + t_ * z1_valid + self.fm_sigma * torch.randn_like(z0_valid)  # [N, D]
        
        # Compute velocities
        v_cond, _ = self.flow_v(z_t=zt, t_emb=t_emb, cond_flat=cond_valid)  # [N, D]
        v_zero, _ = self.flow_v(z_t=zt, t_emb=t_emb, cond_flat=cond_valid_zero)  # [N, D]
        
        # Compute difference
        v_delta = v_cond - v_zero  # [N, D]
        v_delta_norm = v_delta.norm(dim=-1)  # [N]
        v_delta_norm_mean = v_delta_norm.mean().item()
        
        return {
            "v_delta_norm_mean": v_delta_norm_mean,
        }

    # ... (get_graph_delta 等辅助函数保持不变，调用时注意 cond_flat 传递即可) ...
    def get_graph_delta_true_pred_z0_zhat(self, tensors: Dict[str, Any]):
        """
        返回 graph-level 的 (delta_true, delta_pred, r_type)

        delta_true: [B, D]
        delta_pred: [B, D]
        r_type    : [B] 或 None
        """
        src_enc, tgt_enc, src_mask, tgt_mask = self._encode_src_tgt_nodes(tensors)
        # node-level Δz_true
        delta_z_true = tgt_enc - src_enc           # [L,B,D]

        cond_flat = self.build_condition_vector(tensors)
        # flow forward（和训练时一样）
        flow_loss, delta_z_hat_full, _ = self._flow_forward(src_enc, tgt_enc, src_mask, cond_flat=cond_flat, tensors=tensors)
        delta_z_pred = delta_z_hat_full            # [L,B,D]

        # pool 到 graph level
        delta_true_graph = self._pool_graph(delta_z_true, src_mask)  # [B,D]
        delta_pred_graph = self._pool_graph(delta_z_pred, src_mask)  # [B,D]

        # reaction type (如果有的话)
        r_type = tensors.get("r_type", None)
        if r_type is not None:
            r_type = r_type.detach().cpu().numpy()
        z0_graph = self._pool_graph(src_enc, src_mask)  # [B,D]
        zhat_graph = self._pool_graph(delta_z_pred + src_enc, src_mask)  # [B,D]
        return (
            delta_true_graph.detach().cpu().numpy(),
            delta_pred_graph.detach().cpu().numpy(),
            z0_graph.detach().cpu().numpy(),
            zhat_graph.detach().cpu().numpy(),
            r_type,
        )
    def _compute_velocity(self, z: torch.Tensor, t: torch.Tensor, src_mask: torch.Tensor, cond_flat: torch.Tensor = None) -> torch.Tensor:
        """
        计算速度场 v_theta(z, t)
        z: [L,B,D]
        t: [B]
        src_mask: [B,L] True=padding
        cond_flat: [B,C] 条件嵌入（可选）
        返回: [L,B,D]
        """
        L, B, D = z.shape
        valid = (~src_mask).bool()              # [B,L]
        valid_LBD = valid.t().unsqueeze(-1)     # [L,B,1]
        
        t_emb = self.time_embed(t)  # [B,T]
        t_flat = t_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        
        z_flat = z.permute(1, 0, 2).reshape(B * L, D)
        
        # 处理 cond_flat 的广播
        cond_expanded_flat = None
        if cond_flat is not None:
            # cond_flat [B,C] -> [B,1,C] -> [B,L,C] -> [B*L,C]
            cond_expanded_flat = cond_flat.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)
        
        v_flat, _ = self.flow_v(z_flat, t_flat, cond_flat=cond_expanded_flat, B=B, L=L)
        v = v_flat.view(B, L, D).permute(1, 0, 2) * valid_LBD
        
        return v

    def _velocity_fn_for_torchdiffeq(self, t: torch.Tensor, z_flat: torch.Tensor, src_mask: torch.Tensor, L: int, B: int, D: int, cond_flat: torch.Tensor = None) -> torch.Tensor:
        """
        为 torchdiffeq 准备的速度场函数
        t: 标量或形状为 [1] 的张量
        z_flat: [B*L, D] 展平后的 z
        src_mask: [B, L]
        cond_flat: [B,C] 条件嵌入（可选）
        返回: [B*L, D] 展平后的速度
        """
        # 将 t 转换为标量（torchdiffeq 传入的是标量）
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                t_val = t.item()
            else:
                t_val = t[0].item() if t.dim() > 0 else t.item()
        else:
            t_val = float(t)
        
        # 创建 [B] 形状的时间张量
        device = z_flat.device
        t_batch = torch.full((B,), t_val, device=device, dtype=z_flat.dtype)  # [B]
        
        # 将 z_flat 重塑为 [L, B, D]
        z = z_flat.view(B, L, D).permute(1, 0, 2)  # [L, B, D]
        
        # 计算速度
        v = self._compute_velocity(z, t_batch, src_mask, cond_flat=cond_flat)  # [L, B, D]
        
        # 展平回 [B*L, D]
        v_flat = v.permute(1, 0, 2).reshape(B * L, D)
        
        return v_flat

    def _integrate_ode_z1_torchdiffeq(
        self, 
        z0: torch.Tensor, 
        src_mask: torch.Tensor, 
        method: str = "dopri5",
        atol: float = 1e-4,
        rtol: float = 1e-4,
        options: Optional[Dict] = None,
        cond_flat: torch.Tensor = None
    ) -> torch.Tensor:
        """
        使用 torchdiffeq 求解 ODE: dz/dt = v_theta(z,t)
        返回 z1_hat_latent: [L,B,D]
        z0: [L,B,D]
        src_mask: [B,L] True=padding
        method: ODE求解器方法，如 "dopri5", "rk4", "euler" 等
        atol: 绝对容差
        rtol: 相对容差
        options: 其他选项（如 max_num_steps）
        cond_flat: [B,C] 条件嵌入（可选）
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is not available. Please install it with: pip install torchdiffeq")
        
        L, B, D = z0.shape
        device = z0.device
        
        # 将 z0 展平为 [B*L, D] 以符合 torchdiffeq 的接口
        z0_flat = z0.permute(1, 0, 2).reshape(B * L, D)
        
        # 创建速度场函数（使用 lambda 捕获必要的参数）
        def ode_func(t, z_flat):
            return self._velocity_fn_for_torchdiffeq(t, z_flat, src_mask, L, B, D, cond_flat=cond_flat)
        
        # 时间点：从 0 到 1
        t_span = torch.tensor([0.0, 1.0], device=device, dtype=z0.dtype)
        
        # 设置选项
        if options is None:
            options = {}
        
        # 使用 odeint_adjoint 求解（如果不需要梯度，也可以使用 odeint）
        try:
            # 尝试使用 odeint_adjoint（支持 adjoint 方法，内存更高效）
            traj = torchdiffeq.odeint_adjoint(
                ode_func,
                z0_flat,
                t_span,
                atol=atol,
                rtol=rtol,
                method=method,
                options=options,
                adjoint_params=()  # 推理时不需要梯度
            )
        except Exception:
            # 如果 odeint_adjoint 失败，回退到 odeint
            traj = torchdiffeq.odeint(
                ode_func,
                z0_flat,
                t_span,
                atol=atol,
                rtol=rtol,
                method=method,
                options=options
            )
        
        # traj 形状: [2, B*L, D]，取最后一个时间点
        z1_flat = traj[-1]  # [B*L, D]
        
        # 重塑回 [L, B, D]
        z1 = z1_flat.view(B, L, D).permute(1, 0, 2)  # [L, B, D]
        
        return z1

    def _integrate_ode_z1_rk4(self, z0: torch.Tensor, src_mask: torch.Tensor, n_steps: int = 20, cond_flat: torch.Tensor = None) -> torch.Tensor:
        """
        Runge-Kutta 4阶方法求解 ODE: dz/dt = v_theta(z,t)
        返回 z1_hat_latent: [L,B,D]
        z0: [L,B,D]
        src_mask: [B,L] True=padding
        n_steps: 积分步数
        cond_flat: [B,C] 条件嵌入（可选）
        """
        L, B, D = z0.shape
        device = z0.device
        z = z0.clone()
        
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device, dtype=z.dtype)
            
            # RK4 的四个阶段
            # k1 = v(z, t)
            k1 = self._compute_velocity(z, t, src_mask, cond_flat=cond_flat)
            
            # k2 = v(z + dt/2 * k1, t + dt/2)
            z2 = z + 0.5 * dt * k1
            t2 = t + 0.5 * dt
            k2 = self._compute_velocity(z2, t2, src_mask, cond_flat=cond_flat)
            
            # k3 = v(z + dt/2 * k2, t + dt/2)
            z3 = z + 0.5 * dt * k2
            k3 = self._compute_velocity(z3, t2, src_mask, cond_flat=cond_flat)
            
            # k4 = v(z + dt * k3, t + dt)
            z4 = z + dt * k3
            t4 = t + dt
            k4 = self._compute_velocity(z4, t4, src_mask, cond_flat=cond_flat)
            
            # RK4 更新公式: z_{n+1} = z_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        return z  # [L,B,D]
    def _pool_graph(self, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        encoder_output: [L, B, D]  node-level embedding
        mask:           [B, L]     True 表示 padding / 不参与
        返回: [B, D] 的 graph-level embedding（masked average）
        """
        # [B, L, D]
        x = encoder_output.permute(1, 0, 2)
        # valid: 1 表示有效节点，0 表示 padding
        valid = (~mask).unsqueeze(-1).float()   # [B, L, 1]
        x = x * valid
        denom = valid.sum(dim=1).clamp_min(1.0) # [B, 1]
        pooled = x.sum(dim=1) / denom           # [B, D]
        return pooled
    def _integrate_ode_z1(
        self, 
        z0: torch.Tensor, 
        src_mask: torch.Tensor, 
        n_steps: int = 20, 
        method: str = "heun",
        atol: float = 1e-4,
        rtol: float = 1e-4,
        options: Optional[Dict] = None,
        cond_flat: torch.Tensor = None
    ) -> torch.Tensor:
        """
        ODE integrate dz/dt = v_theta(z,t), return z1_hat_latent: [L,B,D]
        z0: [L,B,D]
        src_mask: [B,L] True=padding
        n_steps: 积分步数（仅用于固定步长方法）
        method: 求解器方法，可选:
            - "rk4": Runge-Kutta 4阶（固定步长）
            - "heun": 改进欧拉方法（固定步长）
            - "dopri5": Dormand-Prince 5阶（自适应步长，需要torchdiffeq）
            - "adams": Adams-Bashforth（自适应步长，需要torchdiffeq）
            - 其他 torchdiffeq 支持的方法
        atol: 绝对容差（仅用于自适应方法）
        rtol: 相对容差（仅用于自适应方法）
        options: 其他选项（如 max_num_steps，仅用于自适应方法）
        cond_flat: [B,C] 条件嵌入（可选）
        """
        # 检查是否是 torchdiffeq 方法
        torchdiffeq_methods = ["dopri5", "adams", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4_torchdiffeq"]
        
        if method in torchdiffeq_methods:
            if not TORCHDIFFEQ_AVAILABLE:
                raise ImportError(
                    f"Method '{method}' requires torchdiffeq. "
                    "Please install it with: pip install torchdiffeq, "
                    "or use 'rk4' or 'heun' instead."
                )
            return self._integrate_ode_z1_torchdiffeq(
                z0, src_mask, method=method, atol=atol, rtol=rtol, options=options, cond_flat=cond_flat
            )
        elif method == "rk4":
            return self._integrate_ode_z1_rk4(z0, src_mask, n_steps, cond_flat=cond_flat)
        elif method == "heun":
            # 保留原来的改进欧拉方法作为备选
            L, B, D = z0.shape
            device = z0.device
            z = z0.clone()

            valid = (~src_mask).bool()              # [B,L]
            valid_LBD = valid.t().unsqueeze(-1)     # [L,B,1]

            # 预先广播 Cond
            cond_expanded_flat = None
            if cond_flat is not None:
                # cond_flat [B,C] -> [B,1,C] -> [B,L,C] -> [B*L,C]
                cond_expanded_flat = cond_flat.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)

            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((B,), (i + 0.5) * dt, device=device)
                t_emb = self.time_embed(t)  # [B,T]
                t_flat = t_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)

                z_flat = z.permute(1, 0, 2).reshape(B * L, D)
                v0_flat, _ = self.flow_v(z_flat, t_flat, cond_flat=cond_expanded_flat, B=B, L=L)
                v0 = v0_flat.view(B, L, D).permute(1, 0, 2) * valid_LBD
                z_euler = z + dt * v0

                z_euler_flat = z_euler.permute(1, 0, 2).reshape(B * L, D)
                v1_flat, _ = self.flow_v(z_euler_flat, t_flat, cond_flat=cond_expanded_flat, B=B, L=L)
                v1 = v1_flat.view(B, L, D).permute(1, 0, 2) * valid_LBD

                z = z + 0.5 * dt * (v0 + v1)

            return z  # [L,B,D]
        else:
            raise ValueError(f"Unknown ODE method: {method}, expected 'rk4' or 'heun'")

    def _one_step_euler(
        self,
        z0: torch.Tensor,          
        src_mask: torch.Tensor,    
        cond_flat: torch.Tensor,   
        dt: float = 1.0,
    ) -> torch.Tensor:
        L, B, D = z0.shape
        device = z0.device

        valid = (~src_mask).bool()                       
        valid_flat = valid.reshape(B * L).unsqueeze(-1)  

        # 预先广播 Cond
        if cond_flat is not None:
             # cond_flat [B,C] -> [B,1,C] -> [B,L,C] -> [B*L,C]
            cond_expanded = cond_flat.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)
        else:
            cond_expanded = None

        t = torch.zeros((B,), device=device)
        t_emb = self.time_embed(t)                       
        t_flat = t_emb.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1) 

        z_flat = z0.permute(1, 0, 2).contiguous().view(B * L, D)         
        
        v_flat, _ = self.flow_v(z_t=z_flat, t_emb=t_flat, cond_flat=cond_expanded, B=B, L=L)                              
        v_flat = v_flat * valid_flat                                      

        z1_flat = z_flat + dt * v_flat                                    
        z1 = z1_flat.view(B, L, D).permute(1, 0, 2).contiguous()          
        return z1

    # ... (_build_decoder_embedding 保持不变) ...
    def _build_decoder_embedding(
        self,
        src_enc: torch.Tensor,          # [L,B,D]
        tgt_enc: torch.Tensor,          # [L,B,D]
        src_mask: torch.Tensor,         # [B,L] True=padding
        delta_z_true: torch.Tensor,     # [L,B,D]
        delta_z_flow: torch.Tensor,     # [L,B,D] (from _flow_forward)
        nfe: int = None,                 # ODE integration steps (if None, use self.nfe)
        ode_method: str = "heun",        # ODE solver method
        atol: float = 1e-4,             # Absolute tolerance (for adaptive methods)
        rtol: float = 1e-4,             # Relative tolerance (for adaptive methods)
        options: Optional[Dict] = None,  # Additional options for torchdiffeq
        cond_flat: torch.Tensor=None,         # [B,D] !!! 注意这里还是原始 [B,D]
    ) -> torch.Tensor:
        
        cfg = self.decoder_cfg
        delta_source = (cfg.delta_source or "tf").lower()
        input_mode = (cfg.input_mode or "fuse").lower()
        
        # Use self.nfe if nfe is not provided
        if nfe is None:
            nfe = self.nfe

        if delta_source == "tf":
            delta = delta_z_true
            z1 = tgt_enc
        elif delta_source == "flow":
            delta = delta_z_flow
            z1 = src_enc + delta
        elif delta_source == "ode":
            z1_latent = self._integrate_ode_z1(
                src_enc, src_mask, 
                n_steps=nfe, 
                method=ode_method,
                atol=atol,
                rtol=rtol,
                options=options,
                cond_flat=cond_flat
            )
            delta = z1_latent - src_enc
            z1 = z1_latent
            
            # Compute endpoint consistency loss in training mode
            if self.training and self.lambda_end > 0.0:
                # z1_latent: [L, B, D], tgt_enc: [L, B, D], src_mask: [B, L] (True=padding)
                # Convert to batch-major for easier masking
                z1_batch = z1_latent.permute(1, 0, 2)  # [B, L, D]
                tgt_batch = tgt_enc.permute(1, 0, 2)   # [B, L, D]
                valid = (~src_mask).unsqueeze(-1)      # [B, L, 1] (True=valid)
                
                # Compute MSE only on valid nodes
                diff = (z1_batch - tgt_batch) * valid.float()  # [B, L, D]
                end_loss = (diff ** 2).sum() / valid.sum().clamp_min(1.0)
                self._aux_losses["end_loss"] = end_loss
        elif delta_source == "euler":
            z1 = self._one_step_euler(src_enc, src_mask, cond_flat=cond_flat)
            delta = z1 - src_enc
            
            # Compute endpoint consistency loss in training mode
            if self.training and self.lambda_end > 0.0:
                # z1: [L, B, D], tgt_enc: [L, B, D], src_mask: [B, L] (True=padding)
                # Convert to batch-major for easier masking
                z1_batch = z1.permute(1, 0, 2)  # [B, L, D]
                tgt_batch = tgt_enc.permute(1, 0, 2)   # [B, L, D]
                valid = (~src_mask).unsqueeze(-1)      # [B, L, 1] (True=valid)
                
                # Compute MSE only on valid nodes
                diff = (z1_batch - tgt_batch) * valid.float()  # [B, L, D]
                end_loss = (diff ** 2).sum() / valid.sum().clamp_min(1.0)
                self._aux_losses["end_loss"] = end_loss
        elif delta_source == "enc":
            return src_enc
        else:
            raise ValueError(f"Unknown decoder_cfg.delta_source={cfg.delta_source}")

        if input_mode == "fuse":
            L, B, D = src_enc.shape
            z0_flat = src_enc.reshape(L * B, D)
            d_flat  = delta.reshape(L * B, D)
            fused = self.delta_fuser(torch.cat([z0_flat, d_flat], dim=-1)).view(L, B, D)
            return fused
        elif input_mode == "add":
            return src_enc + delta
        elif input_mode == "delta":
            return delta
        elif input_mode == "z1":
            return z1
        else:
            raise ValueError(f"Unknown decoder_cfg.input_mode={cfg.input_mode}")