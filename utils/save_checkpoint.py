from pathlib import Path
import torch
import datetime

def save_checkpoint(model, optimizer, cfg, num_epochs, save_dir=None):
    if save_dir is None:
        save_dir = cfg["train"]["save_dir"]
    
    # 直接使用save_dir，不创建子目录
    ckpt_dir = Path(save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名包含epoch信息，方便区分
    epoch = num_epochs - 1
    ckpt_filename = f"flow_nerf_baseline_epoch_{epoch}.pt"
    ckpt_path = ckpt_dir / ckpt_filename
    
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
        },
        ckpt_path,
    )

    # torch.save(
    #     model.backbone.M_encoder.state_dict(),
    #     ckpt_dir / "M_encoder_baseline.pt",
    # )
    print(f"[Checkpoint] Saved whole model to {ckpt_path}")
    # print(f"[Checkpoint] Saved M_encoder to checkpoints/M_encoder_baseline_{num_epochs - 1}.pt")