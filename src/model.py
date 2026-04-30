import torch
import torch.nn as nn
import timm


def build_model(cfg, sit_weights_path=None, use_timm_pretrained=None):
    if use_timm_pretrained is None:
        use_timm_pretrained = sit_weights_path is None

    model = timm.create_model(
        cfg["model"]["name"],          # vit_small_patch16_224
        pretrained=use_timm_pretrained,
        num_classes=0,                 # remove default head; we add our own
        drop_rate=cfg["model"]["drop_rate"],
        img_size=cfg["model"]["img_size"],
    )

    if sit_weights_path is not None:
        _load_sit_weights(model, sit_weights_path)
        print(f"Loaded SiT-S weights from {sit_weights_path}")
    elif use_timm_pretrained:
        print("Using timm ImageNet pretrained ViT-Small weights")
    else:
        print("Using randomly initialised ViT-Small weights")

    embed_dim = model.embed_dim
    model.head = nn.Linear(embed_dim, cfg["model"]["num_classes"])
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    return model


def _load_sit_weights(model, weights_path):
    """Load SiT pretrained checkpoint into a timm ViT-Small model."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # SiT checkpoints may be wrapped under different keys depending on training code.
    if isinstance(ckpt, dict):
        if "student" in ckpt and isinstance(ckpt["student"], dict):
            state = ckpt["student"]
        elif "teacher" in ckpt and isinstance(ckpt["teacher"], dict):
            state = ckpt["teacher"]
        else:
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    # Strip any 'module.' prefix from DataParallel wrappers
    state = {k.replace("module.", ""): v for k, v in state.items()}
    state = {
        (k.replace("backbone.", "", 1) if k.startswith("backbone.") else k): v
        for k, v in state.items()
    }

    # Remove the original classification head (we add a new one)
    state = {k: v for k, v in state.items() if not k.startswith("head")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")


def get_optimizer(model, cfg):
    """AdamW with differential LRs: lower for backbone, higher for head."""
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params     = [p for n, p in model.named_parameters() if "head"     in n]

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["training"]["backbone_lr"]},
        {"params": head_params,     "lr": cfg["training"]["head_lr"]},
    ], weight_decay=cfg["training"]["weight_decay"])
