import csv
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(image_tensor):
    image = image_tensor.detach().cpu()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0.0, 1.0)
    return image


def tensor_to_pil(image_tensor):
    image = denormalize_image(image_tensor)
    image_np = image.permute(1, 2, 0).numpy()
    return Image.fromarray((image_np * 255).astype(np.uint8))


class ViTAttentionExtractor:
    def __init__(self, model):
        if not hasattr(model, "blocks"):
            raise ValueError("Expected a ViT-style model with a `blocks` attribute.")

        self.model = model
        self.block_inputs = []
        self.handles = []

    def _hook_block(self, module, inputs, output):
        self.block_inputs.append(inputs[0].detach())

    def __enter__(self):
        self.block_inputs = []
        self.handles = [
            block.attn.register_forward_hook(self._hook_block)
            for block in self.model.blocks
        ]
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _compute_attention(self, attn_module, tokens):
        batch_size, num_tokens, embed_dim = tokens.shape
        head_dim = embed_dim // attn_module.num_heads

        qkv = attn_module.qkv(tokens)
        qkv = qkv.reshape(batch_size, num_tokens, 3, attn_module.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)

        if hasattr(attn_module, "q_norm"):
            q = attn_module.q_norm(q)
        if hasattr(attn_module, "k_norm"):
            k = attn_module.k_norm(k)

        attn = (q * attn_module.scale) @ k.transpose(-2, -1)
        return attn.softmax(dim=-1)

    def get_attention_maps(self):
        attention_maps = []
        for block, tokens in zip(self.model.blocks, self.block_inputs):
            attention_maps.append(self._compute_attention(block.attn, tokens))
        return attention_maps


def compute_attention_rollout(attention_maps):
    if not attention_maps:
        raise ValueError("No attention maps were captured during the forward pass.")

    batch_size = attention_maps[0].shape[0]
    num_tokens = attention_maps[0].shape[-1]
    joint = torch.eye(num_tokens, device=attention_maps[0].device).unsqueeze(0)
    joint = joint.repeat(batch_size, 1, 1)

    for attn in attention_maps:
        attn_heads = attn.mean(dim=1)
        attn_heads = attn_heads + torch.eye(num_tokens, device=attn.device).unsqueeze(0)
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        joint = attn_heads @ joint

    cls_to_patches = joint[:, 0, 1:]
    side = int(cls_to_patches.shape[-1] ** 0.5)
    rollout = cls_to_patches.reshape(batch_size, side, side)
    rollout = rollout / rollout.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return rollout


def compute_last_layer_attention(attention_maps):
    if not attention_maps:
        raise ValueError("No attention maps were captured during the forward pass.")

    last_attn = attention_maps[-1].mean(dim=1)
    cls_to_patches = last_attn[:, 0, 1:]
    side = int(cls_to_patches.shape[-1] ** 0.5)
    attn_map = cls_to_patches.reshape(last_attn.shape[0], side, side)
    attn_map = attn_map / attn_map.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return attn_map


def upscale_attention_map(attn_map, image_size):
    attn_map = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(
        attn_map,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return attn_map.squeeze(0).squeeze(0)


def blend_attention_overlay(image, attn_map, alpha=0.4):
    heatmap = cm.jet(attn_map)[:, :, :3]
    image_np = np.asarray(image).astype(np.float32) / 255.0
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def save_attention_visuals(image_tensor, attn_map, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_pil(image_tensor)
    attn_up = upscale_attention_map(attn_map.detach().cpu(), image.size[::-1]).numpy()
    heatmap = cm.jet(attn_up)[:, :, :3]
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    overlay = blend_attention_overlay(image, attn_up)

    image.save(output_prefix.parent / f"{output_prefix.name}_image.png")
    heatmap_img.save(output_prefix.parent / f"{output_prefix.name}_attention.png")
    overlay.save(output_prefix.parent / f"{output_prefix.name}_overlay.png")


def summarize_attention_run(records, output_dir):
    output_dir = Path(output_dir)

    with open(output_dir / "attention_metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(output_dir / "attention_metadata.csv", "w", newline="", encoding="utf-8") as f:
        if not records:
            return
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_attention_grid(image_paths, title, output_path, cols=3):
    if not image_paths:
        return

    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, image_path in zip(axes.flat, image_paths):
        ax.imshow(Image.open(image_path))
        ax.set_title(Path(image_path).stem, fontsize=9)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
