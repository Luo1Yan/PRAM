"""Extract frozen LanguageBind global and twelfth-block local features.

Run python scripts/extract_languagebind.py --manifest relation.json --output pack.npz
with the official languagebind package on the Python path. Use the stream and
split manifest documented in extract_imagebind.py. Inputs are the checkpoint
processor's pixel_values or text input_ids in NPY format. Video uses [N,C,T,H,W]
and audio uses [N,3,frequency,time]. Text uses [N,L].

Set each stream's layout to spatial, temporal or none. Spatial pooling yields
8 by 8 slots. Temporal pooling averages four consecutive segments. Video can
supply either layout. Use none for the single-vector side of a mixed-grid pair.
The optional checkpoint and revision stream fields select a specific tower.
For text, tower selects the checkpoint's modality family and defaults to image.
Upstream encoders: https://github.com/PKU-YuanGroup/LanguageBind
"""

from pathlib import Path

import torch
from torch.nn.functional import normalize

if __package__:
    from .extract_imagebind import extraction_cli, spatial_pool, temporal_pool
else:
    from extract_imagebind import extraction_cli, spatial_pool, temporal_pool


_CHECKPOINTS = {
    "image": "LanguageBind/LanguageBind_Image",
    "video": "LanguageBind/LanguageBind_Video_FT",
    "audio": "LanguageBind/LanguageBind_Audio_FT",
    "thermal": "LanguageBind/LanguageBind_Thermal",
    "depth": "LanguageBind/LanguageBind_Depth",
}


class LanguageBindEncoder:
    def __init__(self, device, options, root):
        from languagebind import model_dict

        self.models = {}
        self.classes = model_dict
        self.device, self.options, self.root = device, options, root

    def __call__(self, spec, value):
        key = spec["modality"].lower()
        key = {"rgb": "image", "vision": "image", "language": "text"}.get(key, key)
        tower = spec.get("tower", "image") if key == "text" else key
        checkpoint = spec.get("checkpoint", _CHECKPOINTS[tower])
        local = self.root / Path(checkpoint)
        if local.is_dir():
            checkpoint = str(local)
        revision = spec.get("revision", self.options.get("revision", "main"))
        identity = (tower, checkpoint, revision)
        if identity not in self.models:
            model = self.classes[tower].from_pretrained(checkpoint, revision=revision)
            self.models[identity] = model.to(self.device).eval().requires_grad_(False)
        model = self.models[identity]
        if key == "text":
            output = model.text_model(input_ids=value.long(), return_dict=True)
            return normalize(model.text_projection(output.pooler_output), dim=-1), None
        layout = spec.get("layout", "temporal" if key in ("audio", "video") else "spatial")
        output = model.vision_model(
            pixel_values=value.float(), output_hidden_states=layout != "none", return_dict=True,
        )
        global_feat = normalize(model.visual_projection(output.pooler_output), dim=-1)
        if layout == "none":
            return global_feat, None
        tokens = output.hidden_states[12][:, 1:]
        patch = model.config.vision_config.patch_size
        ph, pw = (patch, patch) if isinstance(patch, int) else patch[-2:]
        h, w = value.size(-2) // ph, value.size(-1) // pw
        if tokens.size(1) != h * w or tokens.size(0) % len(value):
            raise ValueError("Block-12 tokens disagree with the processor's patch grid")
        grid = tokens.reshape(len(value), -1, h, w, tokens.size(-1))
        if layout == "spatial":
            return global_feat, spatial_pool(grid.mean(1), 8)
        if layout == "temporal":
            time = grid.mean(2).flatten(1, 2) if key == "audio" else grid.mean((2, 3))
            return global_feat, temporal_pool(time, 4)
        raise ValueError(f"Unknown LanguageBind layout {layout}")


if __name__ == "__main__":
    extraction_cli(__doc__, LanguageBindEncoder)
