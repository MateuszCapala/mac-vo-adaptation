import argparse
from pathlib import Path
import re
import numpy as np
import torch

from DataLoader import SequenceBase, StereoFrame
from Utility.Config import load_config
from Utility.PrettyPrint import Logger, ColoredTqdm
from Module.Network.FlowFormer.configs.submission import get_cfg
from Module.Network.FlowFormerCov import build_flowformer
from Utility.Utils import reflect_torch_dtype


def _infer_index_width(file_paths: list[Path]) -> int:
    if not file_paths:
        return 0
    match = re.search(r"(\d+)", file_paths[0].stem)
    return len(match.group(1)) if match else 0


def _format_index(idx: int, width: int) -> str:
    return f"{idx:0{width}d}" if width > 0 else str(idx)


def _save_flow(path: Path, flow: torch.Tensor) -> None:
    # flow: 2xHxW
    flow_np = flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    np.save(path, flow_np)


def _save_mask(path: Path, mask: torch.Tensor) -> None:
    # mask: 1xHxW (bool)
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    np.save(path, mask_np)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-GT optical flow for RoverStereo.")
    parser.add_argument("--data", type=str, required=True, help="Path to RoverStereo sequence config YAML")
    parser.add_argument("--weight", type=str, default="Model/MACVO_FrontendCov.pth", help="FlowFormerCov checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for flow inference")
    parser.add_argument("--max-flow", type=float, default=400.0, help="Max flow magnitude for valid mask")
    parser.add_argument("--flow-dir", type=str, required=True, help="Output directory for flow .npy files")
    parser.add_argument("--mask-dir", type=str, required=True, help="Output directory for mask .npy files")
    parser.add_argument("--flow-prefix", type=str, default="flow_", help="Filename prefix for flow files")
    parser.add_argument("--mask-prefix", type=str, default="mask_", help="Filename prefix for mask files")
    parser.add_argument("--enc-dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"], help="Encoder dtype")
    parser.add_argument("--dec-dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"], help="Decoder dtype")
    parser.add_argument("--decoder-depth", type=int, default=12, help="FlowFormer decoder depth")
    parser.add_argument("--start", type=int, default=0, help="Start index in sequence")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive), -1 for full")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--ignore-existing-flow",
        action="store_true",
        help="Disable loading gt_flow/flow_mask from config while generating pseudo-flow",
    )
    args = parser.parse_args()

    cfg, _ = load_config(Path(args.data))
    datacfg = cfg
    if args.ignore_existing_flow and hasattr(datacfg, "args"):
        # Avoid warnings when generating flow into directories not yet populated.
        datacfg.args.flow_dir = None
        datacfg.args.flow_mask_dir = None
    sequence = SequenceBase[StereoFrame].instantiate(datacfg.type, datacfg.args)

    if args.end == -1:
        args.end = len(sequence) - 1
    sequence = sequence.clip(args.start, args.end + 1)

    flow_dir = Path(args.flow_dir)
    mask_dir = Path(args.mask_dir)
    flow_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    index_width = _infer_index_width(sequence.left.file_names)

    cfg_ff = get_cfg()
    cfg_ff.latentcostformer.decoder_depth = args.decoder_depth
    model = build_flowformer(
        cfg_ff,
        reflect_torch_dtype(args.enc_dtype),
        reflect_torch_dtype(args.dec_dtype)
    )
    ckpt = torch.load(args.weight, map_location=args.device, weights_only=True)
    model.load_ddp_state_dict(ckpt)
    model.to(args.device)
    model.eval()

    total_pairs = len(sequence) - 1
    batch = args.batch

    with torch.inference_mode():
        for start in ColoredTqdm(range(0, total_pairs, batch), desc="Pseudo flow"):
            end = min(start + batch, total_pairs)
            batch_frames = [sequence[i] for i in range(start, end + 1)]
            img0_list = [f.stereo.imageL for f in batch_frames[:-1]]
            img1_list = [f.stereo.imageL for f in batch_frames[1:]]
            img0 = torch.cat(img0_list, dim=0).to(args.device)
            img1 = torch.cat(img1_list, dim=0).to(args.device)

            flow, _ = model.inference(img0, img1)
            flow = flow.float()
            mask = (flow.norm(dim=1, keepdim=True) < args.max_flow)

            for local_i, frame in enumerate(batch_frames[:-1]):
                idx = frame.frame_idx
                idx_str = _format_index(idx, index_width)
                flow_path = flow_dir / f"{args.flow_prefix}{idx_str}.npy"
                mask_path = mask_dir / f"{args.mask_prefix}{idx_str}.npy"

                if not args.overwrite and (flow_path.exists() or mask_path.exists()):
                    Logger.write("info", f"Skip existing flow/mask for idx {idx}")
                    continue

                _save_flow(flow_path, flow[local_i])
                _save_mask(mask_path, mask[local_i])


if __name__ == "__main__":
    main()
