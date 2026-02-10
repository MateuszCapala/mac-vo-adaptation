import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from datetime import datetime
from torch.utils.data import Dataset

from Utility.PrettyPrint import Logger

from ..Interface import StereoFrame, StereoData
from ..SequenceBase import SequenceBase


class RoverStereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "RoverStereo"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.root = Path(cfg.root)

        self.left = RoverMonocularDataset(self.root, cfg.left_prefix, cfg.format)
        self.right = RoverMonocularDataset(self.root, cfg.right_prefix, cfg.format)

        src_height, src_width = self.left.image_shape
        if hasattr(cfg, "resize"):
            self.target_height = getattr(cfg.resize, "height", src_height)
            self.target_width = getattr(cfg.resize, "width", src_width)
        else:
            self.target_height = src_height
            self.target_width = src_width
        self._needs_resize = (self.target_height != src_height) or (self.target_width != src_width)

        common_indices = sorted(set(self.left.indices).intersection(self.right.indices))
        assert len(common_indices) > 0, "No matching left/right image indices found."

        self.frame_indices = np.array(common_indices, dtype=np.int64)
        self.timestamps = self._load_timestamps(Path(cfg.times))

        self.gt_pose_data: dict[int, pp.LieTensor] | None = None
        if getattr(cfg, "gt_pose", False):
            gps_flip_xy = getattr(cfg, "gps_flip_xy", False)
            gps_swap_xy = getattr(cfg, "gps_swap_xy", False)
            self.gt_pose_data = self._load_gps_topocentric(Path(cfg.gps_topo), gps_flip_xy, gps_swap_xy)

        self.baseline = float(cfg.bl)
        self.T_BS = pp.identity_SE3(1, dtype=torch.float32)
        scale_u = self.target_width / src_width
        scale_v = self.target_height / src_height
        self.K = torch.tensor([
            [cfg.camera.fx * scale_u, 0.0, cfg.camera.cx * scale_u],
            [0.0, cfg.camera.fy * scale_v, cfg.camera.cy * scale_v],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32).unsqueeze(0)

        super().__init__(len(self.frame_indices))
        self.indices = self.frame_indices

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = int(self.get_index(local_index))
        imageL = self.left[index]
        imageR = self.right[index]

        if self._needs_resize:
            imageL = torch.nn.functional.interpolate(
                imageL, size=(self.target_height, self.target_width), mode="bilinear", align_corners=False
            )
            imageR = torch.nn.functional.interpolate(
                imageR, size=(self.target_height, self.target_width), mode="bilinear", align_corners=False
            )

        time_ns = self.timestamps.get(index)
        if time_ns is None:
            Logger.write("warn", f"Missing timestamp for index {index}, using fallback.")
            time_ns = index * 1000

        gt_pose = None
        if self.gt_pose_data is not None:
            gt_pose = self.gt_pose_data.get(index)
            if gt_pose is None:
                Logger.write("warn", f"Missing GPS GT for index {index}.")

        return StereoFrame(
            idx=[local_index],
            time_ns=[time_ns],
            stereo=StereoData(
                T_BS=self.T_BS,
                K=self.K,
                baseline=torch.tensor([self.baseline]),
                width=imageL.size(-1),
                height=imageL.size(-2),
                time_ns=[time_ns],
                imageL=imageL,
                imageR=imageR,
            ),
            gt_pose=gt_pose,
        )

    @staticmethod
    def _load_timestamps(path: Path) -> dict[int, int]:
        assert path.exists(), f"Timestamp file not found: {path}"
        timestamps: dict[int, int] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                idx = int(parts[0])
                dt = datetime.fromisoformat(parts[1])
                timestamps[idx] = int(dt.timestamp() * 1_000_000_000)
        return timestamps

    @staticmethod
    def _load_gps_topocentric(path: Path, flip_xy: bool, swap_xy: bool) -> dict[int, pp.LieTensor]:
        assert path.exists(), f"GPS topocentric file not found: {path}"
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        gt_map: dict[int, pp.LieTensor] = {}
        for row in data:
            idx = int(row[0])
            east = float(row[1])
            north = float(row[2])
            up = float(row[3])

            # Convert ENU -> NED, with optional axis swap/flip for alignment
            tx = north
            ty = east
            tz = -up
            if swap_xy:
                tx, ty = ty, tx
            if flip_xy:
                tx, ty = -tx, -ty
            pose = pp.SE3(torch.tensor([[tx, ty, tz, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
            gt_map[idx] = pose

        return gt_map

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root"        : lambda s: isinstance(s, str),
            "times"       : lambda s: isinstance(s, str),
            "left_prefix" : lambda s: isinstance(s, str),
            "right_prefix": lambda s: isinstance(s, str),
            "format"      : lambda s: isinstance(s, str),
            "bl"          : lambda v: isinstance(v, float),
            "camera"      : lambda v: isinstance(v, dict) and cls._enforce_config_spec(v, {
                "fx": lambda v: isinstance(v, float),
                "fy": lambda v: isinstance(v, float),
                "cx": lambda v: isinstance(v, float),
                "cy": lambda v: isinstance(v, float),
            }, allow_excessive_cfg=True),
            "gt_pose"     : lambda b: isinstance(b, bool),
            "gps_topo"    : lambda s: isinstance(s, str),
            "gps_flip_xy" : lambda b: isinstance(b, bool),
            "gps_swap_xy" : lambda b: isinstance(b, bool),
        }, allow_excessive_cfg=True)

        if hasattr(config, "resize"):
            cls._enforce_config_spec(config.resize, {
                "height": lambda v: isinstance(v, int) and v > 0,
                "width" : lambda v: isinstance(v, int) and v > 0,
            })


class RoverMonocularDataset(Dataset):
    def __init__(self, directory: Path, prefix: str, format: str) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Image directory does not exist: {self.directory}"

        self.file_names = sorted(self.directory.glob(f"{prefix}*.{format}"))
        assert len(self.file_names) > 0, f"No files found with prefix {prefix} and format {format}"

        self._index_map: dict[int, Path] = {}
        for path in self.file_names:
            idx = self._parse_index(path.name, prefix)
            self._index_map[idx] = path

        self.indices = sorted(self._index_map.keys())

        sample = cv2.imread(str(self.file_names[0]), cv2.IMREAD_COLOR)
        assert sample is not None, f"Failed to read image: {self.file_names[0]}"
        self.image_shape = (sample.shape[0], sample.shape[1])

    @staticmethod
    def _parse_index(name: str, prefix: str) -> int:
        suffix = name[len(prefix):]
        number = "".join(ch for ch in suffix if ch.isdigit())
        return int(number)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self._index_map[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, f"Failed to read image: {path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image_tensor /= 255.0
        return image_tensor
