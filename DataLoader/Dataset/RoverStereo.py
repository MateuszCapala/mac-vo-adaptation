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
            init_rot = None
            if getattr(cfg, "gps_use_init_orient", False) and hasattr(cfg, "known_orientations"):
                init_rot = self._load_initial_rotation_matrix(Path(cfg.known_orientations))
            gt_rotations = self._build_gt_rotation_list(cfg)
            if hasattr(cfg, "gps_latlon_sampled"):
                self.gt_pose_data = self._load_gps_latlon_sampled(
                    Path(cfg.gps_latlon_sampled), gps_flip_xy, gps_swap_xy, init_rot, gt_rotations
                )
            else:
                self.gt_pose_data = self._load_gps_topocentric(
                    Path(cfg.gps_topo), gps_flip_xy, gps_swap_xy, init_rot, gt_rotations
                )

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
    def _load_gps_topocentric(
        path: Path,
        flip_xy: bool,
        swap_xy: bool,
        init_rot: torch.Tensor | None,
        gt_rotations: list[dict[str, float | str]],
    ) -> dict[int, pp.LieTensor]:
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

            enu = torch.tensor([east, north, up], dtype=torch.float32)
            if init_rot is not None:
                enu = init_rot @ enu
                east, north, up = enu.tolist()

            # Convert ENU -> NED, with optional axis swap/flip for alignment
            tx = north
            ty = east
            tz = -up
            if swap_xy:
                tx, ty = ty, tx
            if flip_xy:
                tx, ty = -tx, -ty

            tx, ty, tz = RoverStereoSequence._apply_gt_rotations(tx, ty, tz, gt_rotations)
            pose = pp.SE3(torch.tensor([[tx, ty, tz, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
            gt_map[idx] = pose

        return gt_map

    @staticmethod
    def _load_gps_latlon_sampled(
        path: Path,
        flip_xy: bool,
        swap_xy: bool,
        init_rot: torch.Tensor | None,
        gt_rotations: list[dict[str, float | str]],
    ) -> dict[int, pp.LieTensor]:
        assert path.exists(), f"GPS lat/lon sampled file not found: {path}"
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # columns: index, lat[deg], lon[deg], elev[m]
        data = data[np.argsort(data[:, 0])]
        idx0, lat0, lon0, h0 = data[0]
        x0, y0, z0 = RoverStereoSequence._geodetic_to_ecef(lat0, lon0, h0)

        gt_map: dict[int, pp.LieTensor] = {}
        for row in data:
            idx = int(row[0])
            lat, lon, h = float(row[1]), float(row[2]), float(row[3])
            x, y, z = RoverStereoSequence._geodetic_to_ecef(lat, lon, h)
            e, n, u = RoverStereoSequence._ecef_to_enu(x, y, z, lat0, lon0, x0, y0, z0)

            enu = torch.tensor([e, n, u], dtype=torch.float32)
            if init_rot is not None:
                enu = init_rot @ enu
                e, n, u = enu.tolist()

            # Convert ENU -> NED, with optional axis swap/flip for alignment
            tx = n
            ty = e
            tz = -u
            if swap_xy:
                tx, ty = ty, tx
            if flip_xy:
                tx, ty = -tx, -ty

            tx, ty, tz = RoverStereoSequence._apply_gt_rotations(tx, ty, tz, gt_rotations)

            pose = pp.SE3(torch.tensor([[tx, ty, tz, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
            gt_map[idx] = pose

        return gt_map

    @staticmethod
    def _apply_gt_rotations(
        tx: float,
        ty: float,
        tz: float,
        rotations: list[dict[str, float | str]],
    ) -> tuple[float, float, float]:
        v = torch.tensor([tx, ty, tz], dtype=torch.float32)
        for rot in rotations:
            if isinstance(rot, dict):
                axis = str(rot["axis"]).lower()
                angle_deg = float(rot["angle_deg"])
            else:
                axis = str(getattr(rot, "axis")).lower()
                angle_deg = float(getattr(rot, "angle_deg"))
            angle = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
            c = torch.cos(angle)
            s = torch.sin(angle)

            if axis == "x":
                R = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)
            elif axis == "y":
                R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
            elif axis == "z":
                R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported axis '{axis}' in gt_rotations")

            v = R @ v

        return float(v[0]), float(v[1]), float(v[2])

    @staticmethod
    def _build_gt_rotation_list(cfg: SimpleNamespace) -> list[dict[str, float | str]]:
        if hasattr(cfg, "gt_rotations"):
            return list(cfg.gt_rotations)

        rotations: list[dict[str, float | str]] = []
        rot_y = getattr(cfg, "gt_yz_rot90", "none")
        rot_z = getattr(cfg, "gt_zz_rot90", "none")
        if rot_y == "cw":
            rotations.append({"axis": "y", "angle_deg": -90.0})
        elif rot_y == "ccw":
            rotations.append({"axis": "y", "angle_deg": 90.0})

        if rot_z == "cw":
            rotations.append({"axis": "z", "angle_deg": -90.0})
        elif rot_z == "ccw":
            rotations.append({"axis": "z", "angle_deg": 90.0})

        return rotations

    @staticmethod
    def _load_initial_rotation_matrix(path: Path) -> torch.Tensor | None:
        assert path.exists(), f"Known orientations file not found: {path}"
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        rows = data[data[:, 0].astype(int) == 1]
        if rows.size == 0:
            return None

        ax, ay, az, ang = rows[0, 1], rows[0, 2], rows[0, 3], rows[0, 4]
        axis = torch.tensor([ax, ay, az], dtype=torch.float32)
        axis = axis / (axis.norm() + 1e-12)
        half = float(ang) * 0.5
        sin_half = torch.sin(torch.tensor(half, dtype=torch.float32))
        quat = torch.cat([axis * sin_half, torch.tensor([torch.cos(torch.tensor(half))])])
        return pp.SO3(quat).matrix()

    @staticmethod
    def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
        # WGS84 constants
        a = 6378137.0
        e2 = 6.69437999014e-3
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        x = (N + h_m) * cos_lat * cos_lon
        y = (N + h_m) * cos_lat * sin_lon
        z = (N * (1.0 - e2) + h_m) * sin_lat
        return float(x), float(y), float(z)

    @staticmethod
    def _ecef_to_enu(x: float, y: float, z: float, lat0_deg: float, lon0_deg: float,
                    x0: float, y0: float, z0: float) -> tuple[float, float, float]:
        lat0 = np.deg2rad(lat0_deg)
        lon0 = np.deg2rad(lon0_deg)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        sin_lat = np.sin(lat0)
        cos_lat = np.cos(lat0)
        sin_lon = np.sin(lon0)
        cos_lon = np.cos(lon0)

        t = np.array([
            [-sin_lon,            cos_lon,           0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
        ], dtype=np.float64)

        e, n, u = t @ np.array([dx, dy, dz], dtype=np.float64)
        return float(e), float(n), float(u)

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
            "gps_latlon_sampled": lambda s: isinstance(s, str),
            "gps_flip_xy" : lambda b: isinstance(b, bool),
            "gps_swap_xy" : lambda b: isinstance(b, bool),
            "gps_use_init_orient": lambda b: isinstance(b, bool),
            "gt_yz_rot90" : lambda s: isinstance(s, str) and s in {"none", "cw", "ccw"},
            "gt_zz_rot90" : lambda s: isinstance(s, str) and s in {"none", "cw", "ccw"},
            "gt_rotations": lambda v: isinstance(v, list) and all(
                (isinstance(x, dict) and "axis" in x and "angle_deg" in x)
                or (hasattr(x, "axis") and hasattr(x, "angle_deg"))
                for x in v
            ),
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
