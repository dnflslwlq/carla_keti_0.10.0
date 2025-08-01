"""NuScenes export helper. Drop this file next to main.py and import it.

Usage sketch (inside CarlaSyncMode.tick after you computed all per-frame data):
    if self.frame_counter == 1:
        exporter = NuScenesExporter(root_save_dir, scene_name="town01-clear")
        exporter.register_scene(start_timestamp_us)

    exporter.add_ego_pose(vehicle_transform)
    exporter.add_sample(sensor_tokens_dict)  # mapping sensor_id -> saved file path
    exporter.add_annotations(bbox_list)      # list of (class_id, center3d, size3d, yaw)

    if done:
        exporter.finalize()

This keeps the main code change minimal - you only call exporter methods at the
points indicated in the chat instructions. All NuScenes tables are written to
<root_save_dir>/nuscenes/ and are ready for the devkit.
"""
from __future__ import annotations

import os
import json
import uuid
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import carla

# -------- helpers -----------------------------------------------------------

def new_token() -> str:
    """Return a 32-byte lowercase hex UUID, NuScenes compatible."""
    return uuid.uuid4().hex


def transform_to_nusc_xyzlwhyaw(transform, size_xyz) -> Tuple[List[float], List[float], float]:
    """Convert CARLA transform (left-handed) + size to NuScenes right-handed.

    Args:
        transform: carla.Transform with location&rotation in world frame.
        size_xyz: tuple (extent.x*2, extent.y*2, extent.z*2)  (w,l,h in UE axes)

    Returns:
        center   - [x, y, z]  (x forward, y left, z up)
        size_xyz - [w, l, h]   (meters)  (swap axes to nusc convention)
        yaw      - rotation around z (radians, right-handed)
    """
    # UE → NuScenes axis mapping:  nusc = [y, -x, z]
    loc = transform.location
    center = [loc.y, -loc.x, loc.z]

    # carla extent.x is half length (in UE x‑forward).  nusc size order = w(lateral),l(long),h
    w = size_xyz[1]  # UE y‑extent*2 ⇒ width (left‑right) in nusc
    l = size_xyz[0]  # UE x‑extent*2 ⇒ length (forward)  in nusc
    h = size_xyz[2]  # UE z*2 stays
    size_nusc = [w, l, h]

    # Yaw: UE y‑up left-handed → right‑handed:   yaw_nusc = -yaw_ue (rad)
    yaw_nusc = -np.deg2rad(transform.rotation.yaw)
    return center, size_nusc, yaw_nusc

# ---------------------------------------------------------------------------

class NuScenesExporter:
    """Collect rows for all NuScenes tables and write them in one shot."""

    def __init__(self, root_dir: str, scene_name: str):
        self.root = Path(root_dir) / "nuscenes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.scene_name = scene_name

        # buffers
        self.logs = []
        self.scenes = []
        self.samples = []
        self.sample_data = []
        self.sample_annotations = []
        self.ego_poses = []
        self.calibrated_sensors = []
        self.sensors = []
        self.instances = []

        # running state
        self.scene_token = new_token()
        self.last_sample_token: Optional[str] = None
        self.cur_sample_count = 0

        # mapping dicts
        self.sensor_token_map: Dict[str, str] = {}          # sensor_name -> sensor_token
        self.calib_token_map: Dict[str, str] = {}           # sensor_name -> calib_token
        self.instance_token_map: Dict[int, str] = {}        # actor_id -> instance_token

        # timestamp tracking
        self.first_timestamp = None
        self.current_timestamp_us = None
        self._last_pose_token = None

        # ------------- create static rows -----------------------------------
        self._create_static_log()

    # ---------------- public API -------------------------------------------

    def register_scene(self, first_timestamp_us: int):
        """Call ONCE, right after simulation starts."""
        self.first_timestamp = first_timestamp_us
        return self.scene_token  

    def add_sensor(self, sensor_id: str, sensor_bp, transform, K):
        """Register sensor meta once. Provide CARLA blueprint & extr/intrinsic."""
        if sensor_id in self.sensor_token_map:
            return

        bp_id = sensor_bp.id
        if not (bp_id.startswith("sensor.lidar") or 
                bp_id == "sensor.camera.rgb"):
            return  

        sensor_token = new_token()
        calib_token = new_token()
        self.sensor_token_map[sensor_id] = sensor_token
        self.calib_token_map[sensor_id] = calib_token

        modality = "lidar" if bp_id.startswith("sensor.lidar") else "camera"
        self.sensors.append({
            "token": sensor_token,
            "channel": sensor_id.upper(),
            "modality": modality
            #"sensor_type": sensor_bp.id, 
            #"description": sensor_bp.id
        })

        # extrinsic (car frame → sensor frame)  (NuScenes uses sensor in vehicle frame)
        # carla Transform: UE left-handed. Convert to right‑handed.
        loc = transform.location
        rot = transform.rotation
        translation = [loc.y, -loc.x, loc.z]
        rotation = list(carla_quat_to_nusc(rot))  # helper defined later

        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        self.calibrated_sensors.append({
            "token": calib_token,
            "sensor_token": sensor_token,
            "translation": translation,
            "rotation": rotation,
            "camera_intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        })

    def new_sample(self, timestamp_us: int, scene_token: str):
        sample_token = new_token()
        if self.last_sample_token is None:
            prev = ""
        else:
            prev = self.last_sample_token
        
        self.current_timestamp_us = timestamp_us

        # sample row
        self.samples.append({
            "token": sample_token,
            "scene_token": scene_token,
            "timestamp": timestamp_us,
            "prev": prev,
            "next": ""  # filled later
        })
        if prev:
            for sample in self.samples:
                if sample["token"] == prev:
                    sample["next"] = sample_token
                    break

        self.last_sample_token = sample_token
        self.cur_sample_count += 1
        return sample_token

    def add_ego_pose(self, veh_transform, timestamp_us: int):
        """에고 포즈 추가 - timestamp를 파라미터로 받도록 수정"""
        pose_token = new_token()
        t = veh_transform
        loc = [t.location.y, -t.location.x, t.location.z]
        rot = list(carla_quat_to_nusc(t.rotation))
        self.ego_poses.append({
            "token": pose_token,
            "translation": loc,
            "rotation": rot,
            "timestamp": timestamp_us
        })
        self._last_pose_token = pose_token
        return pose_token

    def add_sample(self, sensor_files: Dict[str, str]):
        """sensor_files: sensor_id -> filename (relative path under samples/)"""
        # sample_data rows
        for sensor_id, rel_path in sensor_files.items():
            if sensor_id in self.calib_token_map:
                self.sample_data.append({
                    "token": new_token(),
                    "sample_token": self.last_sample_token,
                    "ego_pose_token": self._last_pose_token,
                    "calibrated_sensor_token": self.calib_token_map[sensor_id],
                    "timestamp": self.current_timestamp_us,
                    "fileformat": Path(rel_path).suffix[1:],
                    "is_key_frame": True,
                    "height": 0,  # fill if image
                    "width": 0,
                    "filename": rel_path
                })

    def add_annotations(self, annos: List[Tuple[int, List[float], List[float], float]]):
        """annos: (class_id, center_xyz, size_xyz, yaw) in right‑handed world frame."""
        if self.last_sample_token is None:
            return
            
        for class_id, center, size, yaw in annos:
            inst_token = self._instance_token(class_id)
            self.sample_annotations.append({
                "token": new_token(),
                "sample_token": self.last_sample_token,
                "instance_token": inst_token,
                "translation": center,
                "size": size,  # [w,l,h]
                "rotation": euler_to_quat(yaw),
                "velocity": [0.0, 0.0, 0.0],
                "attribute_tokens": []
            })

    def finalize(self):
        if not self.samples:
            print("[NuScenesExporter] No samples to export")
            return
            
        # fill scene row
        self.scenes.append({
            "token": self.scene_token,
            "log_token": self.logs[0]["token"],
            "name": self.scene_name,
            "description": "generated by CARLA exporter",
            "nbr_samples": self.cur_sample_count,
            "first_sample_token": self.samples[0]["token"],
            "last_sample_token": self.samples[-1]["token"]
        })

        # write all tables
        tables = {
            "scene": self.scenes,
            "sample": self.samples,
            "sample_data": self.sample_data,
            "ego_pose": self.ego_poses,
            "calibrated_sensor": self.calibrated_sensors,
            "sensor": self.sensors,
            "instance": self.instances,
            "sample_annotation": self.sample_annotations,
            "log": self.logs,
        }
        for name, rows in tables.items():
            with open(self.root / f"{name}.json", "w") as f:
                json.dump(rows, f, indent=2)
        print(f"[NuScenesExporter] Wrote {len(tables)} tables to {self.root}")

    # ---------------- internal helpers -------------------------------------

    def _instance_token(self, class_id: int):
        if class_id not in self.instance_token_map:
            self.instance_token_map[class_id] = new_token()
            self.instances.append({
                "token": self.instance_token_map[class_id],
                "category_token": self._category_token(class_id)
            })
        return self.instance_token_map[class_id]

    def _category_token(self, class_id: int):
        CATEGORY_MAP = {0: "vehicle.car", 1: "vehicle.bicycle", 2: "human.pedestrian"}
        return CATEGORY_MAP.get(class_id, "vehicle.car")

    # 흠;;; 여기도 좀;;; 
    def _create_static_log(self):
        self.logs.append({
            "token": new_token(),
            "logfile": "CARLA",
            "vehicle": "hero",
            "date_captured": "2025-05-28",  #
            "location": "korea",
        })

# ---- quaternion helpers ----------------------------------------------------

def carla_quat_to_nusc(rot):
    """CARLA Euler → NuScenes quaternion [w,x,y,z] right-handed."""
    # CARLA rotation is degrees in UE left-handed X forward Y right Z up
    # Convert to right‑handed yaw‑pitch‑roll then to quaternion.
    roll = math.radians(rot.roll)
    pitch = -math.radians(rot.pitch)
    yaw = -math.radians(rot.yaw)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z

def euler_to_quat(yaw):
    """Yaw‑only quaternion (roll=pitch=0). Input yaw rad, right-handed."""
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return [w, 0.0, 0.0, z]