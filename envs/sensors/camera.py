from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from dataclasses import MISSING
from pxr import Gf, Usd, UsdGeom

import torch
import torchvision.transforms.functional as F
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc import UipcInteractiveScene

@configclass
class CameraCfg(TiledCameraCfg):
    name: str = 'camera'


@configclass
class RigidCameraMountCfg:
    """Post-physics visual/optical pose for a rigid camera mount.

    The mesh must not contain collision geometry. Applying this configuration
    after the robot is initialized keeps its articulation and mass properties
    unchanged while making the rendered housing agree with the camera sensor.
    Quaternion fields use USD's ``wxyz`` convention.
    """

    camera_prim_path: str = MISSING
    mesh_prim_path: str = MISSING
    camera_translation: tuple[float, float, float] = MISSING
    camera_orientation_wxyz: tuple[float, float, float, float] = MISSING
    mesh_translation: tuple[float, float, float] = MISSING
    mesh_orientation_wxyz: tuple[float, float, float, float] = MISSING


class CameraManager:
    def __init__(self, cfg_list: list[CameraCfg], task:'BaseTask'):
        self.task = task
        self.scene = task.scene
        self.cfg_list = cfg_list
        self.cameras = {}

    def setup(self): 
        mount_cfg = self.task.cfg.robot.rigid_camera_mount
        if mount_cfg is not None:
            self._apply_rigid_camera_mount(mount_cfg)
        self.cameras = {
            cam_cfg.name: self.add_camera(cam_cfg) for cam_cfg in self.cfg_list
        }

    def _apply_rigid_camera_mount(self, cfg: RigidCameraMountCfg):
        camera_prims = sim_utils.find_matching_prims(cfg.camera_prim_path)
        mesh_prims = sim_utils.find_matching_prims(cfg.mesh_prim_path)
        if len(camera_prims) != self.task.num_envs or len(mesh_prims) != self.task.num_envs:
            raise RuntimeError(
                "Rigid camera mount prim count mismatch: "
                f"expected {self.task.num_envs}, got "
                f"{len(camera_prims)} cameras and {len(mesh_prims)} meshes"
            )

        for mesh_prim in mesh_prims:
            collision_prims = [
                prim.GetPath()
                for prim in Usd.PrimRange(mesh_prim)
                if any("CollisionAPI" in str(api) for api in prim.GetAppliedSchemas())
            ]
            if collision_prims:
                raise RuntimeError(
                    "Refusing to move a post-physics camera mesh with collision geometry: "
                    f"{collision_prims}"
                )
            self._set_local_pose(
                mesh_prim,
                cfg.mesh_translation,
                cfg.mesh_orientation_wxyz,
            )
            UsdGeom.Imageable(mesh_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

        for camera_prim in camera_prims:
            self._set_local_pose(
                camera_prim,
                cfg.camera_translation,
                cfg.camera_orientation_wxyz,
            )

    @staticmethod
    def _set_local_pose(prim, translation, orientation_wxyz):
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        if not translate_attr.IsValid() or not orient_attr.IsValid():
            raise RuntimeError(
                f"Camera mount prim {prim.GetPath()} lacks translate/orient xform ops"
            )
        translate_attr.Set(Gf.Vec3d(*translation))
        orient_attr.Set(
            Gf.Quatf(
                orientation_wxyz[0],
                Gf.Vec3f(*orientation_wxyz[1:]),
            )
        )

    def add_camera(self, cam_cfg: CameraCfg):
        camera = TiledCamera(cam_cfg)
        camera._initialize_impl()
        camera._is_initialized = True
        self.scene.sensors[f'camera_{cam_cfg.name}'] = camera
        return camera
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgba']
        for name, cam in self.cameras.items():
            obs[name] = {}
            for data_type in data_types:
                if data_type == 'rgb':
                    obs[name]['rgb'] = cam.data.output['rgb'].squeeze(0)
                elif data_type == 'rgba':
                    obs[name]['rgba'] = cam.data.output['rgba'].squeeze(0)
                elif data_type == 'depth':
                    obs[name]['depth'] = cam.data.output['depth'].squeeze(0)
        return obs
