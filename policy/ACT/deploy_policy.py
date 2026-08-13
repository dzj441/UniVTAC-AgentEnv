import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import time
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms
from PIL import Image, ImageDraw

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        # Construct checkpoint directory path
        self.train_config_name = os.environ.get('TRAIN_CONFIG', 'train_config')
        self.ep_num = os.environ.get('EP_NUM', '50')
        ckpt_dir = Path(__file__).parent / "act_ckpt" / f"act-{args['task_name']}" / f"{args['task_config']}-{self.ep_num}" / self.train_config_name
 
        self.task_name = args['task_name']
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

        # Optional audit-only recorder. It is disabled by default and does not
        # alter observations, checkpoint inference, or executed actions.
        trace_root = os.environ.get("UNIVTAC_OFFICIAL_TRACE_DIR")
        self.trace_root = Path(trace_root).resolve() if trace_root else None
        self.trace_episode = -1
        self.trace_step = 0
        if self.trace_root is not None:
            self.trace_root.mkdir(parents=True, exist_ok=True)
            trace_manifest = {
                "policy": "official UniVTAC ACT checkpoint",
                "task": self.task_name,
                "checkpoint_dir": str(ckpt_dir.resolve()),
                "policy_camera_input": self.camera_type,
                "policy_tactile_inputs": ["left rgb_marker", "right rgb_marker"],
                "policy_proprioception": "first 8 joint positions",
                "action_type": "absolute qpos: 7 arm joints + 1 gripper joint",
                "chunk_size": int(self.model.num_queries),
                "temporal_aggregation": bool(self.model.temporal_agg),
                "query_frequency": int(self.model.query_frequency),
                "wrist_camera_is_policy_input": self.camera_type in ["wrist", "all"],
                "note": "Wrist RGB is recorded only as an observer-side diagnostic when present.",
            }
            with open(self.trace_root / "trace_manifest.json", "w", encoding="utf-8") as f:
                json.dump(trace_manifest, f, ensure_ascii=False, indent=2)

    def encode_obs(self, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            return img

        if self.camera_type == 'all':
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = camera_transform(observation["observation"][self.camera_type]["rgb"])

        tactile = observation["tactile"]
        # Current TacEx environments expose the sensors as left_tactile and
        # right_tactile.  Keep accepting the legacy dataset/deployment names
        # so older environments continue to work as well.
        left_tactile = tactile.get("left_tactile", tactile.get("left_gsmini"))
        right_tactile = tactile.get("right_tactile", tactile.get("right_gsmini"))
        if left_tactile is None or right_tactile is None:
            raise KeyError(
                "Expected tactile observations named left/right_tactile or "
                f"left/right_gsmini, got {sorted(tactile)}"
            )
        left_tac = tactile_transform(left_tactile["rgb_marker"])
        right_tac = tactile_transform(right_tactile["rgb_marker"])
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:8]

        ret = {
            "cam_high": cam_high,
            "tac_left": left_tac,
            "tac_right": right_tac,
            "qpos": qpos.cpu().numpy()
        }
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if self.model.t % 10 == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        model_t = int(self.model.t)
        action_np = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action_np).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')
        if self.trace_root is not None:
            self._record_official_trace(
                task=task,
                observation=observation,
                model_t=model_t,
                action=action_np,
                exec_succ=exec_succ,
                eval_succ=eval_succ,
            )

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()
        if self.trace_root is not None:
            self.trace_episode += 1
            self.trace_step = 0

    @staticmethod
    def _trace_rgb(tensor):
        array = tensor.detach().cpu().numpy()
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        array = array[..., :3]
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if np.issubdtype(array.dtype, np.floating) and finite_max <= 1.0:
            array = array * 255.0
        return np.nan_to_num(array).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _save_trace_composite(head, left, right, wrist, path):
        panels = [
            ("POLICY INPUT: head RGB", Image.fromarray(head)),
            ("POLICY INPUT: left tactile marker", Image.fromarray(left)),
            ("POLICY INPUT: right tactile marker", Image.fromarray(right)),
        ]
        if wrist is not None:
            panels.append(("OBSERVER ONLY (not policy input): wrist RGB", Image.fromarray(wrist)))

        panel_width, panel_height, label_height = 480, 270, 24
        canvas = Image.new(
            "RGB", (panel_width * 2, (panel_height + label_height) * 2), "black"
        )
        draw = ImageDraw.Draw(canvas)
        for index, (label, panel) in enumerate(panels):
            x = (index % 2) * panel_width
            y = (index // 2) * (panel_height + label_height)
            draw.text((x + 6, y + 5), label, fill="white")
            canvas.paste(panel.resize((panel_width, panel_height)), (x, y + label_height))
        canvas.save(path)

    def _record_official_trace(self, task, observation, model_t, action, exec_succ, eval_succ):
        episode_dir = self.trace_root / f"episode_{self.trace_episode:03d}"
        frame_dir = episode_dir / "frames" / f"step_{self.trace_step:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        head = self._trace_rgb(observation["observation"]["head"]["rgb"])
        tactile = observation["tactile"]
        left_data = tactile.get("left_tactile", tactile.get("left_gsmini"))
        right_data = tactile.get("right_tactile", tactile.get("right_gsmini"))
        left = self._trace_rgb(left_data["rgb_marker"])
        right = self._trace_rgb(right_data["rgb_marker"])
        wrist_data = observation["observation"].get("wrist")
        wrist = self._trace_rgb(wrist_data["rgb"]) if wrist_data is not None else None

        paths = {
            "head_rgb_policy_input": frame_dir / "head_rgb_policy_input.png",
            "left_tactile_marker_policy_input": frame_dir / "left_tactile_marker_policy_input.png",
            "right_tactile_marker_policy_input": frame_dir / "right_tactile_marker_policy_input.png",
            "composite": frame_dir / "composite.png",
        }
        Image.fromarray(head).save(paths["head_rgb_policy_input"])
        Image.fromarray(left).save(paths["left_tactile_marker_policy_input"])
        Image.fromarray(right).save(paths["right_tactile_marker_policy_input"])
        if wrist is not None:
            paths["wrist_rgb_observer_only"] = frame_dir / "wrist_rgb_observer_only.png"
            Image.fromarray(wrist).save(paths["wrist_rgb_observer_only"])
        self._save_trace_composite(head, left, right, wrist, paths["composite"])

        record = {
            "timestamp_unix": time.time(),
            "episode": self.trace_episode,
            "trace_step": self.trace_step,
            "model_t": model_t,
            "sim_step_before_action": int(observation["step"]),
            "policy_inputs": {
                "camera": ["head RGB"],
                "tactile": ["left rgb_marker", "right rgb_marker"],
                "qpos_8d": observation["embodiment"]["joint"][:8].detach().cpu().tolist(),
            },
            "observer_only": ["wrist RGB"] if wrist is not None else [],
            "action": {
                "type": "absolute_qpos",
                "arm_joint_qpos_7d": np.asarray(action[:7]).tolist(),
                "gripper_qpos_1d": float(action[7]),
            },
            "act_inference": {
                "new_50_step_chunk_queried": model_t % self.model.query_frequency == 0,
                "temporal_aggregation": bool(self.model.temporal_agg),
            },
            "execution_succeeded": bool(exec_succ),
            "evaluator_success_after_action": bool(eval_succ),
            "files": {name: str(path.resolve()) for name, path in paths.items()},
        }
        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(episode_dir / "actions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.trace_step += 1

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')
