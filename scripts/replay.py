import sys

sys.path.append(".")
sys.path.append('../')

import os
import time
import json
import hashlib
import yaml
import torch
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import Literal

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay Data"
)
parser.add_argument(
    "task_name",
    type=str,
    help="Task name",
)
parser.add_argument(
    "task_config",
    type=str,
    help="Task name",
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
parser.add_argument(
    "--data-path",
    type=Path,
    default=None,
    help="Replay one explicit HDF5 trajectory instead of scanning the default dataset.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Environment seed for --data-path; defaults to a numeric HDF5 stem.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Override the replay artifact directory.",
)
parser.add_argument(
    "--trajectory-includes-pre-move",
    action="store_true",
    help=(
        "Reset to the ungrasped state and replay the complete recorded setup/grasp "
        "trajectory instead of executing privileged pre_move again."
    ),
)
parser.add_argument(
    "--stride",
    type=int,
    default=2,
    help="Apply every Nth recorded qpos frame (default preserves legacy replay).",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=60,
    help="Physics steps used for terminal stability verification.",
)
parser.add_argument(
    "--capture-p6-master",
    action="store_true",
    help=(
        "Capture a complete P6 observation at every recorded source waypoint. "
        "This records observation states only and does not derive step_eef actions."
    ),
)
parser.add_argument(
    "--fixed-expert-manifest",
    type=Path,
    default=None,
    help="Replay-proven frozen source manifest required by --capture-p6-master.",
)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
if args_cli.stride <= 0:
    parser.error("--stride must be a positive integer")
if args_cli.settle_steps < 0:
    parser.error("--settle-steps must be non-negative")
if args_cli.capture_p6_master:
    if not args_cli.trajectory_includes_pre_move:
        parser.error("--capture-p6-master requires --trajectory-includes-pre-move")
    if args_cli.stride != 1:
        parser.error("--capture-p6-master requires --stride 1")
    if args_cli.data_path is None or args_cli.fixed_expert_manifest is None:
        parser.error(
            "--capture-p6-master requires --data-path and --fixed-expert-manifest"
        )
    if args_cli.output_dir is None:
        parser.error("--capture-p6-master requires an explicit --output-dir")
    if args_cli.output_dir.exists():
        parser.error(
            f"--capture-p6-master output directory already exists: {args_cli.output_dir}"
        )
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1

if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
from typing import TYPE_CHECKING
from envs.utils.data import HDF5Handler
from agent_env.expert_trajectory import inspect_expert_hdf5
from agent_env.expert_tasks import get_expert_task
from agent_env.p6_expert_master import (
    build_p6_master_manifest,
    capture_p6_observation,
    configure_p6_capture_cfg,
    validate_fixed_expert_source,
)
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def _pose_snapshot(actor):
    pose = actor.get_pose()
    return {
        'position': np.asarray(pose.p, dtype=np.float64).tolist(),
        'quaternion_wxyz': np.asarray(pose.q, dtype=np.float64).tolist(),
    }


def _json_default(value):
    """Convert numeric scalar types produced by NumPy/Torch to JSON values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value, *, indent: int) -> None:
    """Serialize before opening the destination so failures cannot truncate it."""

    serialized = json.dumps(value, indent=indent, default=_json_default)
    path.write_text(serialized + "\n", encoding="utf-8")


def _pose_drift(before, after):
    before_position = np.asarray(before['position'], dtype=np.float64)
    after_position = np.asarray(after['position'], dtype=np.float64)
    before_quaternion = np.asarray(before['quaternion_wxyz'], dtype=np.float64)
    after_quaternion = np.asarray(after['quaternion_wxyz'], dtype=np.float64)
    before_quaternion /= np.linalg.norm(before_quaternion)
    after_quaternion /= np.linalg.norm(after_quaternion)
    translation = float(np.linalg.norm(after_position - before_position))
    rotation = float(
        2.0 * np.arccos(np.clip(abs(np.dot(before_quaternion, after_quaternion)), 0.0, 1.0))
    )
    return translation, rotation


def _terminal_checks(task, task_spec, pose_before, pose_after):
    base_success = bool(task.check_success())
    checks = {
        'base_task_success': base_success,
        'settle_steps': int(args_cli.settle_steps),
    }
    official_success = base_success
    if task_spec.requires_release_stability:
        translation_drift, rotation_drift = _pose_drift(pose_before, pose_after)
        gripper_qpos = float(task._robot_manager.get_gripper_qpos())
        released = gripper_qpos >= 0.0175
        stable = bool(
            translation_drift <= 0.01
            and rotation_drift <= float(np.deg2rad(10.0))
        )
        checks.update({
            'released': released,
            'gripper_qpos_m': gripper_qpos,
            'minimum_release_gripper_qpos_m': 0.0175,
            'stable_after_release': stable,
            'settle_translation_drift_m': translation_drift,
            'settle_rotation_drift_rad': rotation_drift,
            'max_translation_drift_m': 0.01,
            'max_rotation_drift_rad': float(np.deg2rad(10.0)),
        })
        official_success = bool(base_success and released and stable)
    return official_success, checks


def replay(task: 'BaseTask', seed, data_path:Path):
    eval_start = time.perf_counter()
    source_trajectory = None
    if args_cli.trajectory_includes_pre_move:
        # Reject historical post-pre_move episodes before they can be mistaken
        # for a complete ungrasped expert demonstration.
        source_trajectory = inspect_expert_hdf5(data_path)
    task_name = task.__class__.__module__.rsplit('.', 1)[-1]
    task_spec = get_expert_task(task_name)
    p6_source = None
    if args_cli.capture_p6_master:
        p6_source = validate_fixed_expert_source(
            manifest_path=args_cli.fixed_expert_manifest,
            source_hdf5=data_path,
            task=task_name,
            seed=int(seed),
        )
    task.reset(seed=seed)

    traj_data = HDF5Handler().load_hdf5(data_path)
    qpos_list = torch.from_numpy(traj_data['embodiment']['joint'][:, :8]).to(device=task.device)
    ee_list = torch.from_numpy(traj_data['embodiment']['ee'][:, :3]).to(device=task.device)
    source_steps = np.asarray(traj_data['step']).reshape(-1)
    if qpos_list.shape[0] == 0:
        raise ValueError(f'Trajectory contains no joint frames: {data_path}')

    traj_list = []
    applied_indices = list(np.arange(0, qpos_list.shape[0], args_cli.stride))
    if applied_indices[-1] != qpos_list.shape[0] - 1:
        applied_indices.append(qpos_list.shape[0] - 1)
    execution_succeeded = True
    physics_action_count = 0
    p6_waypoints = []
    previous_idx = None
    replay_task_phase_initialized = False
    task_phase_start_frame = (
        source_trajectory['task_phase_start_frame']
        if source_trajectory is not None
        else None
    )
    for idx in applied_indices:
        action = qpos_list[idx]
        if args_cli.trajectory_includes_pre_move and previous_idx is not None:
            sim_step_delta = int(source_steps[idx] - source_steps[previous_idx])
            previous_action = qpos_list[previous_idx]
            control_actions = [
                previous_action + (action - previous_action) * (substep / sim_step_delta)
                for substep in range(1, sim_step_delta + 1)
            ]
        else:
            sim_step_delta = 1
            control_actions = [action]

        for control_action in control_actions:
            if args_cli.trajectory_includes_pre_move:
                # A transiently true legacy checker must not stop a complete
                # replay before release/settling has been applied.
                task.eval_success = False
            exec_succ, eval_succ = task.take_action(
                control_action, action_type='qpos', force=True
            )
            physics_action_count += 1
            execution_succeeded = bool(execution_succeeded and exec_succ)
        if (
            task_phase_start_frame is not None
            and not replay_task_phase_initialized
            and idx >= task_phase_start_frame
        ):
            task.initialize_replay_task_phase()
            replay_task_phase_initialized = True
        observation = task._get_observations()
        arm_dis = torch.abs(action[:7] - observation['embodiment']['joint'][:7])
        gripper_dis = torch.abs(action[7] - observation['embodiment']['joint'][7:])
        ee_dis = torch.abs(ee_list[idx] - observation['embodiment']['ee'][:3])

        if torch.any(gripper_dis > 1e-3) or torch.any(ee_dis > 1e-3):
            log(f"[{idx:3d}] arm_dis: {np.max(arm_dis.cpu().numpy())}, gripper_dis: {np.max(gripper_dis.cpu().numpy())}, ee_dis: {ee_dis.cpu().numpy()}, eval_succ: {eval_succ}, exec_succ: {exec_succ}")
        
        traj_list.append({
            'target_ee': ee_list[idx].cpu().tolist(),
            'target_action': action.cpu().tolist(),
            'result_qpos': observation['embodiment']['joint'][:8].cpu().tolist(),
            'result_ee': observation['embodiment']['ee'][:3].cpu().tolist(),
            'source_sim_step': int(source_steps[idx]),
            'replayed_physics_steps': sim_step_delta,
        })
        if args_cli.capture_p6_master:
            observation_id = f'demo_obs_{idx:03d}'
            p6_observation, tactile_health = capture_p6_observation(
                task=task,
                task_name=task_name,
                observation_id=observation_id,
                observation_root=task.save_root / 'p6_observations',
                initial_observation=idx == applied_indices[0],
            )
            p6_waypoints.append({
                'source_frame_index': int(idx),
                'source_sim_step': int(source_steps[idx]),
                'observation_file': str(
                    (
                        task.save_root
                        / 'p6_observations'
                        / p6_observation['observation_id']
                        / 'observation.json'
                    ).resolve()
                ),
                'tactile_health': tactile_health,
            })
        previous_idx = idx

    manipulated_actor = task_spec.manipulated_actor(task)
    pose_before = _pose_snapshot(manipulated_actor)
    task.eval_success = False
    task.delay(steps=args_cli.settle_steps, is_save=True, force=True)
    pose_after = _pose_snapshot(manipulated_actor)
    official_success, evaluator_checks = _terminal_checks(
        task, task_spec, pose_before, pose_after
    )

    seed_root = task.save_root / 'replay_traj'
    seed_root.mkdir(parents=True, exist_ok=True)
    _write_json(seed_root / f'{seed}.json', traj_list, indent=4)

    eval_cost = time.perf_counter() - eval_start
    succ_status = 'success' if official_success else 'failed'
    task.clean_cache(result=succ_status)
    video_path = task.save_video_path.with_name(
        f'{task.save_video_path.stem}_{succ_status}{task.save_video_path.suffix}'
    )
    report = {
        'schema_version': 'univtac.fixed_expert_replay.v1',
        'task': task_name,
        'seed': int(seed),
        'source_hdf5': str(data_path.resolve()),
        'source_trajectory': source_trajectory,
        'trajectory_includes_pre_move': bool(args_cli.trajectory_includes_pre_move),
        'privileged_pre_move_executed_before_replay': bool(task.cfg.execute_pre_move),
        'source_frame_count': int(qpos_list.shape[0]),
        'stride': int(args_cli.stride),
        'applied_frame_count': len(applied_indices),
        'physics_action_count': physics_action_count,
        'timing_mode': (
            'recorded_step_interpolation'
            if args_cli.trajectory_includes_pre_move
            else 'one_action_per_frame'
        ),
        'execution_succeeded': execution_succeeded,
        'official_task_success': official_success,
        'evaluator_checks': evaluator_checks,
        'terminal_policy': task_spec.terminal_policy,
        'pose_before_settle': pose_before,
        'pose_after_settle': pose_after,
        'replay_video': str(video_path.resolve()),
        'wall_seconds': eval_cost,
    }
    if args_cli.capture_p6_master:
        if p6_source is None:
            raise RuntimeError('P6 source validation was not initialized')
        p6_manifest = build_p6_master_manifest(
            task=task_name,
            seed=int(seed),
            fixed_expert_manifest_path=p6_source['manifest_path'],
            source_hdf5=data_path,
            master_root=task.save_root,
            waypoint_records=p6_waypoints,
            physics_action_count=physics_action_count,
            execution_succeeded=execution_succeeded,
            official_task_success=official_success,
            evaluator_checks=evaluator_checks,
            replay_video=video_path,
        )
        p6_manifest_path = task.save_root / 'p6_master_manifest.json'
        _write_json(p6_manifest_path, p6_manifest, indent=2)
        report['p6_master'] = {
            'manifest': str(p6_manifest_path.resolve()),
            'sha256': hashlib.sha256(p6_manifest_path.read_bytes()).hexdigest(),
            'observation_count': len(p6_waypoints),
            'actions_present': False,
        }
    report_root = task.save_root / 'replay_report'
    report_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_root / f'{seed}.json', report, indent=2)
    return succ_status, eval_cost

def replay_seeds(task: 'BaseTask', data):
    test_num, succ_num = 0, 0
    for seed, data_path in data:
        test_num += 1
        result, eval_cost = replay(task, seed, data_path)
        succ_num += 1 if result == 'success' else 0
        log(f"[{test_num:<3d}] Seed {seed} {result} after {eval_cost:.2f} s.\n"
        f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
        f"Instruction: {task.instruction}\n"
        f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")
    return {
        'test_num': test_num,
        'succ_num': succ_num
    }


def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

task_module, policy_module = None, None
def main():
    global args_cli, task_module, policy_module, log_path

    task_file_name = args_cli.task_name
    task_config_name = args_cli.task_config
    
    task_config, task_config_file = get_config(
        task_config_name, default_root=Path(__file__).parent.parent / 'task_config', type='yaml'
    )

    task_module = importlib.import_module(f"envs.{task_file_name}")

    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg:BaseTaskCfg = task_module.TaskCfg()
    env_cfg.save_dir = args_cli.output_dir or (
        Path('eval_result') / 'replay' / task_file_name / task_config_file.stem / curr_time
    )
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.execute_pre_move = not args_cli.trajectory_includes_pre_move
    if args_cli.trajectory_includes_pre_move:
        env_cfg.step_lim = max(env_cfg.step_lim, 5000)
    if args_cli.capture_p6_master:
        configure_p6_capture_cfg(env_cfg)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None \
        else env_cfg.sim.device

    init_start = time.perf_counter()
    task:BaseTask = task_module.Task(env_cfg, mode='eval')
    task_init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}") 
    log(f"Task init finish in {task_init_cost:.2f} seconds.")
    
    if args_cli.data_path is not None:
        data_path = args_cli.data_path.resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f'Explicit replay trajectory does not exist: {data_path}')
        if args_cli.seed is not None:
            seed = args_cli.seed
        else:
            try:
                seed = int(data_path.stem)
            except ValueError as exc:
                raise ValueError(
                    '--seed is required when --data-path does not have a numeric stem'
                ) from exc
        data_root = data_path.parent
        data = [(seed, data_path)]
    else:
        data_root = Path(__file__).parent.parent / 'data' / task_file_name / task_config_name
        if (data_root / 'hdf5').exists():
            print(f"Found hdf5 data in {data_root / 'hdf5'}, start replaying.")
            # self collect data
            data_root = data_root / 'hdf5'
            data = sorted([(int(p.stem), p) for p in data_root.glob('*.hdf5')], key=lambda x: x[0])
        else:
            print(f"Found downloaded data in {data_root}, start replaying.")
            # dataset
            metadata_file = data_root / 'metadata.json'
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            data = []
            for k, v in metadata.items():
                if (data_root / f'{k}.hdf5').exists() and 'seed' in v:
                    data.append((int(v['seed']), data_root / f'{k}.hdf5'))
 
    log(f"Start replaying {len(data)} seeds from {data_root}.")

    results = replay_seeds(task, data=data)
    success_rate = (
        results['succ_num'] / results['test_num'] * 100
        if results['test_num']
        else 0.0
    )
    log(
        f"Final Result: {results['succ_num']}/{results['test_num']}"
        f"({success_rate:.2f}%) success."
    )
    
    task.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
