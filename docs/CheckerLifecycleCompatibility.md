# Six-task checker lifecycle compatibility record

## Status and scope

This record documents checker-related work that entered while the embodied
benchmark was expanded from `pull_out_key` and `put_bottle_in_shelf` to all
eight UniVTAC manipulation tasks. The intended feature was six-task P6/ICL
support; changing checker lifecycle was not part of the original feature
request. The compatibility layer is nevertheless included in the
`feat/six-task-p6-icl-rollouts` milestone so that its assumptions, current
evidence, and rollback boundary remain explicit.

- Baseline before the six-task work: `a09295c2f265d5e7e0e85c1a9de29ee915077bb5`.
- Affected new online tasks: `grasp_classify`, `insert_HDMI`, `insert_hole`,
  `insert_tube`, `lift_bottle`, and `lift_can`.
- Dynamically recaptured in-hand references: `insert_hole` and `insert_tube`
  only.
- Checker calibration status: provisional. Static/unit coverage and successful
  expert replay exist, but no controlled online Agent success A/B has yet shown
  that the new post-grasp reference timing is the best benchmark definition.

This change is evaluator-private. It does not add checker values, actor poses,
contact state, success hints, or reference poses to Agent-visible observations.

## Why adding tasks touched checker lifecycle

The upstream tasks were written for scripted expert evaluation. Their normal
sequence is:

1. reset actors;
2. execute privileged `pre_move()`;
3. grasp or pre-position the manipulated object;
4. initialize target/checker reference poses as a side effect of `pre_move()`;
5. execute `_play_once()`;
6. call the task's legacy `check_success()`.

The embodied v1 benchmark deliberately defaults to `pre_move=false`: the
robot starts at home and the object is ungrasped. Skipping `pre_move()` therefore
skipped more than privileged motion. For several tasks it also left fields such
as `target_pose`, `hole_pose`, or `origin_inhand_pose` undefined. Merely adding
the tasks to the public registry was not sufficient to run them.

The first compatibility change separates reference initialization from
privileged motion. The existing no-pre-move reset path calls
`initialize_task_references()` without moving the robot or any actor. Each task
also calls the same helper at the original point inside `pre_move()`, preserving
the scripted-expert ordering.

Most target references are functions only of reset-time fixtures and can be
initialized safely before grasp. Two tasks are different:

- `insert_hole` stores `origin_inhand_pose`, the peg pose expressed relative to
  the gripper center, and requires the current relative Z to remain within
  `0.04 m` of it.
- `insert_tube` stores the same kind of reference and uses a `0.03 m` bound.

In the upstream path these values are sampled after the scripted grasp. In an
ungrasped Agent reset, sampling them immediately describes "object on the
table relative to a distant gripper". Comparing a later valid in-hand pose to
that reset-time relation can reject an otherwise correct insertion. This is the
reason for the additional post-grasp lifecycle adapter.

## Exact implementation changes

### 1. Static task-reference extraction

`BaseTask` defines two no-op lifecycle hooks:

- `initialize_task_references()` initializes state that must exist after reset
  even when privileged `pre_move()` is disabled.
- `initialize_replay_task_phase()` refreshes state whose upstream meaning is
  specifically "after pre-move, before task motion".

The task changes are assignment extraction, not checker-formula changes:

| Task | `initialize_task_references()` | `initialize_replay_task_phase()` |
| --- | --- | --- |
| `grasp_classify` | generic benchmark 固定选择的绿色 target-pad pose | no override |
| `insert_HDMI` | connector target and hole poses | no override |
| `insert_hole` | hole/target poses and an initial in-hand fallback | refresh peg/gripper relative pose |
| `insert_tube` | hole pose and an initial in-hand fallback | refresh tube/gripper relative pose |
| `lift_bottle` | target pose beside the wall | no override |
| `lift_can` | can/gripper relative pose used by legacy early-stop logic | refresh the same relation at replay task phase |

The bodies and numerical thresholds of these tasks' `check_success()` methods
remain unchanged relative to `a09295c`. The extracted assignments are still
called from their original positions in `pre_move()`.

### 2. Full-trajectory replay phase boundary

Fixed expert HDF5 trajectories contain a single `pre_move -> task` phase
transition. `scripts/replay.py` now calls `initialize_replay_task_phase()` once
when replay reaches the first recorded task-phase frame. This reconstructs the
same reference timing that the upstream scripted path obtained after privileged
pre-move, while replay itself still starts from the ungrasped reset and applies
the complete recorded qpos trajectory.

This hook currently has task-specific effects for `insert_hole`, `insert_tube`,
and `lift_can`; it is a no-op for the other tasks.

### 3. Online post-grasp state machine

`agent_env/benchmark_checker.py` adds `PostGraspReferenceTracker`. It is enabled
only when all of the following are true:

- the task registry declares `requires_post_grasp_reference=true`;
- the online task starts ungrasped;
- the task is currently `insert_hole` or `insert_tube`.

Its state transitions are:

```text
armed
  -- successful delta_gripper < 0 --> closing + recapture reference

closing
  -- successful delta_gripper < 0 --> closing + recapture reference again
  -- first later successful arm change --> frozen

frozen
  -- successful delta_gripper > 0 --> armed
```

Failed actions do not change tracker state. While `closing`, every successful
incremental close recomputes `origin_inhand_pose`; the first subsequent
translation or rotation freezes the latest value before transport. For an
action that changes the arm and closes the gripper simultaneously, close has
priority: the reference is captured after that action and freezes on the next
successful arm-changing action.

Online lifecycle processing runs after `take_action()` returns. Consequently,
the stored value is the last close-time reference and is not recomputed during
the first transport action, but the `frozen_before_transport` audit event itself
is emitted after that first arm-changing action has executed. The event name
describes the reference boundary, not the event timestamp.

The tracker itself never reads object poses. It emits a lifecycle event, after
which the evaluator calls the task's private `initialize_replay_task_phase()`
hook. The capture/freeze events are written only to
`evaluator_private_audit.json`.

### 4. Terminal-policy routing required by six tasks

Before this milestone, the generic Agent environment recognized only the Key
policy and the special shelf-Bottle policy. Six-task support adds a registry
policy named `base_task_success_v1`; at `finish_episode`, those tasks:

1. settle for 60 physics steps;
2. call the task's unchanged legacy `check_success()` once;
3. use that boolean as `official_task_success`.

`put_bottle_in_shelf` retains its stronger conjunction of legacy success,
released gripper, and post-release stability. Replay now selects this policy by
explicit task metadata instead of `hasattr(task, "bottle")`. This matters
because `lift_bottle` also owns an actor named `bottle` but is a different task
whose upstream checker does not define the shelf release/stability contract.

The online and replay code currently use different identifier strings for the
same shelf-Bottle release/stability concept
(`released_stable_bottle_v1` and
`released_stable_manipulated_object_v1`). Their implemented numerical checks
match, but unifying the identifier is a future cleanup item.

## What did not change

- No task's `check_success()` expression or numerical threshold was relaxed or
  tightened.
- No task's `check_early_stop()` expression was changed.
- Online Agent steps still do not terminate or reveal success when a legacy
  checker becomes transiently true; only `finish_episode` publishes the result.
- Replay still applies the complete source trajectory and evaluates success
  only after terminal settling.
- The adapter performs no grasp, object motion, pose correction, or hidden
  controller action.
- Reference values and intermediate checker values remain evaluator-private.

## Current evidence

The following evidence supports implementation consistency, but not final
checker validity:

1. Unit tests cover open/close/reclose state transitions, failed-action
   behavior, and disabled tasks.
2. All eight registered fixed experts passed independent full-trajectory replay
   and their task terminal checks after 60 settling steps.
3. The six-task P6+bbox+mask+ICL smoke batch at
   `agent_runs/six_task_p6_bbox_mask_icl_seed_1830315042/` exercised the online
   path. Evaluator-private logs for `insert_hole_attempt2` and `insert_tube`
   contain the expected reference-capture events.
4. That smoke batch produced one official Agent success (`grasp_classify`).
   `insert_hole` and `insert_tube` both finished false, so those runs do **not**
   demonstrate that a physically successful insertion is accepted by the new
   lifecycle adapter.
5. At milestone commit time, 208 non-Isaac static tests passed, and all 192
   registered fixed-demo projections (eight tasks, six Profiles, four
   bbox/mask conditions) passed real-asset integrity and visibility validation.

The milestone therefore establishes that the six tasks can be launched,
observed, replayed, and terminally evaluated without missing-reference crashes.
It does not establish calibrated false-positive or false-negative rates for the
legacy checkers.

## Known limitations and risks

1. A successful negative gripper delta is only a grasp attempt, not proof of
   contact or object retention. The tracker does not inspect tactile contact,
   object motion, or attachment state.
2. Closing in free space and then moving the arm can freeze an invalid
   reference. The tracker re-arms only after a successful open command.
3. Reference capture occurs immediately after the gripper action, without a
   dedicated settling or relative-pose stability window.
4. The upstream in-hand checks use only one relative-position component and
   large task-specific tolerances; this milestone preserves those definitions
   rather than endorsing them.
5. `lift_can` refreshes its in-hand reference during expert replay, but online
   `base_task_success_v1` does not call its legacy `check_early_stop()`. The role
   of that early-stop condition in a terminal-only Agent benchmark remains to
   be decided.
6. A single fixed expert replay per task proves a positive scripted trajectory,
   not checker robustness across Agent strategies, seeds, failed grasps, or
   visually borderline outcomes.

## Required follow-up before declaring checker semantics stable

- Construct controlled successful `insert_hole` and `insert_tube` Agent-style
  trajectories and confirm both visual completion and terminal checker success.
- Add negative controls for closing in free space, dropping after grasp, partial
  insertion, and inserting while the object has slipped in the gripper.
- Compare command-based capture with contact/stability-based alternatives; do
  not expose any such private detector to the Agent.
- Review every task's terminal geometry and thresholds against continuous
  simulator-side video, not only sparse Agent observations.
- Decide whether legacy `check_early_stop()` conditions belong in the
  terminal-only benchmark contract.
- Version any future semantic change instead of silently changing
  `base_task_success_v1` results.

## Rollback boundary

The dynamic experiment can be removed without reverting six-task P6/ICL data
support by deleting:

- `agent_env/benchmark_checker.py` and its tests;
- `requires_post_grasp_reference` registry fields;
- `_update_post_grasp_reference()` and its call after online actions;
- task-specific dynamic `initialize_replay_task_phase()` calls where no replay
  reference restoration is desired.

Static target initialization must not simply be removed: tasks whose target
poses were historically created inside `pre_move()` would again fail in the
default ungrasped path. If the dynamic adapter is replaced, static target
initialization and in-hand reference semantics should be separated explicitly.
