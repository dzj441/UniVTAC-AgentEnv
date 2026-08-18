# Agent-visible information audit

This document records the information boundary used by every Codex Level of
the `grasp_classify` benchmark. Rich simulator responses are retained for host
replay and evaluator auditing; they are projected through a separate,
fail-closed allowlist before becoming a dynamic-tool result.

## Complete Codex input surface

During a rollout Codex receives only:

1. capability-isolation base/developer instructions;
2. the task name, semantic instruction, and generic stable-placement guidance;
3. the registered dynamic-tool names, input schemas, units, and bounds;
4. each tool's minimal projected result;
5. the sensor images attached to that result.

Dynamic-tool descriptions state only the static API lifecycle: start once,
optionally probe before committing, and use motion/wait/finish only after the
irreversible commitment. They also state the fixed probe/action limits. These
are callable-interface preconditions, not scene state or task guidance; live
stage and remaining-count values are still withheld.

The task prompt is identical at every Level:

```text
Task: grasp_classify

Determine whether the grasped prism is rough or plain. Place a rough prism on
the orange pad and a plain prism on the green pad.
For a stable placement, align the prism upright over the center of the target
pad before releasing it; avoid releasing near the pad edge.
```

The final sentence is a Level-independent manipulation prior: it says what a
stable placement looks like without revealing scene coordinates, checker
geometry, or a motion plan. The prompt still contains no suggested probe,
coordinate hint, action count, or feedback description.

## Allowed active-episode information

Every fresh observation contains exactly:

- `observation_id`;
- the Level-selected image modalities and transport provenance
  (`artifact_id`, `sha256`);
- `joint_position_8d`;
- `gripper_qpos`;
- `end_effector_pose_robot_base_7d`.

Level 1 receives head/wrist RGB. Level 2 additionally receives left/right
tactile marker RGB. Level 3 receives the Level-2 observation and, only after
the irreversible classification, `task_success`. Level 1 and Level 2 receive
no feedback field. Low-level action completion must be inferred from the fresh
robot state.

## Host-only fields

The following may exist in simulator transcripts, manifests, evaluator files,
or private audit records, but are never serialized into a Codex tool result:

- `targetward_world_y_sign`, target half-space state, or any equivalent
  target-direction hint;
- cumulative displacement and hand-authored motion guidance;
- protocol stage, profile/Level labels, probe/action counters, and remaining
  budgets;
- `inspect_episode_status` output;
- low-level `execution_succeeded` and execution duration;
- tactile marker counts, dark-pixel counts, and all tactile-health statistics;
- seed commitment/reveal values and host paths;
- predicted/committed values echoed by the environment;
- true class, expected target, reward/checker state, and terminal outcome.

Simulator-state rejection causes remain hidden. Pure input-schema errors return
an actionable schema message and repeat the latest already-public
`observation_id`; neither reveals new world state. Other failures remain a
generic contract rejection, preventing accepted/rejected trial calls from
becoming a simulator-state oracle. Invalid calls and their detailed host-only
cause remain fully recorded for offline audit.
The host aborts after 6 consecutive rejected calls so a malformed policy
cannot create an unbounded retry loop; that live counter is never returned to
Codex.

## Behavioral side channels

The former world-Y guard was removed, not merely redacted. A categorical
commitment remains immutable, but the robot may physically move in either sign
of X/Y/Z. Therefore action acceptance cannot reveal which half-space contains
the selected pad.

Level 1 and Level 2 do not auto-stop on private checker success. Level 3 does
auto-stop when its intentionally exposed post-commit success signal becomes
true; that behavior is part of the Level-3 intervention.

## Deliberate task assumptions, not hidden truth

The action schema defines world-frame XYZ/RPY, metres/radians, and numeric
bounds. The task instruction defines `rough -> orange` and `plain -> green`.
Basic robot state gives the end-effector pose in the robot base frame. These
are public task/action semantics, not simulator truth.

The current scene still uses fixed cameras and fixed pad locations. Codex
cannot read their source coordinates, but a policy evaluated repeatedly could
memorize the layout. Randomizing pad positions and, when desired, camera pose
is a separate benchmark-hardening step rather than an interface leak fix.

## Enforcement and tests

`agent_env.visibility` constructs the model-visible projection and then checks
its exact top-level, observation, modality, robot-state, and feedback keys.
Unknown additions fail closed. The gateway independently rejects private
actor/pad poses, raw depth/calibration, reward, contact forces, pre-terminal
truth, and Level-incompatible modalities before projection.

Static coverage is in:

- `tests/agent_env/test_visibility.py`;
- `tests/agent_env/test_capabilities.py`;
- `tests/agent_env/test_protocol.py`.

The Codex app-server process still shares the host UID, so this is a benchmark
capability boundary rather than a hostile-process security boundary. A public
adversarial deployment should additionally use a separate UID or container.
