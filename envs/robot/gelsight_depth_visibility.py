"""Runtime wrist-depth visibility policy for the GelSight Mini gripper."""

from __future__ import annotations

from typing import Any, Iterable


SECONDARY_RAY_ATTRIBUTE = "primvars:invisibleToSecondaryRays"

CASE_MESH_PATHS = ("case/mesh", "plate/mesh")
GELPAD_MESH_PATH = "mesh"


def _prim_path(prim: object) -> str:
    return str(prim.GetPath())


def _robot_root(path: str) -> str:
    prefix, separator, _ = path.partition("/Robot/")
    if not separator or not prefix.startswith("/World/envs/"):
        raise RuntimeError(
            f"GelSight root is outside a cloned robot environment: {path}"
        )
    return f"{prefix}/Robot"


def _unique_prims(prims: Iterable[object], *, role: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for prim in prims:
        path = _prim_path(prim)
        if path in result:
            raise RuntimeError(f"Duplicate GelSight {role} root: {path}")
        result[path] = prim
    return result


def _validate_root_layout(
    case_roots: dict[str, object],
    gelpad_roots: dict[str, object],
    *,
    expected_env_count: int,
) -> list[str]:
    expected_root_count = 2 * expected_env_count
    if len(case_roots) != expected_root_count:
        raise RuntimeError(
            "GelSight case root count mismatch: "
            f"expected {expected_root_count}, found {len(case_roots)} at "
            f"{sorted(case_roots)}"
        )
    if len(gelpad_roots) != expected_root_count:
        raise RuntimeError(
            "GelSight gelpad root count mismatch: "
            f"expected {expected_root_count}, found {len(gelpad_roots)} at "
            f"{sorted(gelpad_roots)}"
        )

    per_robot: dict[str, dict[str, list[str]]] = {}
    for role, roots in (("case", case_roots), ("gelpad", gelpad_roots)):
        for path in roots:
            robot_root = _robot_root(path)
            per_robot.setdefault(robot_root, {"case": [], "gelpad": []})[
                role
            ].append(path)

    if len(per_robot) != expected_env_count:
        raise RuntimeError(
            "GelSight robot count mismatch: "
            f"expected {expected_env_count}, found {len(per_robot)} at "
            f"{sorted(per_robot)}"
        )
    for robot_root, roots in sorted(per_robot.items()):
        if len(roots["case"]) != 2 or len(roots["gelpad"]) != 2:
            raise RuntimeError(
                f"GelSight root layout mismatch under {robot_root}: "
                f"case={sorted(roots['case'])}, "
                f"gelpad={sorted(roots['gelpad'])}"
            )
    return sorted(per_robot)


def _validate_mesh_prim(prim: object, path: str) -> None:
    if not prim or not prim.IsValid():
        raise RuntimeError(f"GelSight depth target does not exist: {path}")
    if not prim.IsActive():
        raise RuntimeError(f"GelSight depth target is inactive: {path}")
    if not prim.IsLoaded():
        raise RuntimeError(f"GelSight depth target is unloaded: {path}")
    if str(prim.GetTypeName()) != "Mesh":
        raise RuntimeError(f"GelSight depth target {path} is not a Mesh")
    if prim.IsInstanceProxy():
        raise RuntimeError(f"GelSight depth target is an instance proxy: {path}")
    if prim.IsInPrototype():
        raise RuntimeError(f"GelSight depth target is in a prototype: {path}")


def apply_gelsight_wrist_depth_visibility(
    stage: object,
    *,
    case_root_prims: Iterable[object],
    gelpad_root_prims: Iterable[object],
    expected_env_count: int,
    bool_type_name: object,
) -> dict[str, Any]:
    """Expose rigid housings to depth while keeping deformable gelpads hidden.

    The caller owns the USD edit target. Production uses the stage session
    layer, so these opinions never modify the vendored TacEx asset.
    """

    if expected_env_count <= 0:
        raise ValueError("expected_env_count must be positive")

    case_roots = _unique_prims(case_root_prims, role="case")
    gelpad_roots = _unique_prims(gelpad_root_prims, role="gelpad")
    robot_roots = _validate_root_layout(
        case_roots,
        gelpad_roots,
        expected_env_count=expected_env_count,
    )

    target_specs: list[tuple[str, str, str, bool]] = []
    for root_path in sorted(case_roots):
        robot_root = _robot_root(root_path)
        for relative_path in CASE_MESH_PATHS:
            target_specs.append(
                (
                    robot_root,
                    f"{root_path}/{relative_path}",
                    "rigid_case_or_plate",
                    False,
                )
            )
    for root_path in sorted(gelpad_roots):
        target_specs.append(
            (
                _robot_root(root_path),
                f"{root_path}/{GELPAD_MESH_PATH}",
                "deformable_gelpad",
                True,
            )
        )

    validated: list[tuple[str, str, str, bool, object, object | None]] = []
    for robot_root, path, role, desired in target_specs:
        prim = stage.GetPrimAtPath(path)
        _validate_mesh_prim(prim, path)
        attribute = prim.GetAttribute(SECONDARY_RAY_ATTRIBUTE)
        if attribute and attribute.IsValid():
            if str(attribute.GetTypeName()) != str(bool_type_name):
                raise RuntimeError(
                    f"GelSight depth target {path} has non-boolean "
                    f"{SECONDARY_RAY_ATTRIBUTE} type: "
                    f"{attribute.GetTypeName()}"
                )
            previous = attribute.Get()
            if not isinstance(previous, bool):
                raise RuntimeError(
                    f"GelSight depth target {path} has non-boolean "
                    f"{SECONDARY_RAY_ATTRIBUTE} value: {previous!r}"
                )
        else:
            attribute = None
        validated.append((robot_root, path, role, desired, prim, attribute))

    overrides: list[dict[str, Any]] = []
    for robot_root, path, role, desired, prim, attribute in validated:
        created = attribute is None
        if created:
            attribute = prim.CreateAttribute(
                SECONDARY_RAY_ATTRIBUTE,
                bool_type_name,
                custom=True,
            )
            if not attribute or not attribute.IsValid():
                raise RuntimeError(
                    f"Failed to create {SECONDARY_RAY_ATTRIBUTE} for {path}"
                )
            previous = None
        else:
            previous = attribute.Get()
        if attribute.Set(desired) is False:
            raise RuntimeError(
                f"Failed to author GelSight depth visibility for {path}"
            )
        current = attribute.Get()
        if not isinstance(current, bool) or current is not desired:
            raise RuntimeError(
                f"GelSight depth visibility did not compose for {path}: "
                f"expected {desired}, got {current!r}"
            )
        overrides.append(
            {
                "robot_root": robot_root,
                "prim_path": path,
                "role": role,
                "attribute_created": created,
                "previous_invisible_to_secondary_rays": previous,
                "invisible_to_secondary_rays": current,
            }
        )

    return {
        "schema_version": "univtac.gelsight_wrist_depth_visibility.v1",
        "robot_roots": robot_roots,
        "rigid_case_plate_visible": [
            item["prim_path"]
            for item in overrides
            if item["role"] == "rigid_case_or_plate"
        ],
        "deformable_gelpads_hidden": [
            item["prim_path"]
            for item in overrides
            if item["role"] == "deformable_gelpad"
        ],
        "overrides": overrides,
    }
