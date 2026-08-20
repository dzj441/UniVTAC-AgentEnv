from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from envs.robot.gelsight_depth_visibility import (
    SECONDARY_RAY_ATTRIBUTE,
    apply_gelsight_wrist_depth_visibility,
)


BOOL_TYPE = "bool"


class FakeAttribute:
    def __init__(
        self,
        value: object,
        *,
        valid: bool = True,
        type_name: object = BOOL_TYPE,
    ) -> None:
        self.value = value
        self.valid = valid
        self.type_name = type_name
        self.set_values: list[object] = []

    def __bool__(self) -> bool:
        return self.valid

    def IsValid(self) -> bool:
        return self.valid

    def Get(self) -> object:
        return self.value

    def GetTypeName(self) -> object:
        return self.type_name

    def Set(self, value: object) -> bool:
        self.value = value
        self.set_values.append(value)
        return True


@dataclass
class FakePrim:
    path: str
    attribute: FakeAttribute | None = None
    type_name: str = "Xform"
    valid: bool = True
    active: bool = True
    loaded: bool = True
    instance_proxy: bool = False
    in_prototype: bool = False
    created_attributes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def GetPath(self) -> str:
        return self.path

    def IsValid(self) -> bool:
        return self.valid

    def IsActive(self) -> bool:
        return self.active

    def IsLoaded(self) -> bool:
        return self.loaded

    def GetTypeName(self) -> str:
        return self.type_name

    def IsInstanceProxy(self) -> bool:
        return self.instance_proxy

    def IsInPrototype(self) -> bool:
        return self.in_prototype

    def GetAttribute(self, name: str) -> FakeAttribute:
        assert name == SECONDARY_RAY_ATTRIBUTE
        return self.attribute or FakeAttribute(None, valid=False)

    def CreateAttribute(
        self,
        name: str,
        type_name: object,
        *,
        custom: bool,
    ) -> FakeAttribute:
        assert custom is True
        self.created_attributes.append(name)
        self.attribute = FakeAttribute(None, type_name=type_name)
        return self.attribute


class FakeStage:
    def __init__(self, prims: list[FakePrim]) -> None:
        self.prims = {prim.path: prim for prim in prims}

    def GetPrimAtPath(self, path: str) -> FakePrim:
        return self.prims.get(path, FakePrim(path, valid=False))


def gelsight_scene(
    env_count: int = 1,
    *,
    pad_initially_hidden: bool = True,
) -> tuple[FakeStage, list[FakePrim], list[FakePrim], list[FakePrim]]:
    all_prims: list[FakePrim] = []
    case_roots: list[FakePrim] = []
    gelpad_roots: list[FakePrim] = []
    mesh_prims: list[FakePrim] = []
    for env_index in range(env_count):
        robot_root = f"/World/envs/env_{env_index}/Robot"
        for side in ("left", "right"):
            case_root = FakePrim(
                f"{robot_root}/gelsight_mini_case_{side}"
            )
            gelpad_root = FakePrim(f"{robot_root}/gelpad_{side}")
            case_roots.append(case_root)
            gelpad_roots.append(gelpad_root)
            all_prims.extend((case_root, gelpad_root))
            for child in ("case/mesh", "plate/mesh"):
                mesh = FakePrim(
                    f"{case_root.path}/{child}",
                    FakeAttribute(True),
                    type_name="Mesh",
                )
                mesh_prims.append(mesh)
                all_prims.append(mesh)
            pad_mesh = FakePrim(
                f"{gelpad_root.path}/mesh",
                FakeAttribute(pad_initially_hidden),
                type_name="Mesh",
            )
            mesh_prims.append(pad_mesh)
            all_prims.append(pad_mesh)
    return FakeStage(all_prims), case_roots, gelpad_roots, mesh_prims


def apply_policy(
    stage: FakeStage,
    case_roots: list[FakePrim],
    gelpad_roots: list[FakePrim],
    *,
    expected_env_count: int = 1,
) -> dict[str, object]:
    return apply_gelsight_wrist_depth_visibility(
        stage,
        case_root_prims=case_roots,
        gelpad_root_prims=gelpad_roots,
        expected_env_count=expected_env_count,
        bool_type_name=BOOL_TYPE,
    )


def test_runtime_policy_exposes_rigid_housing_and_hides_gelpad() -> None:
    stage, case_roots, gelpad_roots, meshes = gelsight_scene(
        env_count=2,
        pad_initially_hidden=False,
    )
    decoy = FakePrim(
        f"{case_roots[0].path}/decorative/mesh",
        FakeAttribute(True),
        type_name="Mesh",
    )
    stage.prims[decoy.path] = decoy

    report = apply_policy(
        stage,
        case_roots,
        gelpad_roots,
        expected_env_count=2,
    )

    assert report["schema_version"] == (
        "univtac.gelsight_wrist_depth_visibility.v1"
    )
    assert len(report["rigid_case_plate_visible"]) == 8
    assert len(report["deformable_gelpads_hidden"]) == 4
    for mesh in meshes:
        expected = "/gelpad_" in mesh.path
        assert mesh.attribute is not None
        assert mesh.attribute.value is expected
        assert mesh.attribute.set_values == [expected]
    assert decoy.attribute is not None
    assert decoy.attribute.value is True
    assert decoy.attribute.set_values == []


def test_missing_mesh_fails_before_authoring_any_override() -> None:
    stage, case_roots, gelpad_roots, meshes = gelsight_scene()
    del stage.prims[f"{gelpad_roots[-1].path}/mesh"]

    with pytest.raises(RuntimeError, match="does not exist"):
        apply_policy(stage, case_roots, gelpad_roots)

    assert all(mesh.attribute.set_values == [] for mesh in meshes)


def test_missing_attribute_is_created_and_pad_policy_is_enforced() -> None:
    stage, case_roots, gelpad_roots, meshes = gelsight_scene(
        pad_initially_hidden=False
    )
    target = meshes[0]
    target.attribute = None

    report = apply_policy(stage, case_roots, gelpad_roots)

    assert target.created_attributes == [SECONDARY_RAY_ATTRIBUTE]
    assert target.attribute is not None
    assert target.attribute.value is False
    created = [item for item in report["overrides"] if item["attribute_created"]]
    assert [item["prim_path"] for item in created] == [target.path]
    assert all(
        mesh.attribute is not None
        and mesh.attribute.value is ("/gelpad_" in mesh.path)
        for mesh in meshes
    )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("active", "inactive"),
        ("loaded", "unloaded"),
        ("instance_proxy", "instance proxy"),
        ("in_prototype", "in a prototype"),
    ),
)
def test_unsafe_mesh_state_is_rejected_before_authoring(
    field: str,
    message: str,
) -> None:
    stage, case_roots, gelpad_roots, meshes = gelsight_scene()
    setattr(meshes[-1], field, field in {"instance_proxy", "in_prototype"})

    with pytest.raises(RuntimeError, match=message):
        apply_policy(stage, case_roots, gelpad_roots)

    assert all(mesh.attribute.set_values == [] for mesh in meshes)


def test_wrong_mesh_or_attribute_type_is_rejected_before_authoring() -> None:
    stage, case_roots, gelpad_roots, meshes = gelsight_scene()
    meshes[-1].type_name = "Xform"
    with pytest.raises(RuntimeError, match="is not a Mesh"):
        apply_policy(stage, case_roots, gelpad_roots)
    assert all(mesh.attribute.set_values == [] for mesh in meshes)

    stage, case_roots, gelpad_roots, meshes = gelsight_scene()
    assert meshes[-1].attribute is not None
    meshes[-1].attribute.type_name = "token"
    with pytest.raises(RuntimeError, match="non-boolean.*type"):
        apply_policy(stage, case_roots, gelpad_roots)
    assert all(mesh.attribute.set_values == [] for mesh in meshes)


def test_root_counts_and_per_robot_layout_are_strict() -> None:
    stage, case_roots, gelpad_roots, _ = gelsight_scene()
    with pytest.raises(RuntimeError, match="case root count mismatch"):
        apply_policy(
            stage,
            case_roots[:-1],
            gelpad_roots,
        )

    stage, case_roots, gelpad_roots, _ = gelsight_scene(env_count=2)
    gelpad_roots[-1].path = (
        "/World/envs/env_0/Robot/extra_gelpad_from_wrong_environment"
    )
    with pytest.raises(RuntimeError, match="root layout mismatch"):
        apply_policy(
            stage,
            case_roots,
            gelpad_roots,
            expected_env_count=2,
        )
