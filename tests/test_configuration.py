"""Level configuration, run planning, and path checks."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sys
import unittest

import pytest
import yaml

from swarmecho.core.compatibility import resolve_level_file
from swarmecho.core.config import LEVEL_DIR, load_level, load_level_cli
from swarmecho.core.paths import normalize_wsl_path
from swarmecho.training.artifacts import (
    checkpoint_artifact_suffix,
    create_eval_run_root,
    eval_checkpoint_replay_root,
    train_replay_root,
)
from swarmecho.training.orchestration import (
    checkpoint_total_steps,
    generate_unique_seeds,
    pop_arg,
    split_levels,
)


# G03 CLU configuration

def test_unknown_environment_parameter_is_rejected(tmp_path):
    source = LEVEL_DIR / "G" / "G03_CLU_random_buildings.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data = deepcopy(data)
    data["env"]["typo_parameter"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown env fields: typo_parameter"):
        load_level(path)


def test_geodesic_reward_does_not_require_generated_obstacles():
    level = load_level_cli([
        "level=G03_CLU_random_buildings", "reward.chain_reward_system=obstacle_geodesic",
        "env.num_obstacles=0",
    ])
    assert level.env.num_obstacles == 0
    assert level.building.solid_min_m.shape[0] > 0
    assert level.reward.chain_reward_system == "obstacle_geodesic"


def test_cli_accepts_dotlist_overrides():
    level = load_level_cli(
        [
            "level=G03_CLU_random_buildings",
            "training.total_timesteps=300000",
            "logging.run_name=inspector_smoke",
            "env.wall_sensing_rays=16",
            "env.drone_radar_bins=16",
            "env.coverage_voxel_size=2.5",
        ]
    )

    assert level.training.total_timesteps == 300_000
    assert level.logging.run_name == "inspector_smoke"
    assert level.env.wall_sensing_rays == level.env.drone_radar_bins == 16
    assert level.env.coverage_voxel_size == 2.5
    assert level.num_updates == 1


def test_cli_can_enable_training_only_action_noise():
    level = load_level_cli(
        [
            "level=G03_CLU_random_buildings",
            "training.training_noise=true",
            "training.noise_level=0.02",
        ]
    )

    assert level.training.training_noise
    assert level.training.noise_level == 0.02
    # This setting is solely consumed by the training rollout; it must not
    # alter the standalone robust-evaluation configuration.
    assert level.evaluation.eval_action_noise_max == 0.0
    assert not level.evaluation.training_heatmap_creation


def test_replays_use_the_maintained_artifact_hierarchy(tmp_path):
    level = load_level("G03_CLU_random_buildings")
    checkpoint = tmp_path / "checkpoints/ckpt_000050"

    assert train_replay_root(tmp_path) == tmp_path / "artifacts/train/replays"
    assert checkpoint_artifact_suffix(checkpoint, level) == "u000050_s00015M"
    assert eval_checkpoint_replay_root(tmp_path, checkpoint, level) == (
        tmp_path / "artifacts/eval/ckpt_u000050_s00015M/replays"
    )


def test_manual_evaluations_use_unique_named_run_folders(tmp_path):
    level = load_level("G03_CLU_random_buildings")
    checkpoint = tmp_path / "checkpoints/ckpt_000050"
    timestamp_ns = 1_725_000_000_123_456_789

    named = create_eval_run_root(
        tmp_path, checkpoint, level, eval_name="agents7", timestamp_ns=timestamp_ns
    )
    collision_safe = create_eval_run_root(
        tmp_path, checkpoint, level, eval_name="agents7", timestamp_ns=timestamp_ns
    )

    assert named.name == "eval_agents7_20240830T064000_123456789Z"
    assert collision_safe.name == f"{named.name}_1"
    assert named.parent == tmp_path / "artifacts/eval/ckpt_u000050_s00015M"


def test_g03_clu_is_solvable_with_automatic_deployment_and_held_out_validation():
    level = load_level()
    canonical = LEVEL_DIR / "G" / "G03_CLU_random_buildings.yaml"
    assert level.name == canonical.stem
    assert resolve_level_file("G03", LEVEL_DIR) == canonical
    assert resolve_level_file("G03_random_buildings", LEVEL_DIR) == canonical
    assert level.ideal_chain_margin_m > 0
    assert level.env.num_agents == level.env.adapt_size.initial_agents == 7
    assert not level.env.adapt_size.recruitment_enabled
    assert level.training.num_maps == 3600
    assert level.validation.enabled and level.validation.num_maps == 400


# Orchestration

def test_generate_unique_seeds_is_reproducible():
    assert generate_unique_seeds(3, 9) == [801681273, 994300728, 1316869688]
    assert len(generate_unique_seeds(20, 9)) == 20


def test_pop_arg_and_split_levels_remove_orchestration_arguments():
    count, remaining = pop_arg(["levels=A,B", "seeds=3", "logging.run_name=x"], "seeds", 3)
    assert count == 3
    assert remaining == ["levels=A,B", "logging.run_name=x"]

    levels, remaining = split_levels(remaining, "default")
    assert levels == ["A", "B"]
    assert remaining == ["logging.run_name=x"]


def test_checkpoint_total_steps_prefers_checkpoint_history(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "ckpt_000050"
    checkpoint.mkdir(parents=True)
    (checkpoint / "step_history.json").write_text(
        json.dumps({"total_steps": 123456, "history": []}), encoding="utf-8"
    )
    assert checkpoint_total_steps(checkpoint, num_envs=4000, num_steps=100) == 123456


def test_checkpoint_total_steps_falls_back_to_checkpoint_name(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "ckpt_000050"
    checkpoint.mkdir(parents=True)
    assert checkpoint_total_steps(checkpoint, num_envs=4000, num_steps=100) == 20_000_000


# Portable paths

def test_normalize_wsl_path_converts_windows_drive_paths():
    assert normalize_wsl_path(
        r"Q:\_0_Projects\000_SwarmEcho\SwarmEcho\outputs\run\checkpoints\ckpt_001201"
    ) == Path(
        "/mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho/outputs/run/checkpoints/ckpt_001201"
    )


def test_normalize_wsl_path_preserves_linux_paths():
    path = "/mnt/q/_0_Projects/000_SwarmEcho/SwarmEcho/outputs/run"
    assert normalize_wsl_path(path) == Path(path)


def test_normalize_wsl_path_accepts_forward_slash_windows_paths():
    assert normalize_wsl_path(r"Q:/outputs/run/checkpoints/ckpt_001201") == Path(
        "/mnt/q/outputs/run/checkpoints/ckpt_001201"
    )


# Cluster capacity planning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"cluster/tools"))
from capacity import collect, summarize, task_metadata


def job(identity, slots, kind="TYPE_EXPERIMENT", workspace=129):
    return dict(jobId=identity, allocatedSlots=slots, requestedSlots=slots+7,
        type=kind, workspaceId=workspace)


def meta(identity, project=367, user="me", kind="experiment"):
    return dict(id=identity, name=identity, project_id=project, project="swarm_echo",
        username=user, kind=kind, state="RUNNING")


class CapacityTests(unittest.TestCase):
    def test_slot_counts_overlap_and_use_ids_not_names(self):
        jobs=[job("own",3),job("same-project-other",4),job("other-project-own",2),
            job("other-project-other",1),job("shell-other",3,"TYPE_SHELL"),
            job("shell-own",1,"TYPE_SHELL"),job("queued",0),job("elsewhere",20,workspace=999)]
        metadata={"own":meta("own"),"same-project-other":meta("same-project-other",user="other"),
            "other-project-own":meta("other-project-own",project=555),
            "other-project-other":meta("other-project-other",project=555,user="other"),
            "shell-other":meta("shell-other",project=None,user="other",kind="shell"),
            "shell-own":meta("shell-own",project=None,kind="shell")}
        result=summarize(jobs,metadata,workspace_id=129,project_id=367,username="me")
        self.assertEqual(result["allocated_slots"],14)
        self.assertEqual(result["outside_project_slots"],7)
        self.assertEqual(result["outside_user_slots"],8)
        self.assertEqual(result["free_slots"],10)
        self.assertEqual(len(result["allocations"]),6)

    def test_missing_metadata_is_reported_unknown(self):
        result=summarize([job("gone",2)],{},workspace_id=129,project_id=367,username="me")
        self.assertEqual(result["allocated_slots"],2)
        self.assertEqual(result["unknown_project_slots"],2)
        self.assertEqual(result["unknown_user_slots"],2)
        self.assertEqual(result["outside_project_slots"],0)
        self.assertEqual(result["outside_user_slots"],0)

    def test_zero_gpu_shells_do_not_consume_slots_or_ownership_counts(self):
        jobs=[job("gpu",3,"TYPE_SHELL"),job("zero",0,"TYPE_SHELL"),job("waiting",0,"TYPE_SHELL")]
        metadata={"gpu":meta("gpu",project=None,user="other",kind="shell"),
            "zero":meta("zero",project=None,user="other",kind="shell"),
            "waiting":dict(meta("waiting",project=None,kind="shell"),state="PENDING")}
        result=summarize(jobs,metadata,workspace_id=129,project_id=367,username="me")
        self.assertEqual(result["allocated_slots"],3)
        self.assertEqual(result["running_shells"],2)
        self.assertEqual(result["zero_gpu_shells"],1)
        self.assertEqual(result["outside_project_slots"],3)
        self.assertEqual(result["outside_user_slots"],3)

    def test_task_metadata_drops_connection_keys(self):
        metadata=task_metadata([dict(jobId="shell",id="uuid",username="other",description="arc",
            state="RUNNING",privateKey="private",publicKey="public",addresses=["connection"])],"shell")
        encoded=json.dumps(metadata)
        self.assertNotIn("private",encoded)
        self.assertNotIn("public",encoded)
        self.assertNotIn("connection",encoded)
        self.assertIsNone(metadata["shell"]["project_id"])

    def test_collection_joins_authoritative_allocations_with_metadata(self):
        class FakeDet:
            def text(self,*args):
                if args[:2]==("user","whoami"):
                    return "You are logged in as user 'me'"
                raise AssertionError("Historical experiment listing should not be queried.")
            def json(self,*args,**kwargs):
                if args[:2]==("project","describe"):
                    return dict(id=367,workspaceId=129)
                if args[:2]==("job","list"):
                    return dict(jobs=[job("own",1),job("other",2),job("shell",3,"TYPE_SHELL")])
                if args[:2]==("task","list"):
                    return {"a":dict(taskId="10.uuid",resourcePool="42_Compute"),
                            "b":dict(taskId="11.uuid",resourcePool="42_Compute")}
                if args[:2]==("experiment","describe"):
                    return [dict(experiment=dict(id=10,jobId="own",name="own",username="me",projectId=367)),
                            dict(experiment=dict(id=11,jobId="other",name="other",username="other",projectId=555))]
                return [dict(id="shell-id",jobId="shell",username="other",description="arc",state="RUNNING")]
        result=collect(dict(workspace="RAI_WS",project="swarm_echo",project_id=367,resource_pool="42_Compute"),FakeDet())
        self.assertEqual(result["allocated_slots"],6)
        self.assertEqual(result["outside_project_slots"],5)
        self.assertEqual(result["outside_user_slots"],5)
        self.assertEqual(result["unknown_project_slots"],0)
        self.assertEqual(result["username"],"me")


if __name__=="__main__":
    unittest.main()
