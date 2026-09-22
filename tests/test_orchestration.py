import json

from swarmecho.training.orchestration import (
    checkpoint_total_steps,
    generate_unique_seeds,
    pop_arg,
    split_levels,
)


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
