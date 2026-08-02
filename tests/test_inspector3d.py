import jax
import jax.numpy as jnp

from swarmecho.core.config3d import load_level_3d
from swarmecho.env.baseline3d import make_baseline_3d_fns
from swarmecho.visualize.inspector3d import HTML, inspector_html, replay_payload
from swarmecho.visualize.replay3d import write_replay


def test_inspector_payload_and_controls(tmp_path):
    level = load_level_3d()
    reset, step, _, _ = make_baseline_3d_fns(level.building, level.env)
    state = reset(jax.random.PRNGKey(0))
    states = [state, step(state, jnp.zeros((level.env.num_agents, 3)))]
    _, manifest = write_replay(
        tmp_path / "inspect",
        states,
        map_name=level.building_name,
        dt=level.env.dt,
        metadata={
            "world_size_m": level.building.world_size_m.tolist(),
            "cell_size_m": level.building.cell_size_m,
            "comm_radius_m": level.env.comm_radius,
        },
    )
    payload = replay_payload(manifest)
    assert payload["position"][0][0][2] == level.building.base_position_m[2]
    assert payload["manifest"]["world_size_m"] == [20.0, 20.0, 20.0]
    assert 'id="timeline"' in HTML
    assert 'id="showCoverage"' in HTML
    assert "Plotly.react" in HTML
    assert '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>' not in inspector_html()
