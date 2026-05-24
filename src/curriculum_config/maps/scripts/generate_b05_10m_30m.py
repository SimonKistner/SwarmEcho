"""
generate_b05_10m_30m.py
=======================
Generates the specific map: square_B05_10m_comm_30m_base
and creates a level yaml copied from B05b but utilizing this map.
"""

from pathlib import Path
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT   = _SCRIPT_DIR.parent.parent
_LEVELS_DIR = _SRC_ROOT / "curriculum_config" / "levels"
_MAPS_DIR   = _SRC_ROOT / "curriculum_config" / "maps"

n_agents_for_map = 4
comm_r = 10.0
comm_r_base = 20.0

# Calculate map dimensions using max agents = 5
min_radius = max(0.0, comm_r_base + (comm_r * (n_agents_for_map - 1)) - 10.0)
max_radius = comm_r_base + (comm_r * (n_agents_for_map - 1)) - float(n_agents_for_map)
side   = 2.0 * (max_radius + 10.0)
center = side / 2.0

map_name = "square_B04_10m_comm_20m_base"

def main():
    map_path = _MAPS_DIR / f"{map_name}.yaml"

    half_cell = 0.5
    map_data = {
        "name": map_name,
        "width":  round(side, 2),
        "height": round(side, 2),
        "spawn_zones": {
            "base": [
                round(center - half_cell, 2), round(center - half_cell, 2),
                round(center + half_cell, 2), round(center + half_cell, 2),
            ],
            "target": [0.0, 0.0, round(side, 2), round(side, 2)],
            "drone": [
                round(center - half_cell, 2), round(center - half_cell, 2),
                round(center + half_cell, 2), round(center + half_cell, 2),
            ],
        },
        "rooms":    [],
        "hallways": [],
        "walls":    [],
    }

    _MAPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w", encoding="utf-8") as f:
        yaml.dump(map_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated map: {map_path.name}")
    print(f"  Radius range: {min_radius} -> {max_radius}")
    print(f"  Side length: {side}")


def mod_level(): 
    # Copy and modify the B05b level yaml
    source_level = _LEVELS_DIR / "B05b_agents_in_square_40m_comm_40m_base.yaml"
    target_level = _LEVELS_DIR / "B05b_agents_in_square_10m_comm_30m_base.yaml"
    
    with open(source_level, "r", encoding="utf-8") as f:
        level_content = f.read()

    # Modify the content:
    level_content = level_content.replace(
        'map_names: ["square_B05_40m_comm_40m_base"]', 
        f'map_names: ["{map_name}"]'
    )
    level_content = level_content.replace(
        'target_spawn_radius: 195.0', 
        f'target_spawn_radius: {round(max_radius, 2)}'
    )
    level_content = level_content.replace(
        'target_spawn_radius_min: 70.0', 
        f'target_spawn_radius_min: {round(min_radius, 2)}'
    )

    with open(target_level, "w", encoding="utf-8") as f:
        f.write(level_content)
    
    print(f"Generated level: {target_level.name}")

if __name__ == "__main__":
    main()
