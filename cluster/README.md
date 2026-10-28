# Cluster submission

Run from PowerShell in `Q:\_0_Projects\000_SwarmEcho\SwarmEcho\cluster`.
`submit.ps1` uses standard-library Python locally (3.7+), without importing
training code or using the project `.venv`. It selects the first system `python`
on PATH; override it with `-Python 'C:\path\to\python.exe'` if needed.

## Settings and runs (step 1)

`settings.json` holds the master, CLI version, workspace/project labels, project
ID 367, `42_Compute`, one slot, PFSS mount and persistent directory. The project
ID selects the destination; workspace/project names are descriptive labels.
If moving projects, update the ID as well as the labels.

- `runs/g03-smoke.json`: small generated-map G03 run, W&B disabled.
- `runs/g03-full.json`: current full G03 level, no training overrides, W&B online.

To add a run, copy a JSON file and change its level/overrides. Unspecified values
come from the level source YAML. The wrapper owns `level`, `logging.run_name`,
`logging.log_dir`, and `logging.wandb_mode`; do not also put them in overrides.
Don't put credentials in configs. Each submission receives a unique run name.

## Preparation and submission (step 3)

```powershell
.\submit.ps1 -Run g03-smoke -DryRun
.\submit.ps1 -Run g03-full -DryRun
```

Each command prepares an ignored `.submissions/<unique-run-name>/` directory:

- `summary.json`: settings, resolved image and source size.
- `experiment.yaml`: generated Determined config (JSON syntax is valid YAML).
- `model/`: exact source snapshot to upload.
- `model/cluster/run.json`: level, overrides, W&B mode and unique run name.

Dry runs make no `det` calls, use no network and allocate no GPU. They validate
basic structure, paths, level existence, reserved/duplicate overrides, and package
size. They do not validate training override semantics, remote storage, secrets,
image availability or runtime compatibility. The first smoke job checks those.
An unset image repository is allowed for inspection but blocks real submission.

After publishing the matching image in step 2:

```powershell
.\submit.ps1 -Run g03-smoke
# After smoke succeeds and the W&B key is ready:
.\submit.ps1 -Run g03-full
```

Real submission prepares current source again, checks the installed `det` version,
sets `DET_MASTER` for its child process, and submits to project 367 using the
existing CLI login. The wrapper prints the experiment ID/URL and returns after
submission. It saves `submission.log` and `result.json` alongside the snapshot.
If the connection fails, inspect the WebUI before trying again: the job may have
been created. There is no automatic submission retry. `max_restarts: 0` prevents
Determined restarts; SwarmEcho's internal OOM recovery remains active.

If execution policy blocks the script, use a process-scoped invocation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\submit.ps1 -Run g03-smoke -DryRun
```

The old standalone G03 YAML files and hardcoded driver are replaced by named
run configs and generated experiment files. `prepare_context.py` remains available
for manual packaging but is called automatically by submission.

## Validate steps 1 and 3

1. Inspect `settings.json`: project 367, `RAI_WS / swarm_echo`, `42_Compute`, one
   slot, and `/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo`.
2. Run both dry runs above. Expect success and a lowercase GHCR image reference.
   A dry run does not check that this image has been published yet.
3. Inspect each printed `experiment.yaml`: pool, slots, image, mount and entrypoint.
4. Inspect each `model/cluster/run.json`: smoke has the small overrides and
   disabled W&B; full has no overrides and online W&B. Names must differ.
5. Inspect the snapshot's G03 YAML for current edits. The snapshot excludes `.env`,
   `.venv`, existing outputs, W&B data, map pools and the rest of the repository.

This completes local preparation validation without submitting a job.

## Dependency image (step 2)

Follow [IMAGE_BUILD.md](IMAGE_BUILD.md). `submit.ps1 -ImageInfo` prints the exact
repository, dependency tag and fingerprint expected by submission. Windows CRLF
and Linux LF produce the same fingerprint. Inputs are `cluster/Dockerfile`,
`pyproject.toml`, and `uv.lock`. Changing them requires publishing the new image.
Source and level changes alone require only a new submission.

The image contains dependencies; `det` uploads the current local training source.
Source edits do not need a GitHub push for training. Dependency image builds use
committed files, so push dependency changes through your normal workflow before
building. The image workflow is in `.github/workflows/cluster-image.yml`; it runs
after relevant changes reach the default branch, or on manual dispatch there.
Adding the workflow locally does not itself build or publish an image.

## Persistence and W&B

The agreed PFSS `swarm_echo` directory already exists. The driver uses `outputs/`
and `map_pools/` beneath it. Full G03 generates a compatible map bank if needed.
Map preparation and compilation can delay the first useful SPS reading.

For online W&B, supply `WANDB_API_KEY` inside the trial or an owner-only
`.wandb_api_key` file in that PFSS directory. Local environment secrets are not
uploaded. In a cluster shell, create the file without echoing the key or storing
it in command history:

```bash
PROJECT_DIR=/pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo
(umask 077; read -r -s -p 'W&B API key: ' wb_key; printf '\n'; printf '%s\n' "$wb_key" > "$PROJECT_DIR/.wandb_api_key")
chmod 600 "$PROJECT_DIR/.wandb_api_key"
stat -c '%a %n' "$PROJECT_DIR/.wandb_api_key"
```

Expect `600`; do not print the file. W&B project/entity can be set through normal
run overrides. Detailed metrics stay in W&B; Determined receives completion only.

## Zero-GPU shell

The existing `pfss-shell.yaml` mounts the workspace PFSS directory:

```powershell
$env:DET_MASTER = 'https://login01.ai.tu-darmstadt.de:8080'
det shell start -w RAI_WS --config-file .\pfss-shell.yaml
```

If Windows rejects the temporary SSH key, reconnect to the same shell with
`det shell open YOUR_SHELL_ID --show-ssh-command`. Use `exit` inside the shell to
disconnect, then `det shell kill YOUR_SHELL_ID` in PowerShell to release it.
The WebUI project does not create a PFSS directory or select shell paths.

Current G03 enables adaptive map and target sampling; its older G03 guide is
stale on those switches. Full submission uses the current source YAML.
