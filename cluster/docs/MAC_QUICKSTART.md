# TL;DR: first connection from a Mac

You need a TU Darmstadt VPN login with 2FA, a Determined cluster account,
and access to workspace `RAI_WS`. Ask the workspace administrator for access
if your account has not been added yet. All commands below run in macOS Terminal.

## 1. Install and connect the VPN

Install **Cisco Secure Client** using the
[TU Darmstadt VPN instructions](https://www.hrz.tu-darmstadt.de/support_und_anleitungen/hrz_anleitungen/tu_vpn/index.de.jsp).
Connect to `vpn.hrz.tu-darmstadt.de` with your TU credentials and second factor.
Keep the VPN connected while using the cluster.

## 2. Install the CLI once

Install [Homebrew](https://brew.sh/) if it is missing:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the installer's **Next steps** to add Homebrew to your PATH, then run:

```bash
brew install python@3.10
"$(brew --prefix python@3.10)/bin/python3.10" -m venv "$HOME/.local/share/swarmecho-det"
source "$HOME/.local/share/swarmecho-det/bin/activate"
python -m pip install 'determined==0.35.0'
det --version
ssh -V
```

Python 3.10 matches the existing project CLI setup; the CLI lives in its own
environment. macOS includes the SSH client. A repository checkout is unnecessary
for this quickstart.

## 3. Log in

```bash
export DET_MASTER='https://login01.ai.tu-darmstadt.de:8080'
det user login YOUR_CLUSTER_USERNAME
det user whoami
```

Replace `YOUR_CLUSTER_USERNAME` with your own Determined account and enter its
password when prompted. The web UI is at
[the same cluster address](https://login01.ai.tu-darmstadt.de:8080).

## 4. SSH into a zero-GPU shell

Create the shell configuration locally (once):

```bash
mkdir -p "$HOME/.config/swarmecho"
cat > "$HOME/.config/swarmecho/pfss-shell.yaml" <<'YAML'
resources:
  resource_pool: 42_Compute
  slots: 0
bind_mounts:
  - host_path: /pfss/mlde/workspaces/mlde_wsp_RAI_WS
    container_path: /pfss/mlde/workspaces/mlde_wsp_RAI_WS
    read_only: false
YAML
```

Start the shell and connect over SSH:

```bash
det shell start -w RAI_WS --config-file "$HOME/.config/swarmecho/pfss-shell.yaml"
```

This creates a remote shell and connects interactively. Inside it:

```bash
ls /pfss/mlde/workspaces/mlde_wsp_RAI_WS
# Existing SwarmEcho run files, if your account has permission:
ls /pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/swarm_echo/outputs
exit
```

The `ar44ruca` path is the existing project's directory, not your username.
The shell mounts the shared workspace read-write and requests **zero GPU slots**.

## 5. Reconnect or clean up

On your Mac, list shells and copy the UUID of the one you created:

```bash
det shell list
SHELL_ID='PASTE_YOUR_SHELL_UUID'
det shell open "$SHELL_ID"
```

For the literal `ssh ...` command instead, run this and copy/run its output:

```bash
det shell show-ssh-command "$SHELL_ID"
```

After `exit`, release your shell from the Mac:

```bash
det shell kill "$SHELL_ID"
```

`exit` disconnects; `kill` releases the shell. Only kill your own shell.

For a new Terminal session: reconnect the VPN, then run
`source "$HOME/.local/share/swarmecho-det/bin/activate"` and
`export DET_MASTER='https://login01.ai.tu-darmstadt.de:8080'` before `det` commands.
The CLI retains your login; repeat `det user login YOUR_CLUSTER_USERNAME` if it expires.

If the cluster times out, check the VPN first. If login succeeds but `RAI_WS`
is unavailable, ask its administrator to check your workspace access.

Infrastructure and CLI version follow [settings.json](../settings.json) and the
[command register](COMMANDS.md); the shell mount matches [pfss-shell.yaml](../pfss-shell.yaml).
Installation reference: [Homebrew Python 3.10](https://formulae.brew.sh/formula/python%403.10).
These Mac instructions were prepared from the project configuration; they have
not been executed on a fresh Mac.
