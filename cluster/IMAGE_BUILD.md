# Step 2: GitHub Actions to GHCR

The workflow is installed at `.github/workflows/cluster-image.yml`. It will build
and publish after relevant changes reach the repository's default branch, or
when manually dispatched on that branch. Adding it locally does not publish.

## 1. Registry owner

`cluster/settings.json` is configured for this repository's lowercase owner:

```json
"image_repository": "ghcr.io/simonkistner/swarmecho-deps"
```

From PowerShell in `cluster`, inspect the expected image reference:

```powershell
.\submit.ps1 -ImageInfo
```

This prints the exact image with its dependency-derived `deps-...` tag without
accessing GHCR. The same metadata command runs in CI.

## 2. Workflow setup

The workflow is ready to push with the cluster settings, scripts, Dockerfile,
`pyproject.toml` and `uv.lock`. It uses GitHub's built-in token with
`contents: read` and `packages: write`. No additional publishing token,
Docker Hub account, cluster credentials, VPN credentials or W&B key is needed.
Package publishing must be permitted by the repository/organization settings.

Only dependency metadata is included in the Docker build context. The Dockerfile
installs the dependencies; `det` uploads current local training source separately.
No GPU is needed for the image build. The base image is large: if the first build
fails for runner disk space, address the build failure or build/push the same tag
locally rather than changing the training job.

## 3. Build and inspect

Push and merge through your normal repository workflow. Open GitHub **Actions →
Build cluster dependency image** after the merge reaches the default branch.
The merge should trigger a run because the workflow/settings/dependency files
changed. If no run starts, select **Run workflow** on the default branch. Other
branches are deliberately skipped.

Check that **Build and publish** succeeds. Compare its published image reference
in the workflow summary with local `submit.ps1 -ImageInfo`. A mismatch means
committed dependencies differ from current local dependencies; publish the
matching version before submitting.

## 4. Permit anonymous pulling

GHCR packages initially default to private. Open **SimonKistner → Packages →
swarmecho-deps → Package settings → Change visibility → Public**, and confirm.
Public repository visibility does not itself make the package public. This image
contains the base environment and dependency metadata/packages. The current local
submitter does not configure private-registry credentials.

## 5. Submit the smoke job

With VPN connected and your existing Determined login available, from `cluster`:

```powershell
.\submit.ps1 -Run g03-smoke
```

Open the experiment link and watch its logs. This is the first check that agents
can pull GHCR and run our environment. An Actions build succeeding does not prove
cluster network access or GPU compatibility. Inspect pull/runtime errors before
resubmitting. After smoke succeeds and the PFSS W&B key is ready:

```powershell
.\submit.ps1 -Run g03-full
```

Compare sustained full-run SPS after map preparation and compilation with the
equivalent local 4090 run. The short smoke is not a throughput benchmark.

References:

- [GitHub image publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)
- [GHCR authentication and visibility](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Cluster custom image guide](https://login01.ai.tu-darmstadt.de:8080/docs/model-dev-guide/prepare-container/custom-env.html#custom-images)
