# Export and publish selected inspector results

The public inspector serves one completed replay or heatmap per page. Its 3D
view, camera, building visibility, playback, reward plots and heatmap filters
use the same rendering code as the local inspector. It has no run discovery,
artifact navigation, refresh controls, stars or evaluation commands. Clicking
a heatmap point shows its coordinates and outcome category.

Python only runs during export. The published site loads static files and
does not need a Python service, GPU, database or visitor installation.

## Export from the inspector

Run the inspector in the existing WSL environment, from the repository root:

```bash
uv run swarmecho-inspect root=outputs
```

1. Select the exact replay or heatmap to share.
2. Click **Export public page** in the top right.
3. **Page name and title** is prefilled from the selected run name and training
   steps, with URL-safe lowercase letters, numbers and hyphens. For example,
   `solo_B02c_[186M]_[Replay]` becomes `solo-b02c-186m`. Optionally enter an
   **Additional artifact name**, such as `deadend-chain`, to append it to both
   the page title and URL name. The combined name is limited to 80 characters;
   long run names are shortened while retaining the steps and suffix.
   Select an existing **Section**, or select **Create a
   new section…** and enter its name. Add optional comma-separated **Tags**.
   The dialog captures a **preview of your current 3D view**, preserving its
   viewing angle and visible layers. A separate offscreen view fits the building,
   then removes empty borders and scales it into the thumbnail with a small
   margin. Your live zoom/pan remains unchanged. Close the dialog and adjust the
   angle before reopening it if you want a different preview, or uncheck the
   preview option to omit the image.
4. For a replay, leave **Precompute reference path** checked if wanted.
   Export runs in the inspector's background worker. Playback stays available
   and no visitor will trigger path computation.
5. Click **Export**. When it finishes, click **Preview exported page** to open
   the static result through the local inspector.

Exporting writes local files only. It does not commit, push or deploy them.
Existing page names are refused unless **Replace an existing page with this
name** is checked. Replacement is limited to recognized public inspector
exports. If a reference route cannot be computed, the artifact is still
exported, the local dialog reports the reason, and the public path control is
disabled. Without a reference path, replay viewing still works normally.

## Section galleries and publication order

The landing page displays one thumbnail gallery per section, with horizontal
lines between sections. Cards show titles, publication dates, custom tags,
and an automatic **Success** or **Fail** badge for replays. A replay is marked
successful if any recorded frame achieves success. Heatmaps show their mean
chain success rate instead of a binary result. Search and tag/artifact filters
work entirely in the browser. Each public inspector also links back to the
gallery through **All results**.

Cards are sorted by their first publication date, newest first. Sections are
sorted by the most recent publication among their cards, not by when the
section was created. For example, a January 3 export into a section created
on January 1 places that section above a section whose latest export was
January 2. Filters retain this section ordering.

Publication dates are recorded in UTC on the first local export and displayed
in the visitor's local date format; actual GitHub deployment happens after
you push. Replacing an export preserves its first publication date and records
a separate updated date. Publish under a new page name for a new dated entry.
The export dialog lets you change a replacement page's section and tags without
changing its URL. Section names differing only by capitalization are combined.
Empty sections do not appear until they contain a published export.

Older pages without gallery metadata appear in **General**. After the next
export rebuilds the catalogue, their export metadata file's modification date
is used as a publication-date fallback. Missing preview images have a building
placeholder. Re-export older pages to add sections, tags and thumbnails.

## Checkpoint configuration

Each exported replay or heatmap has a separate **Checkpoint configuration**
panel showing:

- Training maps: for example, **2,500 randomly generated maps**, **1 static map**,
  or **5 static maps**.
- Agents during training, read from the producing run's saved `config.yaml`.
- Agents in this evaluation, read from the replay's recorded agent count or
  the heatmap's evaluation metadata.

These fields are automatic and do not require custom tags. Export uses the
checkpoint's run folder, or the artifact's enclosing run when the checkpoint
path is unavailable. Random map counts use the training selection in
`random_buildings.json`, excluding evaluation maps and unused pool maps.
Static map counts use `static_maps.json`, excluding maps assigned zero training
lanes. When those records are absent, the saved run configuration supplies the
persistent random map count (`training.num_envs`) or the static map selection.
The current project level files are not used to reconstruct training settings.

The panel describes the run/stage that produced the checkpoint; it does not
sum maps or agents from earlier curriculum stages. Unavailable metadata shows
**Unknown**. Only this small summary is included in the export, without local
paths or the full training configuration. Re-export existing public pages to
add the panel.

The private inspector displays the same configuration. For replays, the left
sidebar places **Episode** at the top left, with **Checkpoint configuration**
above **Building visibility** in the adjacent column. Reward debug plots start
disabled in both versions; enable them under **Playback**. The reward lines,
legend and axis scaling include only agent slots active in at least one
recorded frame of the replay, including agents activated later or subsequently
decommissioned. Drone labels and colors retain their original slot numbers.
Public replay pages
use a compact header with a full-width frame timeline and an **All results**
link; the orbit/zoom hint sits inside the 3D view.
New artifacts open at 10° camera elevation with two 15% zoom-in steps applied
to the starting distance. This deterministic zoom avoids browser-dependent
mouse-wheel timing. Selecting an artifact from the run picker resets to that
pose. The step buttons and arrow navigation retain the current camera angle,
zoom and pan while loading the selected heatmap or replay; map and category
navigation also retain the camera. Refreshing the same selected artifact
retains its camera.

## Export from the command line

Use the replay's JSON manifest, not its NPZ file or reference-path cache:

```bash
uv run swarmecho-inspect export=outputs/RUN/artifacts/eval/replays/REPLAY.json name=b02c-nine-agents "title=B02c with nine agents"
```

For a heatmap, use its evaluation CSV. Its metadata and layout sidecars are
resolved automatically, including checkpoint and map-suite artifact layouts:

```bash
uv run swarmecho-inspect export=outputs/RUN/artifacts/eval/HEATMAP_info.csv name=b02c-heatmap "title=B02c evaluation heatmap"
```

These are placeholder paths; use an existing completed artifact. Options:

| Option | Behavior |
| --- | --- |
| `name=...` | Required URL name, maximum 80 characters. |
| `title=...` | Display title, maximum 160 characters; defaults to the page name. |
| `reference_path=true` | Default for replays; reuse a valid cached route or compute locally. Ignored for heatmaps. |
| `reference_path=false` | Skip reference-path calculation. |
| `overwrite=true` | Replace a recognized export with the same name. Default is false. |
| `output=...` | Optional alternate site root; the deployment workflow publishes only the default `public-inspector/`. |
| `section=...` | Section name; creates the section if needed. Defaults to General for new pages, or preserves the section when replacing a page. |
| `tags=...` | Comma-separated custom tags, up to 12 tags of 32 characters each. Omission preserves tags when replacing a page; `tags=` clears them. |
| `thumbnail=...` | Optional local PNG file (maximum 1 MiB). Command-line exports without an image show a placeholder. |

For example:

```bash
uv run swarmecho-inspect export=outputs/RUN/artifacts/eval/replays/REPLAY.json name=b02c-nine-agents "section=Swarm size experiments" "tags=nine agents,static maps" thumbnail=preview.png
```

UI exports capture their PNG automatically using the existing Plotly viewer;
command-line exports do not launch a browser to generate images. Omitting a
preview when replacing a page removes its previous preview.

Command-line export exits when finished and does not launch an inspector server.
To preview a command-line export, start the normal inspector and visit
`http://127.0.0.1:8765/public-inspector/replays/b02c-nine-agents/`.
Alternatively serve the folder in WSL:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1 --directory public-inspector
```

Then visit `http://localhost:8000/`. Use HTTP for previews: opening an
`index.html` directly as a `file://` URL prevents browsers from fetching its data.

## Exported files

```text
public-inspector/
  index.html                    # Landing page listing selected exports
  catalogue.json                # Section, dates, tags and card metadata
  assets/plotly.min.js           # Shared, locally bundled Plotly runtime
  replays/b02c-nine-agents/
    index.html                  # Reduced viewer for this replay
    data.json                   # Index of compressed data chunks
    data-0000.gz                # Recorded arrays and prepared geometry
    reference-path.json         # Precomputed route or unavailable status
    export.json                 # Public title, kind and relative URL
    thumbnail.png               # Optional 640 × 400 preview from the inspector
  heatmaps/b02c-heatmap/
    ...
```

The exporter prepares JSON from the same payload functions as the local
inspector, compresses it, and splits it into files no larger than 8 MiB.
Visitors need a current browser with WebGL and `DecompressionStream` support.
Plotly is bundled during export, so viewing does not depend on an external
JavaScript CDN. All asset URLs are relative, including when hosted beneath
GitHub's `/SwarmEcho/` repository prefix.

The original `outputs/` folder remains ignored. Checkpoints, local filesystem
paths, run discovery indices and raw configuration snapshots are not copied.
Only selected artifact data, visualization metadata, building geometry, and
the optional route are exported. Public Pages exports are downloadable;
select the results you intend to share publicly.

## Enable GitHub Pages once

The repository already includes `.github/workflows/public-inspector-pages.yml`.
It uploads only `public-inspector/`, runs no Python, and deploys only from the
repository's default branch. Pushes touching that folder or the workflow
trigger deployment; it can also be started manually.

1. Commit and push the implementation, workflow, and any selected exports to
   the repository's default branch using your normal Git workflow.
2. Open the GitHub repository's **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **GitHub Actions**.
   No Jekyll setup or additional generated workflow is needed.
4. Open **Actions → Publish public inspector → Run workflow**, choose the
   default branch, and run it. This also recovers an initial run that happened
   before Pages was enabled.
5. Wait for the deploy job to succeed. **Settings → Pages** and the workflow's
   `github-pages` deployment both provide the website URL.

For the repository linked in the README, with no custom domain, the site is:

```text
https://m4gi3r.github.io/SwarmEcho/
https://m4gi3r.github.io/SwarmEcho/replays/b02c-nine-agents/
https://m4gi3r.github.io/SwarmEcho/heatmaps/b02c-heatmap/
```

Subsequent exports are published by committing and pushing their files,
`catalogue.json`, and any updated shared assets. Existing exports are snapshots:
after changing the inspector's renderer, re-export pages you want to update.
Changing only Python source does not automatically rebuild exported pages.
Keep the total published site within GitHub Pages' 1 GB limit.

To remove a result, remove its named export folder and its entry in
`catalogue.json`, then commit and push. The next export also rebuilds the
catalogue from remaining `export.json` markers.

If deployment fails, check the failed job in **Actions**. Confirm that Pages
uses **GitHub Actions**, that Actions are enabled for the repository, and
that the run is on the default branch. If a browser shows an old export after
deployment, reload with its cache disabled.

GitHub documentation: [publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site),
[custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
and [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
