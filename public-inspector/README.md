# Public inspector

This folder is the GitHub Pages publishing root. Only selected exports belong
here; the original `outputs/` folder remains ignored.

Start `uv run swarmecho-inspect root=outputs`, select a replay or heatmap, and
choose **Export public page**. The exported page can be previewed immediately
through the local inspector. An export does not publish anything by itself.

During export, select or create a section, add comma-separated tags, and include
an optional preview of the current 3D view. The landing page groups thumbnail
cards into sections separated by horizontal lines. Both cards and sections are
ordered by newest publication; a section's order uses its most recent card.
Cards show dates, automatic replay Success/Fail badges or heatmap success
rates, and custom tags. Search and filters run in the browser.

Each result page also shows automatic checkpoint configuration: the number
and type of training maps, agents during training, and agents in that evaluation.
These values come from the saved run and artifact metadata; unavailable values
show as Unknown. Re-export existing pages to include this panel.

Pages are written to `replays/<name>/` or `heatmaps/<name>/`. The export prepares
compressed JSON data, building geometry and an optional reference path, and
uses the same renderer as the local inspector. A shared Plotly runtime is
written to `assets/plotly.min.js`. No Python runs on the public website.

See [the export and deployment guide](../docs/02_guide/04_public_inspector.md)
for command-line exports and the one-time GitHub Pages setup.
