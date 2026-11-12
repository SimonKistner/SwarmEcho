"""Install only lifecycle changes into an otherwise frozen training snapshot."""
from pathlib import Path
import shutil

from prepare_context import ROOT


def patch_training(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if "cluster_checkpoint = checkpoint_on_preempt(" in text:
        return
    patches = (
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n\nfrom swarmecho.training.cluster_preemption import checkpoint_metrics, checkpoint_on_preempt\n"),
        ("    def save_current_checkpoint(*args,**kwargs):\n",
         "    def save_current_checkpoint(*args,**kwargs):\n        kwargs[\"metrics\"] = checkpoint_metrics(kwargs.get(\"metrics\"), training_started)\n"),
        ("        validation_stop = False\n",
         "        cluster_checkpoint = checkpoint_on_preempt(\n"
         "            save_current_checkpoint, model, checkpoint_dir, run_name, update, training,\n"
         "            resume, steps_done, adaptive_spawn, target_curriculum, adaptive_maps,\n"
         "            validation_runner, destination, latest_stats, wandb_run)\n"
         "        if cluster_checkpoint is not None:\n"
         "            return cluster_checkpoint, dict(latest_stats, stop_reason=\"preempted\")\n"
         "        validation_stop = False\n"),
    )
    for old, new in patches:
        if text.count(old) != 1:
            raise ValueError("Frozen training snapshot does not match the lifecycle patch: " + str(path))
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def install(context):
    context = Path(context)
    patch_training(context / "src/swarmecho/training/train.py")
    for relative in ("src/swarmecho/training/cluster_preemption.py", "cluster/tools/run_driver.py"):
        shutil.copy2(ROOT / relative, context / relative)
