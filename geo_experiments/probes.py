"""Small supervised MLPs over saved features; no connection to policy gradients."""
from __future__ import annotations

import time

import numpy as np

from geo_experiments.artifacts import HISTORY_COLUMNS
from geo_experiments.console import terminal_print as print


def train_probe(task, source, seed, dataset, labels, splits, graph, config, *, class_values=None):
    import jax
    import jax.numpy as jnp
    import optax

    settings = config["probe"]
    features = np.asarray(dataset[source], dtype=np.float32)
    mean = features[splits["train"]].mean(axis=0)
    std = np.maximum(features[splits["train"]].std(axis=0), 1e-6)
    features = (features - mean) / std
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite {source} features.")
    y = labels[task]
    cell_task = task in ("target_cell", "own_cell")
    if class_values is None:
        class_values = labels["cell_offsets"] if cell_task else labels["storey_offsets"]
    classes = len(class_values)
    if any(not len(splits[key]) for key in ("train", "validation", "test")):
        raise ValueError("Every fitted decoder needs nonempty training, validation and evaluation splits.")
    widths = [features.shape[1], *settings["hidden_sizes"], classes]
    keys = jax.random.split(jax.random.PRNGKey(seed), len(widths) - 1)
    params = tuple({"weight": jax.random.normal(key, (left, right)) * np.sqrt(2.0 / left),
                    "bias": jnp.zeros(right)}
                   for key, left, right in zip(keys, widths[:-1], widths[1:], strict=True))

    def forward(weights, x):
        for layer in weights[:-1]:
            x = jax.nn.relu(x @ layer["weight"] + layer["bias"])
        return x @ weights[-1]["weight"] + weights[-1]["bias"]

    optimizer = optax.adamw(settings["learning_rate"], weight_decay=settings["weight_decay"])
    opt_state = optimizer.init(params)

    @jax.jit
    def update(weights, state, x, targets):
        def objective(w):
            return optax.softmax_cross_entropy_with_integer_labels(forward(w, x), targets).mean()
        loss, grads = jax.value_and_grad(objective)(weights)
        updates, state = optimizer.update(grads, state, weights)
        return optax.apply_updates(weights, updates), state, loss

    @jax.jit
    def predict(weights, x):
        return jax.nn.softmax(forward(weights, x), axis=-1)

    def predict_indices(weights, indices):
        return np.concatenate([np.asarray(predict(weights, jnp.asarray(features[chunk])))
                               for chunk in _batches(indices, settings["batch_size"])])

    rng = np.random.default_rng(seed)
    best_loss, best_epoch, stale = np.inf, 0, 0
    best_params, history = None, []
    start = time.monotonic()
    for epoch in range(1, settings["epochs"] + 1):
        for indices in _batches(rng.permutation(splits["train"]), settings["batch_size"]):
            params, opt_state, _ = update(params, opt_state, jnp.asarray(features[indices]), jnp.asarray(y[indices]))
        row = {"epoch": epoch}
        for split in ("train", "validation"):
            indices = splits[split]
            probabilities = predict_indices(params, indices)
            row[f"{split}_loss"] = float(-np.log(np.maximum(probabilities[np.arange(len(indices)), y[indices]], 1e-12)).mean())
            row[f"{split}_accuracy"] = float((probabilities.argmax(axis=1) == y[indices]).mean())
        history.append(row)
        if row["validation_loss"] < best_loss - settings["min_delta"]:
            best_loss, best_epoch, stale = row["validation_loss"], epoch, 0
            best_params = jax.tree.map(lambda x: np.array(x, copy=True), params)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= settings["patience"]:
            print(f"[probe] {task}/{source}/seed={seed}: epoch {epoch}, validation accuracy "
                  f"{row['validation_accuracy']:.1%}, loss {row['validation_loss']:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    if best_params is None:
        raise RuntimeError("No finite validation loss; inspect feature data and probe settings.")
    weights = {"feature_mean": mean, "feature_std": std,
               "class_values": class_values}
    for i, layer in enumerate(best_params):
        weights[f"weight_{i}"] = layer["weight"]
        weights[f"bias_{i}"] = layer["bias"]
    test = splits["test"]
    probabilities = predict_indices(best_params, test)
    saved = {"predicted": probabilities.argmax(axis=1), "probabilities": probabilities,
             "best_epoch": np.asarray(best_epoch, dtype=np.int32),
             "history": np.asarray([[row[key] for key in HISTORY_COLUMNS] for row in history])}
    saved.update({f"model/{key}": value for key, value in weights.items()})
    metadata = {"task": task, "source": source, "seed": seed, "best_epoch": best_epoch,
                "best_validation_loss": best_loss, "epochs_trained": len(history),
                "training_seconds": time.monotonic() - start, "hidden_sizes": settings["hidden_sizes"],
                "input_dim": features.shape[1], "num_classes": classes}
    return metadata, saved


def _batches(indices, size):
    for offset in range(0, len(indices), size):
        yield indices[offset:offset + size]


