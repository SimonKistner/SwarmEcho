"""Sigmoid/BCE vector readouts with episode -> agent -> snapshot sampling.

Sampling with the product of the three uniform choices is equivalent to explicit
nested draws. Validation uses those same weights, not an unbalanced row average.
An epoch has a configured number of optimizer updates, not one pass over rows.
"""
import time

import numpy as np

from geo_experiments.shared.artifacts import HISTORY_COLUMNS
from geo_experiments.shared.console import terminal_print
from geo_experiments.shared.sampling import hierarchical_weights, mask_scores, HierarchicalSampler


def train_mask(source, seed, rows, masks, splits, config):
    import jax
    import jax.numpy as jnp
    import optax

    settings = config["probe"]
    threshold = float(config["report"]["threshold"])
    indices = {name: np.flatnonzero(np.isin(rows["episode"], episodes)) for name, episodes in splits.items()}
    missing = [name for name, ix in indices.items() if not len(ix)]
    if missing:
        return {"source": source, "seed": seed, "status": "unavailable", "reason": "No rows in " + ", ".join(missing)}, {}
    train = indices["train"]
    weights = {name: hierarchical_weights(rows["episode"][ix], rows["agent"][ix]) for name, ix in indices.items()}
    features = np.asarray(rows["features"], np.float32)
    mean = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(train), 4096):
        ix = train[start:start + 4096]
        mean += np.sum(features[ix] * weights["train"][start:start + len(ix), None], axis=0)
    mean = mean.astype(np.float32)
    variance = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(train), 4096):
        ix = train[start:start + 4096]
        variance += np.sum((features[ix] - mean) ** 2 * weights["train"][start:start + len(ix), None], axis=0)
    std = np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)
    if any(not np.isfinite(features[start:start + 4096]).all() for start in range(0, len(features), 4096)):
        raise ValueError(f"Non-finite {source} features.")
    def x(ix):
        return (features[ix] - mean) / std
    def y(ix):
        return masks[rows["episode"][ix]].astype(np.float32)
    widths = [features.shape[1], *settings["hidden_sizes"], masks.shape[1]]
    keys = jax.random.split(jax.random.PRNGKey(seed), len(widths) - 1)
    params = tuple(dict(weight=jax.random.normal(key, (a, b)) * np.sqrt(2. / a), bias=jnp.zeros(b))
                   for key, a, b in zip(keys, widths[:-1], widths[1:], strict=True))

    def forward(p, features):
        for layer in p[:-1]:
            features = jax.nn.relu(features @ layer["weight"] + layer["bias"])
        return features @ p[-1]["weight"] + p[-1]["bias"]

    optimizer = optax.adamw(settings["learning_rate"], weight_decay=settings["weight_decay"])
    state = optimizer.init(params)

    @jax.jit
    def update(p, state, features, targets):
        loss, grads = jax.value_and_grad(lambda w: optax.sigmoid_binary_cross_entropy(forward(w, features), targets).mean())(p)
        changes, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, changes), state, loss

    @jax.jit
    def evaluate(p, features, targets):
        logits = forward(p, features)
        return jax.nn.sigmoid(logits), optax.sigmoid_binary_cross_entropy(logits, targets).mean(axis=-1)

    def metrics(p, split):
        ix = indices[split]
        loss = exact = similarity = 0.
        for offset in range(0, len(ix), settings["batch_size"]):
            chunk = ix[offset:offset + settings["batch_size"]]
            targets = y(chunk)
            probability, costs = evaluate(p, jnp.asarray(x(chunk)), jnp.asarray(targets))
            w = weights[split][offset:offset + len(chunk)]
            loss += float(np.sum(np.asarray(costs) * w))
            correct, overlap = mask_scores(np.asarray(probability), targets, threshold)
            exact += float(np.sum(correct * w))
            similarity += float(np.sum(overlap * w))
        return loss, exact, similarity

    rng = np.random.default_rng(seed)
    sampler = HierarchicalSampler(rows["episode"][train], rows["agent"][train], rng)
    best, best_epoch, best_loss, stale, history = None, 0, np.inf, 0, []
    best_scores = None
    started = time.monotonic()
    for epoch in range(1, settings["epochs"] + 1):
        train_losses = []
        for _ in range(settings["updates_per_epoch"]):
            selected = train[sampler.draw(settings["batch_size"])]
            params, state, loss = update(params, state, jnp.asarray(x(selected)), jnp.asarray(y(selected)))
            train_losses.append(float(loss))
        # Training accuracy is measured on a reproducible balanced draw; validation
        # loss is the full episode/agent-balanced validation mean for early stop.
        check = train[sampler.draw(settings["batch_size"])]
        probability, _ = evaluate(params, jnp.asarray(x(check)), jnp.asarray(y(check)))
        train_accuracy = float(mask_scores(np.asarray(probability), y(check), threshold)[0].mean())
        validation_loss, validation_accuracy, validation_similarity = metrics(params, "validation")
        history.append([epoch, np.mean(train_losses), train_accuracy, validation_loss, validation_accuracy])
        if validation_loss < best_loss - settings["min_delta"]:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best = jax.tree.map(lambda value: np.array(value, copy=True), params)
            best_scores = validation_accuracy, validation_similarity
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == settings["epochs"] or stale >= settings["patience"]:
            terminal_print(f"[probe] path/{source}/seed={seed}: epoch {epoch}; "
                           f"validation exact-path accuracy={100 * validation_accuracy:.1f}%; "
                           f"path similarity={100 * validation_similarity:.1f}%", flush=True)
        if stale >= settings["patience"]:
            break
    if best is None:
        raise RuntimeError("No finite validation loss for mask readout.")
    terminal_print(f"[probe] path/{source}/seed={seed}: selected epoch {best_epoch}; "
                   f"validation exact-path accuracy={100 * best_scores[0]:.1f}%; "
                   f"path similarity={100 * best_scores[1]:.1f}%", flush=True)
    test = indices["test"]
    probabilities = np.concatenate([np.asarray(evaluate(best, jnp.asarray(x(ix)), jnp.asarray(y(ix)))[0])
        for ix in (test[i:i + settings["batch_size"]] for i in range(0, len(test), settings["batch_size"]))])
    # The configured threshold defaults to .5 and is frozen before fitting;
    # held-out labels never tune it. The example slider is purely local display.
    saved = dict(probabilities=probabilities, predicted=probabilities >= threshold, row_indices=test.astype(np.int32),
                 threshold=np.array(threshold), history=np.array(history), best_epoch=np.array(best_epoch),
                 **{"model/feature_mean": mean, "model/feature_std": std})
    for i, layer in enumerate(best):
        for name, value in layer.items():
            saved[f"model/{name}_{i}"] = value
    metadata = dict(task="path", source=source, seed=seed, status="complete", best_epoch=best_epoch,
                    best_validation_loss=best_loss, epochs_trained=len(history), training_seconds=time.monotonic() - started,
                    best_validation_exact_path_accuracy=best_scores[0], best_validation_path_similarity=best_scores[1],
                    hidden_sizes=settings["hidden_sizes"], input_dim=features.shape[1], outputs=masks.shape[1],
                    rows={s: len(ix) for s, ix in indices.items()}, updates_per_epoch=settings["updates_per_epoch"],
                    sampler="uniform episode -> agent -> snapshot", history_columns=HISTORY_COLUMNS)
    return metadata, saved


def fit_masks(config, shared, run_dir, *, shared_reference=True):
    from geo_experiments.shared.artifacts import add_group, group, save_data, update_run
    output, jobs = ({} if shared_reference else dict(shared)), []
    for source in config["definition"]["inputs"]:
        rows = group(shared, f"inputs/{source}")
        for seed in config["probe"]["seeds"]:
            metadata, fitted = train_mask(source, seed, rows, shared["labels/path"], group(shared, "splits"), config)
            if fitted:
                add_group(output, f"probes/path/{source}/seed_{seed}", fitted)
            jobs.append(metadata)
            save_data(run_dir, output)
            update_run(run_dir, jobs=jobs, progress={"stage": "probes", "completed": len(jobs),
                "total": len(config["definition"]["inputs"]) * len(config["probe"]["seeds"])})
    return output, jobs
