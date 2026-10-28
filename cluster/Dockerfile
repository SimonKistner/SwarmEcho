# Build from a small prepared context produced by prepare_context.py.
# Publish this image to a registry accessible to the Determined agents.
FROM determinedai/pytorch-ngc:0.35.0

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/swarmecho/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/swarmecho/python \
    UV_NO_DEV=1

WORKDIR /opt/swarmecho
COPY pyproject.toml uv.lock README.md ./
RUN uv python install 3.11 && \
    uv sync --frozen --no-dev --no-install-project --python 3.11

# Trial source comes from the Determined model context, so code edits do not
# require rebuilding this dependency image.
ENV SWARMECHO_TRAIN_PYTHON=/opt/swarmecho/.venv/bin/python
