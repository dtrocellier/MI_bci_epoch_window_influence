FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/env/.venv
ENV PATH="/env/.venv/bin:$PATH"

WORKDIR /env

COPY pyproject.toml uv.lock .python-version ./

RUN uv python install 3.11 \
    && uv sync --frozen --no-dev

WORKDIR /app

CMD ["bash"]
