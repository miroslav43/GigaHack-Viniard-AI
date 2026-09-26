# syntax=docker/dockerfile:1
# Sireț3 vineyard AI pipeline (src/AI), CPU only. Build context = src/AI.
#
# Build:  docker build -t vineyard .                    (or: make docker-build -> vineyard:arm64)
#   The default EXTRAS="route" gives the classical pipeline + route + measurements (tiles -> route.geojson
#   + measurements.csv), without torch: ~460 MB compressed, ~1.5 GB unpacked. EXTRAS="route nn waste-ml"
#   adds torch 2.14.0+cpu from the pytorch-cpu index pinned in pyproject/uv.lock. The only apt package is
#   libexpat1 (see the runtime stage).
#
# Run (organiser data read-only, results in a named volume, no network needed):
#   D="$PWD/../../data & info"
#   docker run --rm --init --network none -v "$D":/data:ro -v vineyard-work:/work vineyard doctor
#   docker run --rm --init --network none -v "$D":/data:ro -v vineyard-work:/work vineyard \
#     run --set waste.probe.enabled=false                      # perception -> AnnSet(model); add --tiles <id>
#   docker run --rm --init --network none -v "$D":/data:ro -v vineyard-work:/work \
#     -v "$PWD/marcaj":/marcaj:ro vineyard final /marcaj/<export>.zip \
#     --set paths.publish_dir=/work/publish --set web.out_dir=/work/web
#   --init reaps the --workers process pool and forwards `docker stop` (vineyard is not a PID-1 init).
#   waste.probe.enabled=false: the core image has no open_clip, so the waste stage runs its rule-only
#   verifier (the final waste layer comes from configs/waste_confirmed.csv either way).
#   The two path --set keep route.geojson/measurements.csv and the web bundle in /work; without them they go
#   to the repo-shaped paths inside the container (/opt/vineyard, /opt/vineyard/src/Web/data) and are lost.
#   Trained weights are never baked in: mount them read-only on /opt/vineyard/src/AI/models when needed.
#   On Linux with a bind-mounted /work, build with --build-arg APP_UID="$(id -u)" or run with --user.
#
# The core image downloads nothing at run time. PROJ_DATA / PROJ_LIB / GDAL_DATA are deliberately left
# unset: the rasterio and pyproj wheels ship their own GDAL/PROJ data (vineyard._env drops conda leftovers).

ARG PYTHON_IMAGE=python:3.12-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.1
ARG EXTRAS="route"

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- builder: locked venv + editable project
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /uvx /bin/

ARG EXTRAS
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12

# The project sits at a repo-shaped path so the config's project-relative paths (publish_dir "../..",
# web.out_dir "../Web/data") resolve inside the image instead of to "/".
WORKDIR /opt/vineyard/src/AI

# Third-party dependencies only: this layer is reused until uv.lock or pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    set -eu; flags=""; for extra in ${EXTRAS}; do flags="${flags} --extra ${extra}"; done; \
    uv sync --frozen --no-dev --no-install-project ${flags}

# The project itself, editable: vineyard.config locates configs/ next to the package (PROJECT_ROOT).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; flags=""; for extra in ${EXTRAS}; do flags="${flags} --extra ${extra}"; done; \
    uv sync --frozen --no-dev ${flags} \
 && python -m compileall -q -j 0 vineyard

# ---------------------------------------------------------------- runtime: venv + source, non-root
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=1000
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="vineyard-siret3" \
      org.opencontainers.image.description="Sireț3 vineyard AI: perception, CVAT export, route and measurements (CPU)" \
      org.opencontainers.image.source="https://github.com/miroslav43/GigaHack-Viniard-AI" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

# The rasterio wheel bundles GDAL/PROJ but links the system expat, which the slim image does not ship.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_UID}" vineyard \
 && useradd --uid "${APP_UID}" --gid vineyard --home-dir /home/vineyard --create-home \
            --shell /usr/sbin/nologin vineyard

COPY --from=builder /opt/vineyard/src/AI /opt/vineyard/src/AI

# /data is the read-only mount point; /work (runs, caches) and the repo-shaped output dirs are writable.
RUN mkdir -p /data /work /opt/vineyard/src/Web/data \
 && chown vineyard:vineyard /work /opt/vineyard /opt/vineyard/src/Web/data

ENV PATH="/opt/vineyard/src/AI/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VINEYARD_DATA_ROOT=/data \
    VINEYARD_WORK_DIR=/work

USER vineyard
WORKDIR /work

# Fail the build, not the first run, as the non-root user: no GDAL/PROJ override leaks in; every module
# `vineyard doctor` requires imports (all native libs present); a GeoParquet + GPKG round-trip in EPSG:32635
# works with the wheels' own GDAL/PROJ data; each installed extra imports; the CLI starts.
ARG EXTRAS
RUN set -eu; test -z "${PROJ_DATA-}${PROJ_LIB-}${GDAL_DATA-}"; \
    python -c "import sys; from vineyard import doctor as d; \
bad = [c for c in (*d.check_imports(), d.check_geo_versions(), *d.check_vector_io()) if c.required and not c.ok]; \
sys.exit('; '.join(f'{c.name}: {c.detail}' for c in bad) or None)"; \
    for extra in ${EXTRAS}; do case "${extra}" in \
      route) python -c "import ortools.constraint_solver.pywrapcp, python_tsp.heuristics, sknw" ;; \
      nn) python -c "import torch, torchvision, segmentation_models_pytorch" ;; \
      waste-ml) python -c "import transformers, open_clip, huggingface_hub" ;; \
      *) echo "no import check for extra '${extra}': add one here" >&2; exit 1 ;; \
    esac; done; \
    vineyard --version

ENTRYPOINT ["vineyard"]
CMD ["--help"]
