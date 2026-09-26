# Sireț3 vineyard AI — run from src/AI. Paths containing spaces/& are resolved by the config, never by make.
SHELL := /bin/bash
.DEFAULT_GOAL := help

unexport PROJ_DATA PROJ_LIB GDAL_DATA CONDA_PREFIX
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export GDAL_NUM_THREADS := 1
export GDAL_CACHEMAX := 256
export PYTORCH_ENABLE_MPS_FALLBACK := 1
export UV_PYTHON_PREFERENCE := only-managed

UV ?= uv
RUN := $(UV) run --no-sync
WORKERS ?= 8
ANNSET ?= LATEST_MODEL

.PHONY: help setup lock doctor test test-all cov lint ingest prep preannotate reference eval test-zip export \
        validate-zips final post publish web bench nn-smoke nn-train docker-build docker-push clean-work

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## install the locked environment (all extras)
	$(UV) sync --locked --all-extras

lock: ## re-resolve and export requirements.lock
	$(UV) lock
	$(UV) export --format requirements-txt --no-hashes --all-extras --no-dev -o requirements.lock

doctor: ## environment checks
	$(RUN) vineyard doctor

test: ## fast tests
	$(RUN) pytest -m "not slow"

test-all: ## all tests; tiles required
	VINEYARD_REQUIRE_TILES=1 $(RUN) pytest

cov: ## coverage report
	$(RUN) pytest --cov=vineyard --cov-report=term-missing

lint: ## ruff
	$(RUN) ruff check vineyard tests

ingest: ## verify + copy the 311 tiles, build tile_index and route inputs
	$(RUN) vineyard ingest

prep: ## ingest + tile_prep
	$(RUN) vineyard run --until tile_prep --workers $(WORKERS)

preannotate: ## full perception -> AnnSet(model) -> validated upload ZIPs
	$(RUN) vineyard run --until qa_previews --workers $(WORKERS)
	$(RUN) vineyard export-cvat --annset LATEST_MODEL

reference: ## AnnSet(reference) from 05_examples
	$(RUN) vineyard import-reference

eval: ## official metrics on the 2 example tiles with gates
	$(RUN) vineyard eval-examples --annset $(ANNSET) --gates

test-zip: ## Marcaj format-test ZIP
	$(RUN) vineyard cvat make-test-zip

export: ## upload ZIPs from an AnnSet
	$(RUN) vineyard export-cvat --annset $(ANNSET)

validate-zips: ## independent validation of upload ZIPs
	$(RUN) vineyard cvat validate work/runs/$(ANNSET)/exports/marcaj_upload/*.zip

final: ## Marcaj export -> route.geojson + measurements.csv in repo root
	$(RUN) vineyard from-marcaj $(MARCAJ)
	$(RUN) vineyard post --annset LATEST_MARCAJ --set route.solver.final=true
	$(RUN) vineyard publish --annset LATEST_MARCAJ

post: ## derive -> passable -> targets -> route -> measure -> web_bundle
	$(RUN) vineyard post --annset $(ANNSET)

publish: ## validated copy of the deliverables to the repo root
	$(RUN) vineyard publish --annset $(ANNSET)

web: ## web data bundle for src/Web
	$(RUN) vineyard web build --annset $(ANNSET)

bench: ## timed perception only (ingest -> qa_previews, 311 tiles, no cache) -> metrics/timings.json
	$(RUN) vineyard bench

nn-smoke: ## 1-epoch MPS smoke run
	$(RUN) vineyard nn train --smoke

nn-train: ## full NN training
	$(RUN) vineyard nn train

DOCKER_EXTRAS ?= route
DOCKER_ARGS = --build-arg EXTRAS="$(DOCKER_EXTRAS)" --build-arg VCS_REF="$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

docker-build: ## native arm64 image (DOCKER_EXTRAS="route nn waste-ml" adds torch CPU)
	docker buildx build --platform linux/arm64 $(DOCKER_ARGS) -t vineyard:arm64 --load .

docker-push: ## multi-arch push, run once: IMAGE=<user>/vineyard:<tag>
	@test -n "$(IMAGE)" || { echo "IMAGE is required, e.g. make docker-push IMAGE=<user>/vineyard:1.0" >&2; exit 1; }
	docker buildx build --platform linux/amd64,linux/arm64 $(DOCKER_ARGS) -t $(IMAGE) --push .

clean-work: ## remove caches and runs (irreversible)
	rm -rf work/cache work/runs
