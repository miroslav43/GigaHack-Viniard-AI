"""Thread-env defaults and conda PROJ/GDAL scrubbing (arch §4.1)."""

import subprocess
import sys
from pathlib import Path

import pytest

from vineyard import _env

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_thread_defaults_match_arch() -> None:
    assert _env.THREAD_ENV_DEFAULTS == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "GDAL_NUM_THREADS": "1",
        "GDAL_CACHEMAX": "256",
    }


def test_plan_sets_missing_defaults_only() -> None:
    plan = _env.plan_env({"OMP_NUM_THREADS": "4"})
    assert "OMP_NUM_THREADS" not in plan.set_defaults
    assert plan.set_defaults["GDAL_CACHEMAX"] == "256"
    assert plan.drop == ()


def test_plan_drops_conda_proj_and_gdal_vars() -> None:
    env = {
        "CONDA_PREFIX": "/opt/anaconda3",
        "PROJ_DATA": "/opt/anaconda3/share/proj",
        "PROJ_LIB": "/usr/share/proj",
        "GDAL_DATA": "/Users/x/miniconda3/share/gdal",
    }
    plan = _env.plan_env(env)
    assert plan.drop == ("GDAL_DATA", "PROJ_DATA")


def test_plan_is_pure() -> None:
    env = {"PROJ_DATA": "/opt/anaconda3/share/proj"}
    _env.plan_env(env)
    assert env == {"PROJ_DATA": "/opt/anaconda3/share/proj"}


def test_apply_mutates_given_environ_and_warns() -> None:
    env = {"PROJ_DATA": "/opt/anaconda3/share/proj"}
    with pytest.warns(RuntimeWarning, match="PROJ_DATA"):
        plan = _env.apply_env(env)
    assert "PROJ_DATA" not in env
    assert env["OMP_NUM_THREADS"] == "1"
    assert plan.drop == ("PROJ_DATA",)


def test_package_import_sets_env_before_numpy() -> None:
    code = (
        "import os, sys; import vineyard; "
        "print(os.environ['OMP_NUM_THREADS'], 'numpy' in sys.modules, 'PROJ_DATA' in os.environ)"
    )
    env = {"PATH": "/usr/bin:/bin", "PROJ_DATA": "/opt/anaconda3/share/proj", "PYTHONWARNINGS": "ignore",
           "PYTHONPATH": str(PROJECT_ROOT)}
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=PROJECT_ROOT, capture_output=True, text=True,
                         check=True)
    assert out.stdout.split() == ["1", "False", "False"]
