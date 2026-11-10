from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_packages_product_and_compatibility_namespaces() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'wheel.packages = ["analysis", "kangaroo"]' in pyproject
    assert '"numpy"' in pyproject


def test_generated_flatbuffers_do_not_overwrite_analysis_package_initializer() -> None:
    cmake = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert '${KANGAROO_FLATBUFFER_PYTHON_DIR}/analysis/fb/' in cmake
    assert '${KANGAROO_FLATBUFFER_PYTHON_DIR}/analysis/"' not in cmake
