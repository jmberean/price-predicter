"""
Comprehensive test runner for Aetheris Oracle.

Runs all test suites and generates performance/quality reports:
- Unit tests (modules, pipeline, connectors)
- Performance benchmarks
- API validation
- Data quality checks
- SOTA components (if available)
"""

import sys
import time
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    start = time.perf_counter()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        print(f"✓ {description} - PASSED ({elapsed:.1f}s)")
        return True
    else:
        print(f"✗ {description} - FAILED ({elapsed:.1f}s)")
        return False


def main():
    """Run all test suites."""
    print("\n" + "="*60)
    print("AETHERIS ORACLE - COMPREHENSIVE TEST SUITE")
    print("="*60)

    # Set PYTHONPATH
    project_root = Path(__file__).parent
    src_path = str(project_root / "src")
    sys.path.insert(0, src_path)

    # Also set environment variable for subprocess
    import os
    os.environ['PYTHONPATH'] = src_path

    results = {}

    # 1. Data Quality Tests
    results["data_quality"] = run_command(
        "pytest tests/test_data_quality.py -v --tb=short",
        "Data Quality & Connector Tests"
    )

    # 2. Pipeline Tests
    results["pipeline"] = run_command(
        "pytest tests/test_pipeline.py -v --tb=short",
        "Pipeline Integration Tests"
    )

    # 3. Performance Tests
    results["performance"] = run_command(
        "pytest tests/test_performance.py -v --tb=short",
        "Performance & Validation Tests"
    )

    # 4. API Tests
    results["api"] = run_command(
        "pytest tests/test_api_validation.py -v --tb=short",
        "API & Service Tests"
    )

    # 5. SOTA Component Tests (may skip if dependencies not available)
    results["sota"] = run_command(
        "pytest tests/test_sota_components.py -v --tb=short",
        "SOTA Components Tests"
    )

    # 6. Service Tests
    results["service"] = run_command(
        "pytest tests/test_service.py -v --tb=short",
        "Service Tests"
    )

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for name, passed_flag in results.items():
        status = "✓ PASSED" if passed_flag else "✗ FAILED"
        print(f"{name:20s}: {status}")

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
