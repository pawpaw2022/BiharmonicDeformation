# HarmonicDeformation Tests

This directory contains unit tests and benchmarks for the PyTorch implementation of harmonic deformation in `src/HarmonicDeformation.py`. The tests validate the correctness of the implementation and compare its performance against the libigl implementation.

## Structure

The test suite is organized as follows:

- `test_utils.py`: Utility functions for creating test meshes and computing errors
- `test_harmonic_functions.py`: Unit tests for individual functions in HarmonicDeformation.py
- `test_accuracy.py`: Accuracy comparison between PyTorch and libigl implementations
- `libigl_harmonic.py`: libigl implementation wrapper for comparison
- `pytorch_harmonic.py`: PyTorch benchmarking utilities
- `benchmark_comparison.py`: Detailed benchmarks comparing PyTorch and libigl implementations
- `run_tests.py`: Script to run all tests and benchmarks

## Requirements

The tests require the following packages:
- PyTorch
- libigl
- NumPy
- pandas
- matplotlib
- All dependencies specified in the project's requirements.txt

## Running Tests

You can run the entire test suite or specific test components:

### Run all tests and benchmarks

```bash
python -m tests.run_tests
```

### Run only unit tests (skipping benchmarks)

```bash
python -m tests.run_tests --tests-only
```

### Run only benchmarks (skipping unit tests)

```bash
python -m tests.run_tests --benchmark-only
```

### Run benchmarks with CPU only (no CUDA)

```bash
python -m tests.run_tests --cpu-only
```

### Run benchmarks with smaller test cases (faster, for quick testing)

```bash
python -m tests.run_tests --small-only
```

### Run individual test files

```bash
python -m unittest tests.test_harmonic_functions
python -m unittest tests.test_accuracy
```

## Benchmark Results

The benchmark results are saved in the `results` directory:
- CSV files with timing information
- Plots comparing the performance of different implementations

### Key benchmark plots:
- `harmonic_performance.png`: Performance comparison for k=1 (harmonic)
- `biharmonic_performance.png`: Performance comparison for k=2 (biharmonic)
- `performance_scaling.png`: Log-log plot showing asymptotic complexity
- `implementation_comparison.png`: Bar chart comparing implementations for specific mesh sizes

## Notes

- The accuracy tests use relative error thresholds of 1% for harmonic (k=1) and 5% for biharmonic (k=2) deformation.
- The benchmarks include tests on plane meshes of various sizes, from 10×10 to 100×100 vertices.
- For GPU benchmarks, CUDA synchronization is used for accurate timing measurements. 