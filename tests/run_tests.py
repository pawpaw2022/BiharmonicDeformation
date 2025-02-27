"""
Runner script to execute all tests and benchmarks.
"""
import unittest
import os
import sys
import argparse

def run_unit_tests():
    """Run all unit tests."""
    print("\n=== Running Unit Tests ===")
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_benchmark(cpu_only=False, small_only=False):
    """Run benchmarks."""
    print("\n=== Running Benchmarks ===")
    
    # Import benchmark module
    from tests.benchmark_comparison import run_benchmarks
    
    # Reduce test cases if small_only is True
    kwargs = {
        "cpu_only": cpu_only,
        "repeat": 2 if small_only else 3,
        "save_results": True,
        "make_plots": True
    }
    
    # Patch create_benchmark_cases if small_only is True
    if small_only:
        from tests.benchmark_comparison import create_benchmark_cases
        
        # Store original function
        original_create_benchmark_cases = create_benchmark_cases
        
        # Create patched function
        def patched_create_benchmark_cases():
            test_cases = []
            # Only use smaller mesh sizes
            for n in [10, 20, 30]:
                # Create plane mesh
                import torch
                from tests.test_utils import create_plane_mesh
                
                vertices, faces = create_plane_mesh(n)
                
                # Use the outer ring of vertices as boundary
                x_coords = vertices[:, 0]
                y_coords = vertices[:, 1]
                boundary_mask = (x_coords == -1) | (x_coords == 1) | (y_coords == -1) | (y_coords == 1)
                boundary_vertices = torch.nonzero(boundary_mask).squeeze()
                
                # Create boundary values (tent shape)
                boundary_values = vertices[boundary_vertices].clone()
                
                test_cases.append({
                    'name': f'plane_{n}x{n}_k1',
                    'vertices': vertices,
                    'faces': faces,
                    'boundary_vertices': boundary_vertices,
                    'boundary_values': boundary_values,
                    'k': 1,
                    'vertices_count': len(vertices),
                    'faces_count': len(faces)
                })
                
                # Add biharmonic case
                test_cases.append({
                    'name': f'plane_{n}x{n}_k2',
                    'vertices': vertices,
                    'faces': faces,
                    'boundary_vertices': boundary_vertices,
                    'boundary_values': boundary_values,
                    'k': 2,
                    'vertices_count': len(vertices),
                    'faces_count': len(faces)
                })
            
            return test_cases
        
        # Replace function
        import tests.benchmark_comparison
        tests.benchmark_comparison.create_benchmark_cases = patched_create_benchmark_cases
    
    # Run benchmarks
    results = run_benchmarks(**kwargs)
    
    # Print speedup analysis
    from tests.benchmark_comparison import print_speedup_analysis
    print_speedup_analysis(results)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests and benchmarks for HarmonicDeformation")
    parser.add_argument("--tests-only", action="store_true", help="Run only unit tests, skip benchmarks")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only benchmarks, skip unit tests")
    parser.add_argument("--cpu-only", action="store_true", help="Run benchmarks on CPU only")
    parser.add_argument("--small-only", action="store_true", help="Run benchmarks with small test cases only")
    
    args = parser.parse_args()
    
    success = True
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    if not args.benchmark_only:
        print("\n=== Starting Unit Tests ===")
        success = run_unit_tests() and success
    
    if not args.tests_only:
        print("\n=== Starting Benchmarks ===")
        success = run_benchmark(args.cpu_only, args.small_only) and success
    
    if success:
        print("\nAll tests and benchmarks completed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests or benchmarks failed!")
        sys.exit(1) 