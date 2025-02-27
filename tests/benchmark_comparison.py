"""
Benchmark script to compare the performance of PyTorch and libigl implementations
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import (
    create_plane_mesh,
    create_cube_mesh,
    torch_to_numpy,
    numpy_to_torch
)
from tests.pytorch_harmonic import benchmark_pytorch_harmonic
from tests.libigl_harmonic import benchmark_libigl_harmonic

def create_benchmark_cases() -> List[Dict]:
    """
    Create benchmark test cases with different mesh sizes.
    
    Returns:
        List[Dict]
            List of test cases for benchmarking
    """
    test_cases = []
    
    # Test with plane meshes of different sizes
    for n in [10, 20, 30, 40, 50, 75, 100]:
        # Create plane mesh
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
        
        # biharmonic case
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

def run_benchmarks(
    cpu_only: bool = False,
    repeat: int = 3,
    save_results: bool = True,
    make_plots: bool = True
) -> Dict:
    """
    Run benchmarks and return results.
    
    Parameters:
        cpu_only: bool, default False
            If True, only run on CPU
        repeat: int, default 3
            Number of times to repeat each test
        save_results: bool, default True
            If True, save results to CSV
        make_plots: bool, default True
            If True, generate and save plots
            
    Returns:
        Dict
            Dictionary with benchmark results
    """
    print("Creating benchmark cases...")
    test_cases = create_benchmark_cases()
    
    # Run CPU benchmarks
    print("\nRunning libigl benchmarks (CPU)...")
    libigl_results = benchmark_libigl_harmonic(test_cases, repeat=repeat)
    
    print("\nRunning PyTorch benchmarks (CPU)...")
    pytorch_cpu_results = benchmark_pytorch_harmonic(
        test_cases, 
        repeat=repeat,
        device=torch.device('cpu')
    )
    
    results = {
        'libigl_cpu': libigl_results,
        'pytorch_cpu': pytorch_cpu_results
    }
    
    # Run GPU benchmarks if available and not cpu_only
    if not cpu_only and torch.cuda.is_available():
        print("\nRunning PyTorch benchmarks (CUDA)...")
        pytorch_cuda_results = benchmark_pytorch_harmonic(
            test_cases, 
            repeat=repeat,
            device=torch.device('cuda')
        )
        results['pytorch_cuda'] = pytorch_cuda_results
    
    # Save results and make plots if requested
    if save_results:
        save_benchmark_results(results)
    
    if make_plots:
        create_benchmark_plots(results)
    
    return results

def save_benchmark_results(results: Dict) -> None:
    """
    Save benchmark results to CSV files.
    
    Parameters:
        results: Dict
            Dictionary with benchmark results
    """
    os.makedirs('results', exist_ok=True)
    
    # Create DataFrame for each implementation
    for impl_name, impl_results in results.items():
        data = []
        for case_name, case_results in impl_results.items():
            row = {
                'case': case_name,
                'avg_time': case_results['avg_time'],
                'std_time': case_results['std_time'],
                'vertices_count': case_results['vertices_count'],
                'faces_count': case_results['faces_count'],
                'k': case_results['k']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(f'results/{impl_name}_results.csv', index=False)
    
    # Create a combined DataFrame for easy comparison
    combined_data = []
    for impl_name, impl_results in results.items():
        for case_name, case_results in impl_results.items():
            row = {
                'implementation': impl_name,
                'case': case_name,
                'avg_time': case_results['avg_time'],
                'std_time': case_results['std_time'],
                'vertices_count': case_results['vertices_count'],
                'faces_count': case_results['faces_count'],
                'k': case_results['k']
            }
            combined_data.append(row)
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('results/combined_results.csv', index=False)

def create_benchmark_plots(results: Dict) -> None:
    """
    Create plots of benchmark results.
    
    Parameters:
        results: Dict
            Dictionary with benchmark results
    """
    os.makedirs('results', exist_ok=True)
    
    # Extract data
    data = []
    for impl_name, impl_results in results.items():
        for case_name, case_results in impl_results.items():
            row = {
                'implementation': impl_name,
                'case': case_name,
                'avg_time': case_results['avg_time'],
                'vertices_count': case_results['vertices_count'],
                'k': case_results['k']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # 1. Plot time vs mesh size for k=1
    plt.figure(figsize=(10, 6))
    for impl in df['implementation'].unique():
        subset = df[(df['implementation'] == impl) & (df['k'] == 1)]
        plt.plot(subset['vertices_count'], subset['avg_time'], 'o-', label=impl)
    
    plt.xlabel('Number of Vertices')
    plt.ylabel('Average Time (s)')
    plt.title('Harmonic Deformation (k=1) Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/harmonic_performance.png', dpi=300)
    
    # 2. Plot time vs mesh size for k=2
    plt.figure(figsize=(10, 6))
    for impl in df['implementation'].unique():
        subset = df[(df['implementation'] == impl) & (df['k'] == 2)]
        plt.plot(subset['vertices_count'], subset['avg_time'], 'o-', label=impl)
    
    plt.xlabel('Number of Vertices')
    plt.ylabel('Average Time (s)')
    plt.title('Biharmonic Deformation (k=2) Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/biharmonic_performance.png', dpi=300)
    
    # 3. Create log-log plot to show asymptotic complexity
    plt.figure(figsize=(10, 6))
    for impl in df['implementation'].unique():
        for k in [1, 2]:
            subset = df[(df['implementation'] == impl) & (df['k'] == k)]
            plt.loglog(
                subset['vertices_count'], 
                subset['avg_time'], 
                'o-', 
                label=f'{impl} (k={k})'
            )
    
    plt.xlabel('Number of Vertices (log scale)')
    plt.ylabel('Average Time (s) (log scale)')
    plt.title('Performance Scaling (Log-Log Plot)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/performance_scaling.png', dpi=300)
    
    # 4. Bar chart comparing implementations
    plt.figure(figsize=(12, 6))
    
    # Filter for specific mesh sizes to make the chart readable
    target_vertices = [400, 2500, 10000]  # corresponds to 20x20, 50x50, 100x100
    filtered_df = df[df['vertices_count'].isin(target_vertices)]
    
    # Group by implementation, vertices count and k
    grouped = filtered_df.groupby(['implementation', 'vertices_count', 'k'])['avg_time'].mean().reset_index()
    
    # Pivot for plotting
    pivot_df = grouped.pivot(index=['vertices_count', 'k'], columns='implementation', values='avg_time')
    
    # Plot
    pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('(Vertices Count, k)')
    plt.ylabel('Average Time (s)')
    plt.title('Performance Comparison for Different Mesh Sizes')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('results/implementation_comparison.png', dpi=300)

def print_speedup_analysis(results: Dict) -> None:
    """
    Print speedup analysis from benchmark results.
    
    Parameters:
        results: Dict
            Dictionary with benchmark results
    """
    print("\n=== Speedup Analysis ===")
    
    # Extract data
    data = []
    for impl_name, impl_results in results.items():
        for case_name, case_results in impl_results.items():
            row = {
                'implementation': impl_name,
                'case': case_name,
                'avg_time': case_results['avg_time'],
                'vertices_count': case_results['vertices_count'],
                'k': case_results['k']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Compute speedup for each case
    cases = df['case'].unique()
    
    print("\nSpeedup relative to libigl (CPU):")
    print("="*50)
    print(f"{'Case':<15} {'Vertices':<10} {'k':<5} {'PyTorch CPU':<15} {'PyTorch CUDA':<15}")
    print("-"*50)
    
    for case in cases:
        case_df = df[df['case'] == case]
        
        # Get baseline time (libigl)
        if 'libigl_cpu' not in case_df['implementation'].values:
            continue
            
        baseline_time = case_df[case_df['implementation'] == 'libigl_cpu']['avg_time'].values[0]
        vertices = case_df['vertices_count'].values[0]
        k = case_df['k'].values[0]
        
        # Compute speedups
        pytorch_cpu_time = case_df[case_df['implementation'] == 'pytorch_cpu']['avg_time'].values[0]
        pytorch_cpu_speedup = baseline_time / pytorch_cpu_time
        
        # Check if CUDA results exist
        if 'pytorch_cuda' in case_df['implementation'].values:
            pytorch_cuda_time = case_df[case_df['implementation'] == 'pytorch_cuda']['avg_time'].values[0]
            pytorch_cuda_speedup = baseline_time / pytorch_cuda_time
            print(f"{case:<15} {vertices:<10} {k:<5} {pytorch_cpu_speedup:<15.2f} {pytorch_cuda_speedup:<15.2f}")
        else:
            print(f"{case:<15} {vertices:<10} {k:<5} {pytorch_cpu_speedup:<15.2f} {'N/A':<15}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmark comparison between PyTorch and libigl implementations')
    parser.add_argument('--cpu-only', action='store_true', help='Run only CPU benchmarks (no CUDA)')
    parser.add_argument('--repeat', type=int, default=3, help='Number of times to repeat each benchmark')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to CSV')
    parser.add_argument('--no-plots', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()
    
    print("Starting benchmarks...")
    results = run_benchmarks(
        cpu_only=args.cpu_only,
        repeat=args.repeat,
        save_results=not args.no_save,
        make_plots=not args.no_plots
    )
    
    print_speedup_analysis(results)
    
    print("\nBenchmarks completed.")
    if not args.no_save and not args.no_plots:
        print("Results and plots saved in the 'results' directory.") 