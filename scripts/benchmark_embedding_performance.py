#!/usr/bin/env python3
"""
Benchmark script for embedding service performance testing.
"""
import asyncio
import time
import json
import logging
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.services.embedding.embedding_service import EmbeddingService
from src.app.services.embedding.optimized_embedding_service import OptimizedEmbeddingService
from src.app.database.mock_collections import create_mock_databases

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingBenchmark:
    """Benchmark class for testing embedding service performance."""
    
    def __init__(self):
        self.test_columns = self._generate_test_columns()
        self.results = {}
    
    def _generate_test_columns(self) -> List[Dict[str, Any]]:
        """Generate test columns for benchmarking."""
        return [
            {
                "name": "patient_id",
                "data_type": "VARCHAR",
                "description": "Unique patient identifier",
                "sample_values": ["P001", "P002", "P003", "P004", "P005"]
            },
            {
                "name": "patient_name",
                "data_type": "VARCHAR",
                "description": "Full name of the patient",
                "sample_values": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Wilson"]
            },
            {
                "name": "date_of_birth",
                "data_type": "DATE",
                "description": "Patient's date of birth",
                "sample_values": ["1990-01-15", "1985-03-22", "1992-07-10", "1988-11-05", "1995-04-18"]
            },
            {
                "name": "email_address",
                "data_type": "VARCHAR",
                "description": "Patient's email address",
                "sample_values": ["john@email.com", "jane@email.com", "bob@email.com", "alice@email.com", "charlie@email.com"]
            },
            {
                "name": "phone_number",
                "data_type": "VARCHAR",
                "description": "Patient's phone number",
                "sample_values": ["+1-555-123-4567", "+1-555-234-5678", "+1-555-345-6789", "+1-555-456-7890", "+1-555-567-8901"]
            },
            {
                "name": "blood_type",
                "data_type": "VARCHAR",
                "description": "Patient's blood type",
                "sample_values": ["A+", "B-", "O+", "AB+", "A-"]
            },
            {
                "name": "height_cm",
                "data_type": "INTEGER",
                "description": "Patient's height in centimeters",
                "sample_values": [175, 162, 180, 168, 185]
            },
            {
                "name": "weight_kg",
                "data_type": "FLOAT",
                "description": "Patient's weight in kilograms",
                "sample_values": [70.5, 55.2, 85.0, 62.8, 90.3]
            },
            {
                "name": "is_active",
                "data_type": "BOOLEAN",
                "description": "Whether the patient is active",
                "sample_values": [True, False, True, True, False]
            },
            {
                "name": "created_at",
                "data_type": "TIMESTAMP",
                "description": "Record creation timestamp",
                "sample_values": ["2023-01-15 10:30:00", "2023-02-20 14:45:00", "2023-03-10 09:15:00", "2023-04-05 16:20:00", "2023-05-12 11:00:00"]
            }
        ] * 10  # Multiply to create more test data
    
    async def benchmark_original_service(self) -> Dict[str, Any]:
        """Benchmark the original embedding service."""
        logger.info("Benchmarking original embedding service...")
        
        service = EmbeddingService()
        results = {}
        
        # Test single embedding generation
        start_time = time.time()
        for column in self.test_columns[:10]:  # Test with first 10
            await service.generate_column_embedding(column)
        single_generation_time = time.time() - start_time
        
        results["single_generation_time"] = single_generation_time
        results["single_generation_rate"] = 10 / single_generation_time
        
        # Test batch processing (simulated)
        start_time = time.time()
        for i in range(0, len(self.test_columns), 5):
            batch = self.test_columns[i:i+5]
            for column in batch:
                await service.generate_column_embedding(column)
        batch_generation_time = time.time() - start_time
        
        results["batch_generation_time"] = batch_generation_time
        results["batch_generation_rate"] = len(self.test_columns) / batch_generation_time
        
        # Test vector index building
        start_time = time.time()
        await service.build_vector_index(self.test_columns[:20])
        index_build_time = time.time() - start_time
        
        results["index_build_time"] = index_build_time
        
        # Test similarity search
        test_embedding = await service.generate_column_embedding(self.test_columns[0])
        start_time = time.time()
        for _ in range(10):
            await service.find_similar_columns(test_embedding["embedding"])
        search_time = time.time() - start_time
        
        results["search_time"] = search_time
        results["search_rate"] = 10 / search_time
        
        return results
    
    async def benchmark_optimized_service(self) -> Dict[str, Any]:
        """Benchmark the optimized embedding service."""
        logger.info("Benchmarking optimized embedding service...")
        
        service = OptimizedEmbeddingService(
            batch_size=32,
            max_workers=4,
            enable_caching=True
        )
        results = {}
        
        # Test single embedding generation with caching
        start_time = time.time()
        for column in self.test_columns[:10]:  # Test with first 10
            await service.generate_column_embedding(column)
        single_generation_time = time.time() - start_time
        
        results["single_generation_time"] = single_generation_time
        results["single_generation_rate"] = 10 / single_generation_time
        
        # Test batch processing
        start_time = time.time()
        await service.batch_generate_embeddings(self.test_columns)
        batch_generation_time = time.time() - start_time
        
        results["batch_generation_time"] = batch_generation_time
        results["batch_generation_rate"] = len(self.test_columns) / batch_generation_time
        
        # Test vector index building
        start_time = time.time()
        await service.build_vector_index(self.test_columns[:20])
        index_build_time = time.time() - start_time
        
        results["index_build_time"] = index_build_time
        
        # Test similarity search
        test_embedding = await service.generate_column_embedding(self.test_columns[0])
        start_time = time.time()
        for _ in range(10):
            await service.find_similar_columns(test_embedding["embedding"])
        search_time = time.time() - start_time
        
        results["search_time"] = search_time
        results["search_rate"] = 10 / search_time
        
        # Get performance metrics
        metrics = service.get_performance_metrics()
        results["performance_metrics"] = metrics
        
        return results
    
    async def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage of both services."""
        import psutil
        import gc
        
        logger.info("Benchmarking memory usage...")
        
        # Test original service memory usage
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        service_original = EmbeddingService()
        await service_original.build_vector_index(self.test_columns[:50])
        
        memory_after_original = process.memory_info().rss / 1024 / 1024  # MB
        original_memory_usage = memory_after_original - initial_memory
        
        del service_original
        gc.collect()
        
        # Test optimized service memory usage
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        service_optimized = OptimizedEmbeddingService()
        await service_optimized.build_vector_index(self.test_columns[:50])
        
        memory_after_optimized = process.memory_info().rss / 1024 / 1024  # MB
        optimized_memory_usage = memory_after_optimized - initial_memory
        
        del service_optimized
        gc.collect()
        
        return {
            "original_memory_usage_mb": original_memory_usage,
            "optimized_memory_usage_mb": optimized_memory_usage,
            "memory_improvement_percent": (
                (original_memory_usage - optimized_memory_usage) / original_memory_usage * 100
            )
        }
    
    async def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with different dataset sizes."""
        logger.info("Benchmarking scalability...")
        
        service_original = EmbeddingService()
        service_optimized = OptimizedEmbeddingService()
        
        scalability_results = {}
        
        # Test different dataset sizes
        for size in [10, 50, 100, 200]:
            test_data = self.test_columns[:size]
            
            # Original service
            start_time = time.time()
            await service_original.build_vector_index(test_data)
            original_time = time.time() - start_time
            
            # Optimized service
            start_time = time.time()
            await service_optimized.build_vector_index(test_data)
            optimized_time = time.time() - start_time
            
            scalability_results[f"size_{size}"] = {
                "original_time": original_time,
                "optimized_time": optimized_time,
                "speedup": original_time / optimized_time if optimized_time > 0 else float('inf')
            }
        
        return scalability_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# Embedding Service Performance Benchmark Report")
        report.append("")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Original vs Optimized comparison
        if "original" in self.results and "optimized" in self.results:
            report.append("## Performance Comparison")
            report.append("")
            
            orig = self.results["original"]
            opt = self.results["optimized"]
            
            # Single generation
            speedup = orig["single_generation_rate"] / opt["single_generation_rate"]
            report.append(f"### Single Embedding Generation")
            report.append(f"- Original: {orig['single_generation_rate']:.2f} columns/sec")
            report.append(f"- Optimized: {opt['single_generation_rate']:.2f} columns/sec")
            report.append(f"- Speedup: {speedup:.2f}x")
            report.append("")
            
            # Batch generation
            speedup = opt["batch_generation_rate"] / orig["batch_generation_rate"]
            report.append(f"### Batch Embedding Generation")
            report.append(f"- Original: {orig['batch_generation_rate']:.2f} columns/sec")
            report.append(f"- Optimized: {opt['batch_generation_rate']:.2f} columns/sec")
            report.append(f"- Speedup: {speedup:.2f}x")
            report.append("")
            
            # Index building
            speedup = orig["index_build_time"] / opt["index_build_time"]
            report.append(f"### Vector Index Building")
            report.append(f"- Original: {orig['index_build_time']:.2f} seconds")
            report.append(f"- Optimized: {opt['index_build_time']:.2f} seconds")
            report.append(f"- Speedup: {speedup:.2f}x")
            report.append("")
            
            # Similarity search
            speedup = opt["search_rate"] / orig["search_rate"]
            report.append(f"### Similarity Search")
            report.append(f"- Original: {orig['search_rate']:.2f} queries/sec")
            report.append(f"- Optimized: {opt['search_rate']:.2f} queries/sec")
            report.append(f"- Speedup: {speedup:.2f}x")
            report.append("")
        
        # Memory usage
        if "memory" in self.results:
            report.append("## Memory Usage")
            report.append("")
            mem = self.results["memory"]
            report.append(f"- Original Service: {mem['original_memory_usage_mb']:.2f} MB")
            report.append(f"- Optimized Service: {mem['optimized_memory_usage_mb']:.2f} MB")
            report.append(f"- Memory Improvement: {mem['memory_improvement_percent']:.1f}%")
            report.append("")
        
        # Scalability
        if "scalability" in self.results:
            report.append("## Scalability Analysis")
            report.append("")
            report.append("| Dataset Size | Original Time | Optimized Time | Speedup |")
            report.append("|--------------|---------------|----------------|---------|")
            
            for size, data in self.results["scalability"].items():
                size_num = size.split("_")[1]
                report.append(
                    f"| {size_num} columns | {data['original_time']:.2f}s | "
                    f"{data['optimized_time']:.2f}s | {data['speedup']:.2f}x |"
                )
            report.append("")
        
        # Performance metrics
        if "optimized" in self.results and "performance_metrics" in self.results["optimized"]:
            report.append("## Optimized Service Metrics")
            report.append("")
            metrics = self.results["optimized"]["performance_metrics"]
            report.append(f"- Average Embedding Time: {metrics['avg_embedding_time']:.4f} seconds")
            report.append(f"- Average Search Time: {metrics['avg_search_time']:.4f} seconds")
            report.append(f"- Average Memory Usage: {metrics['avg_memory_usage']:.2f}%")
            report.append(f"- Cache Hit Rate: {metrics['avg_cache_hit_rate']:.2f}%")
            report.append(f"- Total Columns Processed: {metrics['total_columns_processed']}")
            report.append(f"- Cache Hits: {metrics['cache_hits']}")
            report.append(f"- Cache Misses: {metrics['cache_misses']}")
            report.append("")
        
        # Recommendations
        report.append("## Performance Recommendations")
        report.append("")
        report.append("1. **Use Batch Processing**: Always use batch processing for multiple columns")
        report.append("2. **Enable Caching**: Enable caching for frequently accessed embeddings")
        report.append("3. **Optimize Batch Size**: Use batch size of 32-64 for best performance")
        report.append("4. **Monitor Memory**: Keep track of memory usage for large datasets")
        report.append("5. **Use GPU**: Enable GPU acceleration if available")
        report.append("6. **Regular Cleanup**: Clear cache periodically to prevent memory bloat")
        report.append("")
        
        return "\n".join(report)
    
    async def run_all_benchmarks(self):
        """Run all benchmark tests."""
        logger.info("Starting comprehensive benchmark...")
        
        # Run individual benchmarks
        self.results["original"] = await self.benchmark_original_service()
        self.results["optimized"] = await self.benchmark_optimized_service()
        self.results["memory"] = await self.benchmark_memory_usage()
        self.results["scalability"] = await self.benchmark_scalability()
        
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        with open("benchmark_report.md", "w") as f:
            f.write(report)
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(report)
        print("\nDetailed report saved to: benchmark_report.md")
        
        return self.results

async def main():
    """Main function to run benchmarks."""
    try:
        # Create mock databases for testing
        create_mock_databases()
        
        # Run benchmarks
        benchmark = EmbeddingBenchmark()
        results = await benchmark.run_all_benchmarks()
        
        # Save results as JSON
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: benchmark_results.json")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 