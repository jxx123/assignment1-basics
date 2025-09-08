from cs336_basics import transformer, train
import torch
import timeit
import numpy as np
import argparse
from typing import Any
import pandas as pd
import os


def benchmark_model(
    config: dict[str, Any],
    num_steps: int = 20,
    warmup_steps: int = 5,
    compile: bool = True,
) -> dict[str, Any]:
    # Extract parameters from config
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    rope_theta = config["rope_theta"]
    batch_size = config["batch_size"]
    device = config["device"]

    model = transformer.TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    if compile:
        if device == "mps":
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model)
    model.to(device)

    forward_times = []
    backward_times = []
    total_times = []
    for step in range(num_steps):
        x_batch = torch.randint(0, vocab_size, size=(batch_size, context_length)).to(device)
        y_batch = torch.randint(0, vocab_size, size=(batch_size, context_length)).to(device)

        ts = timeit.default_timer()
        logits = model(x_batch)
        loss = train.cross_entropy(logits, y_batch)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        tfe = timeit.default_timer()

        loss.backward()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        tbe = timeit.default_timer()

        if step >= warmup_steps:
            forward_times.append(tfe - ts)
            backward_times.append(tbe - tfe)
            total_times.append(tbe - ts)

    forward_times = np.array(forward_times)
    backward_times = np.array(backward_times)
    total_times = np.array(total_times)

    mean_forward_time = forward_times.mean()
    std_forward_time = forward_times.std()
    mean_backward_time = backward_times.mean()
    std_backward_time = backward_times.std()
    mean_total_time = total_times.mean()
    std_total_time = total_times.std()

    # Calculate model parameters
    num_params = sum(p.numel() for p in model.parameters())

    result = {
        "config": config.copy(),
        "performance": {
            "forward_time_mean": float(mean_forward_time),
            "forward_time_std": float(std_forward_time),
            "backward_time_mean": float(mean_backward_time),
            "backward_time_std": float(std_backward_time),
            "total_time_mean": float(mean_total_time),
            "total_time_std": float(std_total_time),
        },
        "model_info": {
            "num_parameters": num_params,
            "num_steps": num_steps,
            "warmup_steps": warmup_steps,
            "compiled": compile,
        },
    }

    print(f"Config: {config['name'] if 'name' in config else 'Unnamed'}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Forward time: {mean_forward_time:.4f} ± {std_forward_time:.4f} s")
    print(f"  Backward time: {mean_backward_time:.4f} ± {std_backward_time:.4f} s")
    print()

    return result


def run_benchmarks(
    device: str = "mps", num_steps: int = 20, warmup_steps: int = 5, compile: bool = False
) -> list[dict[str, Any]]:
    """Run benchmarks on multiple model configurations and return results."""

    # Define model configurations based on the comments in the original file
    # Size d_model d_ff num_layers num_heads
    # small 768 3072 12 12
    # medium 1024 4096 24 16
    # large 1280 5120 36 20
    # xl 1600 6400 48 25
    # 2.7B 2560 10240 32 32

    base_config = {
        "vocab_size": 10000,
        "context_length": 256,
        "rope_theta": 10000.0,
        "batch_size": 4,
        "device": device,
    }

    model_configs = [
        {"name": "tiny", "d_model": 256, "d_ff": 1024, "num_layers": 4, "num_heads": 8, **base_config},
        {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, **base_config},
        {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, **base_config},
        # {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, **base_config},
        # Note: Larger configs commented out as they may be too large for some systems
        # {
        #     'name': 'xl',
        #     'd_model': 1600,
        #     'd_ff': 6400,
        #     'num_layers': 48,
        #     'num_heads': 25,
        #     **base_config
        # },
        # {
        #     'name': '2.7B',
        #     'd_model': 2560,
        #     'd_ff': 10240,
        #     'num_layers': 32,
        #     'num_heads': 32,
        #     **base_config
        # },
    ]

    results = []
    print("Running benchmarks on multiple model configurations...\n")

    for config in model_configs:
        print(f"Benchmarking {config['name']} model...")
        try:
            result = benchmark_model(config, num_steps=num_steps, warmup_steps=warmup_steps, compile=compile)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {config['name']}: {e}")
            continue

    return results


def display_results_summary(results: list[dict[str, Any]]) -> None:
    """Display a summary comparison of all benchmark results."""
    if not results:
        print("No results to display.")
        return

    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Table header
    print(f"{'Model':<10} {'Params':<12} {'Forward (s)':<15} {'Backward (s)':<15} {'Total (s)':<15}")
    print("-" * 80)

    # Sort results by number of parameters
    sorted_results = sorted(results, key=lambda x: x["model_info"]["num_parameters"])

    for result in sorted_results:
        config = result["config"]
        perf = result["performance"]
        model_info = result["model_info"]

        name = config["name"]
        params = f"{model_info['num_parameters']:,}"
        forward_time = f"{perf['forward_time_mean']:.4f}±{perf['forward_time_std']:.4f}"
        backward_time = f"{perf['backward_time_mean']:.4f}±{perf['backward_time_std']:.4f}"
        total_time = f"{perf['total_time_mean']:.4f}±{perf['total_time_std']:.4f}"

        print(f"{name:<10} {params:<12} {forward_time:<15} {backward_time:<15} {total_time:<15}")

    print("\n" + "=" * 80)


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert benchmark results to a pandas DataFrame."""
    if not results:
        return pd.DataFrame()

    # Flatten the results for DataFrame creation
    flattened_data = []
    for result in results:
        config = result["config"]
        perf = result["performance"]
        model_info = result["model_info"]

        row = {
            "model_name": config["name"],
            "vocab_size": config["vocab_size"],
            "context_length": config["context_length"],
            "d_model": config["d_model"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "d_ff": config["d_ff"],
            "rope_theta": config["rope_theta"],
            "batch_size": config["batch_size"],
            "device": config["device"],
            "num_parameters": model_info["num_parameters"],
            "forward_time_mean": perf["forward_time_mean"],
            "forward_time_std": perf["forward_time_std"],
            "backward_time_mean": perf["backward_time_mean"],
            "backward_time_std": perf["backward_time_std"],
            "total_time_mean": perf["total_time_mean"],
            "total_time_std": perf["total_time_std"],
            "num_steps": model_info["num_steps"],
            "warmup_steps": model_info["warmup_steps"],
            "compiled": model_info["compiled"],
        }
        flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    # Sort by number of parameters
    df = df.sort_values("num_parameters").reset_index(drop=True)

    return df


def save_results_to_markdown(results: list[dict[str, Any]], filename: str = "benchmark_results.md") -> None:
    """Save benchmark results to a markdown file using pandas DataFrame."""
    df = results_to_dataframe(results)

    if df.empty:
        print("No results to save.")
        return

    # Create a simplified view for markdown export
    summary_df = df[
        [
            "model_name",
            "num_parameters",
            "forward_time_mean",
            "forward_time_std",
            "backward_time_mean",
            "backward_time_std",
            "total_time_mean",
            "total_time_std",
        ]
    ].copy()

    # Format the numbers for better readability
    summary_df["num_parameters"] = summary_df["num_parameters"].apply(lambda x: f"{x:,}")
    summary_df["forward_time"] = summary_df.apply(
        lambda row: f"{row['forward_time_mean']:.4f}±{row['forward_time_std']:.4f}", axis=1
    )
    summary_df["backward_time"] = summary_df.apply(
        lambda row: f"{row['backward_time_mean']:.4f}±{row['backward_time_std']:.4f}", axis=1
    )
    summary_df["total_time"] = summary_df.apply(
        lambda row: f"{row['total_time_mean']:.4f}±{row['total_time_std']:.4f}", axis=1
    )

    # Select final columns for markdown
    final_df = summary_df[["model_name", "num_parameters", "forward_time", "backward_time", "total_time"]].copy()
    final_df.columns = ["Model", "Parameters", "Forward Time (s)", "Backward Time (s)", "Total Time (s)"]

    # Export to markdown
    markdown_content = final_df.to_markdown(index=False, tablefmt="github")

    with open(filename, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(markdown_content)
        f.write("\n")

    print(f"Results saved to {filename}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Performance")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Default markdown output filename")
    parser.add_argument(
        "--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Device to run benchmarks on"
    )
    parser.add_argument("--num_steps", type=int, default=20, help="Number of benchmark steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (default)")
    args = parser.parse_args()

    # Determine compilation setting
    use_compile = args.compile and not args.no_compile

    # Run benchmarks
    results = run_benchmarks(args.device, args.num_steps, args.warmup_steps, use_compile)

    # Display summary
    display_results_summary(results)

    # Convert to DataFrame for analysis
    if results:
        df = results_to_dataframe(results)
        print(f"\nDataFrame created with {len(df)} rows and {len(df.columns)} columns")

        output_dir = "benchmark_results"
        os.makedirs(output_dir, exist_ok=True)
        save_results_to_markdown(results, os.path.join(output_dir, f"{args.output}.md"))

        return df
    else:
        print("No results to save.")
        return None


if __name__ == "__main__":
    main()
