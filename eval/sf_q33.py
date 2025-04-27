import re
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def extract_query_data(log_text):
    # Regular expression to find JSON patterns like {"query":33,"time_query":4.93018}
    pattern = r'{"query":\d+,"time_query":\d+\.\d+}'

    # Find all matches
    matches = re.findall(pattern, log_text)

    # Parse each match into a Python dictionary
    results = []
    for match in matches:
        data = json.loads(match)
        results.append(data)

    return results


def process_file(file_path):
    try:
        # Read input from the specified file
        with open(file_path, "r") as f:
            log_content = f.read()

        # Extract the data
        query_data = extract_query_data(log_content)

        return query_data, None

    except FileNotFoundError:
        return [], f"Error: File '{file_path}' not found"
    except Exception as e:
        return [], f"Error processing file '{file_path}': {e}"


def save_results(query_data, output_file):
    try:
        with open(output_file, "w") as f:
            # Calculate average time
            if query_data:
                avg_time = sum(data["time_query"] for data in query_data) / len(
                    query_data
                )
                f.write(f"{avg_time:.5f}\n")
                return True, avg_time
            else:
                f.write("No data found\n")
                return False, None
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        return False, None


def process_logs_and_save(input_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = defaultdict(dict)

    for file_path in input_files:
        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        print(f"Processing file: {file_path}")
        query_data, error = process_file(file_path)

        if error:
            print(error)
            continue

        success, avg_time = save_results(query_data, output_path)
        if success:
            # Extract scale factor and type (naive or lip) from filename format <sf>-5-{lip,naive}.txt
            file_parts = os.path.splitext(base_name)[0].split("-")
            if len(file_parts) >= 3:
                scale_factor = file_parts[0]
                query_type = file_parts[2]  # lip or naive

                # Store the average time
                results[scale_factor][query_type] = avg_time

                print(f"  Saved average time ({avg_time:.5f}s) to {output_path}")
            else:
                print(
                    f"  Warning: Could not parse scale factor and type from filename: {base_name}"
                )
        else:
            print(f"  Error: Could not save results for {file_path}")

    return results


def plot_scale_factor_results(results):
    # Convert results to lists for plotting
    scale_factors = []
    naive_times = []
    lip_times = []

    for sf, times in sorted(results.items(), key=lambda x: int(x[0])):
        if "naive" in times and "lip" in times:
            scale_factors.append(int(sf))
            naive_times.append(times["naive"])
            lip_times.append(times["lip"])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    # Plot the data points with lines connecting them
    plt.plot(
        scale_factors,
        naive_times,
        "o-",
        markersize=12,
        color="black",
        markerfacecolor="orange",
        markeredgecolor="black",
        label="Naive",
    )
    plt.plot(
        scale_factors,
        lip_times,
        "^-",
        markersize=12,
        color="black",
        markerfacecolor="lightblue",
        markeredgecolor="black",
        label="LIP",
    )

    # Set the axis labels and title
    plt.xlabel("Scale Factor", fontsize=14)
    plt.ylabel("Execution time (ms)", fontsize=14)

    # Add the legend
    plt.legend(loc="upper left", fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.savefig("ssb-q33-sf.png", dpi=300)
    plt.show()

    # Print the data for reference
    print("\nExecution Times by Scale Factor:")
    print("Scale Factor | Naive Time (ms) | LIP Time (ms) | Speedup")
    print("-" * 60)
    for i in range(len(scale_factors)):
        print(
            f"{scale_factors[i]:^12} | {naive_times[i]:^13.5f} | {lip_times[i]:^11.5f} | {naive_times[i] / lip_times[i]:^7.5f}"
        )


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python process_scale_factor.py <output_directory> <log_file1> [log_file2] ..."
        )
        return

    output_dir = sys.argv[1]
    input_files = sys.argv[2:]

    print(f"Processing {len(input_files)} log files...")
    results = process_logs_and_save(input_files, output_dir)

    if results:
        print("\nGenerating scale factor plot...")
        plot_scale_factor_results(results)
    else:
        print("No valid results to plot.")


if __name__ == "__main__":
    main()
