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
                return True
            else:
                f.write("No data found\n")
                return False
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        return False


def process_logs_and_save(input_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for file_path in input_files:
        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        print(f"Processing file: {file_path}")
        query_data, error = process_file(file_path)

        if error:
            print(error)
            continue

        if save_results(query_data, output_path):
            # Extract query number and type (naive or lip)
            file_parts = os.path.splitext(base_name)[0].split("-")
            if len(file_parts) >= 2:
                query_num = file_parts[0]
                query_type = file_parts[1]

                # Store the average time
                avg_time = sum(data["time_query"] for data in query_data) / len(
                    query_data
                )
                if query_num not in results:
                    results[query_num] = {}
                results[query_num][query_type] = avg_time

                print(f"  Saved average time ({avg_time:.5f}s) to {output_path}")
            else:
                print(
                    f"  Warning: Could not parse query number and type from filename: {base_name}"
                )
        else:
            print(f"  Error: Could not save results for {file_path}")

    return results


def plot_results(results):
    # Convert results to lists for plotting
    query_nums = []
    naive_times = []
    lip_times = []

    for query_num, times in results.items():
        if "naive" in times and "lip" in times:
            query_nums.append(int(query_num))
            naive_times.append(times["naive"])
            lip_times.append(times["lip"])

    # Sort everything based on naive execution times
    sorted_indices = np.argsort(naive_times)
    sorted_query_nums = [query_nums[i] for i in sorted_indices]
    sorted_naive_times = [naive_times[i] for i in sorted_indices]
    sorted_lip_times = [lip_times[i] for i in sorted_indices]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    # Create x-axis positions (1 through N)
    x_positions = list(range(1, len(sorted_query_nums) + 1))

    # Draw the vertical lines from LIP to Naive
    for i in range(len(x_positions)):
        plt.plot(
            [x_positions[i], x_positions[i]],
            [sorted_lip_times[i], sorted_naive_times[i]],
            "k-",
            linewidth=1,
        )

    # Plot the data points with the specific markers
    plt.plot(
        x_positions,
        sorted_naive_times,
        "D",
        markersize=12,
        markerfacecolor="orange",
        markeredgecolor="black",
        label="Naive",
    )
    plt.plot(
        x_positions,
        sorted_lip_times,
        "o",
        markersize=12,
        markerfacecolor="lightblue",
        markeredgecolor="black",
        label="LIP",
    )

    # Set the axis labels and title
    plt.xlabel("Query plans", fontsize=14)
    plt.ylabel("Execution time (ms)", fontsize=14)
    plt.xlim(0, len(x_positions) + 1)
    plt.ylim(0, max(sorted_naive_times) * 1.2)  # Add 20% margin to y-axis

    # Add the legend
    plt.legend(loc="upper left", fontsize=14)

    # Add the figure title
    # plt.figtext(
    #     0.5,
    #     0.01,
    #     f"Figure 1: All {len(sorted_query_nums)} possible left-deep query plans\n"
    #     f"in increasing order of execution time.",
    #     ha="center",
    #     fontsize=14,
    # )

    # Show the plot
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("ssb-q33-jo.png", dpi=300)
    plt.show()

    # Print the sorted data for reference
    print("\nSorted Query Execution Times:")
    print("Query Plan | Naive Time (ms) | LIP Time (ms) | Speedup")
    print("-" * 60)
    for i in range(len(sorted_query_nums)):
        print(
            f"{sorted_query_nums[i]:^10} | {sorted_naive_times[i]:^13.5f} | {sorted_lip_times[i]:^11.5f} | {sorted_naive_times[i] / sorted_lip_times[i]:^7.5f}"
        )


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python process_logs.py <output_directory> <log_file1> [log_file2] ..."
        )
        return

    output_dir = sys.argv[1]
    input_files = sys.argv[2:]

    print(f"Processing {len(input_files)} log files...")
    results = process_logs_and_save(input_files, output_dir)

    if results:
        print("\nGenerating plot...")
        plot_results(results)
    else:
        print("No valid results to plot.")


if __name__ == "__main__":
    main()
