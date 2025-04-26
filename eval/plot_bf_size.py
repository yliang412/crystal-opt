import json

import numpy as np
import matplotlib.pyplot as plt

dim_size = 131072
dim_sizes = []
logs = []
for i in range(12):
    dim_sizes.append(dim_size)
    dim_logs = []
    with open("../logs/sj_{}.txt".format(dim_size), "r") as fi:
        fi_lines = fi.readlines()

    for j in range(11):
        trial_1 = json.loads(fi_lines[10 * j + 3].strip())
        trial_2 = json.loads(fi_lines[10 * j + 5].strip())
        trial_3 = json.loads(fi_lines[10 * j + 7].strip())
        dim_logs.append([trial_1, trial_2, trial_3])

    logs.append(dim_logs)
    dim_size *= 2

bf_sizes = [0] + dim_sizes.copy()[2:]
dim_sizes_text = ["128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M"]
bf_sizes_text = ["None", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M"]

# First plot: Execution time over Bloom Filter sizes
dim_avg_times = []
for i in range(len(dim_sizes)):
    dim_size = dim_sizes[i]
    dim_logs = logs[i]
    dim_avg_time_over_bf = []
    for j in range(len(dim_logs)):
        bf_logs = dim_logs[j]
        avg_time = sum([trial['time_join_total'] for trial in bf_logs]) / len(bf_logs)
        dim_avg_time_over_bf.append(avg_time)
    dim_avg_times.append(dim_avg_time_over_bf)
    if i >= 3:
        line, = plt.plot(bf_sizes[1:], dim_avg_time_over_bf[1:], label='dim={}'.format(dim_sizes_text[i]), marker='.')
        plt.axhline(y=dim_avg_time_over_bf[0], color=line.get_color(), linestyle=':')

# print(dim_avg_times[9])

plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xscale('log')
plt.xticks(bf_sizes[1:], labels=bf_sizes_text[1:])
plt.xlabel('Bloom Filter Length (bits)')
plt.ylabel('Execution Time (ms)')

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0, 1, 1])

plt.savefig('bf_size_plot.png', dpi=300)

# Second plot: Execution time over Dim table sizes
plt.figure()

no_bf_times = [dim_avg_time[0] for dim_avg_time in dim_avg_times]
plt.plot(dim_sizes, no_bf_times, label='Without bloom filter', marker='.')

dim_avg_times_np = np.array(dim_avg_times)
dim_min_times = np.min(dim_avg_times_np, axis=1)
dim_min_times_bf_ind = np.argmin(dim_avg_times_np, axis=1)
print([bf_sizes_text[bf_ind] for bf_ind in dim_min_times_bf_ind])
plt.plot(dim_sizes, dim_min_times, label='Optimal filter length', marker='.')

dim_times_bf_32m = [dim_avg_time[7] for dim_avg_time in dim_avg_times]
plt.plot(dim_sizes, dim_times_bf_32m, label='Filter length = 32M', marker='.')

dim_times_bf_fixed_ratio_opt = []
for dim_ind in range(len(dim_sizes)):
    bf_ind = dim_ind - 4 if dim_ind > 4 else 1
    dim_times_bf_fixed_ratio_opt.append(dim_avg_times[dim_ind][bf_ind])
plt.plot(dim_sizes, dim_times_bf_fixed_ratio_opt, label='Filter length =\n(# items in filter) * 16',
         marker='.')

dim_times_bf_fixed_ratio_alt = []
for dim_ind in range(len(dim_sizes)):
    bf_ind = dim_ind - 2 if dim_ind > 2 else 1
    dim_times_bf_fixed_ratio_alt.append(dim_avg_times[dim_ind][bf_ind])
plt.plot(dim_sizes, dim_times_bf_fixed_ratio_alt, label='Filter length =\n(# items in filter) * 64',
         marker='.')

plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xscale('log')
plt.xticks(dim_sizes, labels=dim_sizes_text)
plt.xlabel('Dimension Table Sizes (rows)')
plt.ylabel('Execution Time (ms)')
plt.legend(loc='upper left')
plt.savefig('dim_size_plot.png', dpi=300)

# Third plot: Execution time breakdown
plt.figure()

categories = ["dim=512K", "dim=64M"]
part_names = ["Build Time", "Probe Time", "Extra Time"]
part_color = ["blue", "orange", "green"]
group_1 = {
    0: [sum([trial['time_build'] for trial in logs[2][0]]) / len(logs[2][0]),
        sum([trial['time_build'] for trial in logs[9][0]]) / len(logs[9][0])],
    1: [sum([trial['time_probe'] for trial in logs[2][0]]) / len(logs[2][0]),
        sum([trial['time_probe'] for trial in logs[9][0]]) / len(logs[9][0])],
    2: [sum([trial['time_extra'] for trial in logs[2][0]]) / len(logs[2][0]),
        sum([trial['time_extra'] for trial in logs[9][0]]) / len(logs[9][0])]
}
# Dim=512K, BF=4K
# {"time_memset":0.044928,"time_build"0.073728,"time_probe":20.1574}
# {"num_dim":524288,"num_fact":1073741824,"num_select":4096,"num_bloom_filter_bits":4096,"time_build":0.073728,"time_probe":20.1574,"time_extra":0.044928,"time_join_total":20.2761}
# {"time_memset":0.019456,"time_build"0.018432,"time_probe":20.0898}
# {"num_dim":524288,"num_fact":1073741824,"num_select":4096,"num_bloom_filter_bits":4096,"time_build":0.018432,"time_probe":20.0898,"time_extra":0.019456,"time_join_total":20.1277}
# {"time_memset":0.018432,"time_build"0.014336,"time_probe":20.098}
# {"num_dim":524288,"num_fact":1073741824,"num_select":4096,"num_bloom_filter_bits":4096,"time_build":0.014336,"time_probe":20.098,"time_extra":0.018432,"time_join_total":20.1308}
group_2 = {
    0: [0.0354987,
        sum([trial['time_build'] for trial in logs[9][1]]) / len(logs[9][1])],
    1: [20.115067,
        sum([trial['time_probe'] for trial in logs[9][1]]) / len(logs[9][1])],
    2: [0.0276053,
        sum([trial['time_extra'] for trial in logs[9][1]]) / len(logs[9][1])]
}
group_3 = {
    0: [sum([trial['time_build'] for trial in logs[2][1]]) / len(logs[2][1]),
        sum([trial['time_build'] for trial in logs[9][5]]) / len(logs[9][5])],
    1: [sum([trial['time_probe'] for trial in logs[2][1]]) / len(logs[2][1]),
        sum([trial['time_probe'] for trial in logs[9][5]]) / len(logs[9][5])],
    2: [sum([trial['time_extra'] for trial in logs[2][1]]) / len(logs[2][1]),
        sum([trial['time_extra'] for trial in logs[9][5]]) / len(logs[9][5])]
}
group_4 = {
    0: [sum([trial['time_build'] for trial in logs[2][10]]) / len(logs[2][10]),
        sum([trial['time_build'] for trial in logs[9][10]]) / len(logs[9][10])],
    1: [sum([trial['time_probe'] for trial in logs[2][10]]) / len(logs[2][10]),
        sum([trial['time_probe'] for trial in logs[9][10]]) / len(logs[9][10])],
    2: [sum([trial['time_extra'] for trial in logs[2][10]]) / len(logs[2][10]),
        sum([trial['time_extra'] for trial in logs[9][10]]) / len(logs[9][10])]
}
groups = [group_1, group_2, group_3, group_4]
group_names_per_category = [['Without bloom filter', 'Very small filter (4K)',
                             'Ideal filter length (64K)', 'Very big filter (256M)'],
                            ['Without bloom filter', 'Very small filter (512K)',
                             'Ideal filter length (8M)', 'Very big filter (256M)']]
bar_width = 0.5
fig, axs = plt.subplots(1, len(categories), figsize=(10, 6), sharey=False)

for cat_idx, ax in enumerate(axs):
    offsets = np.arange(len(groups))
    plotted_parts = set()

    for grp_idx, group in enumerate(groups):
        bottom = 0
        for part_id, values in group.items():
            label = part_names[part_id] if part_id not in plotted_parts else "_nolegend_"
            height = values[cat_idx]
            ax.bar(offsets[grp_idx], height, bar_width, label=label, color=part_color[part_id],
                   bottom=bottom)
            y_text = bottom + height / 2 if part_id == 1 else bottom + height * 3
            ax.text(offsets[grp_idx], y_text, str(round(height, 2)),
                    ha='center', va='center', fontsize=8, color='black')
            bottom += height
            plotted_parts.add(part_id)

        ax.text(offsets[grp_idx], -0.5, group_names_per_category[cat_idx][grp_idx],
                ha='right', va='top', fontsize=9, rotation=20)
    x_fig = ax.transData.transform((len(groups) / 2 - 0.5, 0))[0]
    x_fig = fig.transFigure.inverted().transform((x_fig, 0))[0]
    ax_pos = ax.get_position()
    x_center = (ax_pos.x0 + ax_pos.x1) / 2
    fig.text(x_center, 0.02, categories[cat_idx], ha='center', va='top',
             fontsize=10, fontweight='bold')

    ax.set_xticks(offsets)
    ax.set_xticklabels([''] * len(groups))  # suppress default ticks
    ax.set_ylabel('Execution Time (ms)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right')

plt.tight_layout()
plt.savefig('breakdown_plot.png', dpi=300)
