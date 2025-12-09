import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def generate_plots(df_results, dataset_name, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set(style="whitegrid", context="paper", font_scale=1.4)

    custom_palette = {
        'Direct': '#4c72b0',
        'Majoritary (SAT)': '#dd8452',
        'Majoritary (SMT)': '#eec13f',
        'Sufficient (SAT)': '#9d00ff',
        'Sufficient (SMT)': '#55a868'
    }

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # 1) Scatter Complexity vs Sparsity (log scale)
    sns.scatterplot(
        data=df_results, x="Time", y="Size",
        hue="Method", style="Method",
        s=120, alpha=0.8,
        palette=custom_palette,
        ax=ax1
    )
    ax1.set_xscale("log")
    ax1.set_title("")
    ax1.set_xlabel("Time (s) - Log Scale")
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    # 2) Boxplot Size Distribution
    sns.boxplot(
        data=df_results, x="Method", y="Size",
        hue="Method", dodge=False,
        palette=custom_palette,
        medianprops={'color': 'red', 'linewidth': 1.5},
        ax=ax2
    )
    ax2.set_title("")
    ax2.set_xlabel("")
    ax2.tick_params(axis='x', rotation=30)

    # 3) SMT Comparative: Maj vs Suf
    col_x = "Majoritary (SMT)"
    col_y = "Sufficient (SMT)"

    df_pivot = df_results.pivot_table(
        index="Instance_ID",
        columns="Method",
        values="Size",
        aggfunc="mean"
    )

    if col_x in df_pivot.columns and col_y in df_pivot.columns:
        df_clean = df_pivot.dropna(subset=[col_x, col_y])

        sns.scatterplot(
            data=df_clean,
            x=col_x, y=col_y,
            s=120, color='red', alpha=0.6,
            ax=ax3, label="Instances"
        )

        if not df_clean.empty:
            min_val = min(df_clean[col_x].min(), df_clean[col_y].min()) - 1
            max_val = max(df_clean[col_x].max(), df_clean[col_y].max()) + 1

            ax3.plot([min_val, max_val], [min_val, max_val],
                     'r--', linewidth=2, label='y = x')

            ax3.set_xlim(min_val, max_val)
            ax3.set_ylim(min_val, max_val)

        ax3.set_title("")
        ax3.set_xlabel(col_x)
        ax3.set_ylabel(col_y)
        ax3.legend()
        ax3.set_aspect('equal', adjustable='box')
    else:
        ax3.text(0.5, 0.5, "Insufficient data for comparison", ha='center')

    plt.tight_layout()

    filename = f"{output_dir}/{dataset_name}_full_analysis.png"
    plt.savefig(filename, dpi=300)
    print(f"Graphs saved in: {filename}")
    plt.close()

