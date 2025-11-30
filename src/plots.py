import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def generate_plots(df_results, dataset_name, output_dir="plots"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configs
    sns.set(style="whitegrid", context="paper", font_scale=1.4)
    
    # Layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2) 

    # Eixos
    ax1 = fig.add_subplot(gs[0, 0]) # Linha 0, Coluna 0
    ax2 = fig.add_subplot(gs[0, 1]) # Linha 0, Coluna 1
    ax3 = fig.add_subplot(gs[1, :]) # Linha 1, Todas as colunas

    # Scatter Plot
    sns.scatterplot(
        data=df_results, 
        x="Time", 
        y="Size", 
        hue="Method", 
        style="Method", 
        s=100, 
        alpha=0.8,
        ax=ax1
    )
    ax1.set_xscale("log")
    ax1.set_title(f"Complexity vs Sparsity ({dataset_name})")
    ax1.set_xlabel("Time (seconds) - Log Scale")
    ax1.set_ylabel("Explanation Size (# Literals)")
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    # Box Plot
    sns.boxplot(
        data=df_results, 
        x="Method", 
        y="Size", 
        hue="Method", 
        dodge=False,
        palette="Set2",
        ax=ax2
    )
    ax2.set_title(f"Size Distribution ({dataset_name})")
    ax2.set_ylabel("Size")
    ax2.set_xlabel("")

    # Scatterplot comparativo
    try:
        df_pivot = df_results.pivot(index='Instance_ID', columns='Method', values='Size')
    except ValueError:
        df_pivot = df_results.pivot_table(index='Instance_ID', columns='Method', values='Size', aggfunc='mean')

    col_x = 'Majoritary'
    col_y = 'Sufficient (Z3)'

    if col_x in df_pivot.columns and col_y in df_pivot.columns:
        x_vals = df_pivot[col_x]
        y_vals = df_pivot[col_y]

        sns.scatterplot(x=x_vals, y=y_vals, s=120, color='purple', alpha=0.7, ax=ax3)
        
        min_val = min(x_vals.min(), y_vals.min()) - 1
        max_val = max(x_vals.max(), y_vals.max()) + 1
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (Identity)')

        ax3.set_title(f"Trade-off: Majoritary vs Sufficient ({dataset_name})")
        ax3.set_xlabel("Majoritary Size")
        ax3.set_ylabel("Sufficient Size (Z3)")
        ax3.legend()
        
        ax3.set_xlim(min_val, max_val)
        ax3.set_ylim(min_val, max_val)
        ax3.set_aspect('equal', adjustable='box')
    else:
        ax3.text(0.5, 0.5, "Insufficient data for comparison", 
                 ha='center', va='center', fontsize=14)

    plt.tight_layout()
    
    filename = f"{output_dir}/{dataset_name}_full_analysis.png"
    plt.savefig(filename, dpi=300)
    print(f"Full image saved in: {filename}")
    plt.close()