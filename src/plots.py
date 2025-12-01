import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def generate_plots(df_results, dataset_name, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configurações visuais gerais
    sns.set(style="whitegrid", context="paper", font_scale=1.4)
    
    custom_palette = {
        'Direct': '#4c72b0',                  # Azul
        'Majoritary (Greedy)': '#dd8452',     # Laranja
        'Majoritary (Z3)': '#eec13f', # Amarelo/Dourado
        'Sufficient (Z3)': '#55a868',         # Verde
        'Sufficient (Greedy)': '#9d00ff'      # Roxo
    }
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2) 

    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, :]) 

    # 1. Scatter: Complexity vs Sparsity
    sns.scatterplot(
        data=df_results, x="Time", y="Size", 
        hue="Method", style="Method", 
        s=120, alpha=0.8, 
        palette=custom_palette,
        ax=ax1
    )
    ax1.set_xscale("log")
    ax1.set_title(f"Complexity vs Sparsity ({dataset_name})")
    ax1.set_xlabel("Time (s) - Log Scale")
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    # 2. Boxplot: Size Distribution
    
    median_style = {'color': 'red', 'linewidth': 1.5} 
    
    sns.boxplot(
        data=df_results, x="Method", y="Size", 
        hue="Method", dodge=False, 
        palette=custom_palette,
        medianprops=median_style,
        ax=ax2
    )
    ax2.set_title(f"Size Distribution ({dataset_name})")
    ax2.set_xlabel("")
    ax2.tick_params(axis='x', rotation=30)

    # 3. Comparative: Greedy vs Z3
    try:
        df_pivot = df_results.pivot(index='Instance_ID', columns='Method', values='Size')
    except ValueError:
        df_pivot = df_results.pivot_table(index='Instance_ID', columns='Method', values='Size', aggfunc='mean')

    col_x = 'Majoritary (Greedy)'
    col_y = 'Sufficient (Z3)'

    if col_x in df_pivot.columns and col_y in df_pivot.columns:
        df_clean = df_pivot.dropna(subset=[col_x, col_y])
        x_vals = df_clean[col_x]
        y_vals = df_clean[col_y]

        sns.scatterplot(
            x=x_vals, y=y_vals, 
            s=120, color='red', alpha=0.6, # para combinar com a mediana
            ax=ax3, label='Instâncias'
        )
        
        if not df_clean.empty:
            min_val = min(x_vals.min(), y_vals.min()) - 1
            max_val = max(x_vals.max(), y_vals.max()) + 1
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
            ax3.set_xlim(min_val, max_val)
            ax3.set_ylim(min_val, max_val)
        
        ax3.set_title(f"Trade-off: {col_x} vs {col_y}")
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
