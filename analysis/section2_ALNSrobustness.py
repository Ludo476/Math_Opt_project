import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv
from pandas.plotting import table

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_scalability")
FILE_ALNS = os.path.join(BASE_RESULTS_DIR, "global_scalability_report_test_ALSN_seeds.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "2_alns_scalability")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def save_table_as_image(df, filename):
    
    if df.empty: return

    w = max(12, len(df.columns) * 1.8) 
    h = max(4, len(df) * 0.6 + 1.8)
    
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis('off')
    
    tab = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center', 
        bbox=[0, 0, 1, 1]
    )
    
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    
    for key, cell in tab.get_celld().items():
        row, col = key
        cell.set_height(0.12) # hight row
        
        if row == 0:
            # Header
            cell.set_text_props(weight='bold', color='white', fontsize=15)
            cell.set_facecolor("#525C9A") 
            cell.set_edgecolor('white')
            cell.set_linewidth(1.3)
        else:
            # Body
            cell.set_facecolor('#f4f4f4' if row % 2 else 'white')
            cell.set_edgecolor('#dddddd')

    plt.title(filename.replace(".png", "").replace("_", " ").upper(), 
              y=1.01, fontsize=16, fontweight='bold',color='#333333')
    
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight', dpi=180)
    plt.close()
    print(f"   -> Immage saved: {filename}")


def load_data():
    if not os.path.exists(FILE_ALNS): return None
    print("Loading ALNS data...")
    try:
        df = pd.read_csv(FILE_ALNS)
        df.columns = df.columns.str.strip()
    except:
        return None
        
    cols = ['objective', 'percent_cs_delivery', 'time_to_best_s', 'iterations', 'reheats', 'improvement_pct', 'num_cs', 'num_parcels', 'class', 'family']
    rename_map = {'improvement_pct': 'improvement_%', 'iterations': 'iter_to_best'} 
    df = df.rename(columns=rename_map)
    
    for c in df.columns:
        if c in cols or c in rename_map.values():
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df

def process_scalability(df):
    def get_specs(row):
        n_map = {0: 24, 1: 36, 3: 44}
        f = int(row['family'])
        N = n_map.get(f, 24)
        K = int(row['num_cs'])
        P = int(row['num_parcels'])
        size_idx = K * P 
        return N, K, P, size_idx

    df[['|N|', '|K|', '|P|', 'size_index']] = df.apply(get_specs, axis=1, result_type='expand')
    return df

# ============================================================================
# SCALABILITY PREPROCESSING
# ============================================================================
def process_scalability(df):
    """Adds |N|, |K|, |P| and size_index columns."""
    def get_specs(row):
        n_map = {0: 24, 1: 36, 3: 44}
        f = int(row['family'])
        N = n_map.get(f, 24)
        K = int(row['num_cs'])
        P = int(row['num_parcels'])
        size_idx = K * P
        return N, K, P, size_idx

    df[['|N|', '|K|', '|P|', 'size_index']] = df.apply(get_specs, axis=1, result_type='expand')
    return df

# ============================================================================
# SUMMARY TABLE (AVG OVER SEEDS -> AVG OVER INSTANCES)
# ============================================================================
def generate_table(df):
    print("Generating ALNS Summary Table...")
    metrics = ['objective', 'percent_cs_delivery', 'time_to_best_s', 'iter_to_best', 'reheats', 'improvement_%']
    
    metrics = [m for m in metrics if m in df.columns]

    df_inst = df.groupby(['|N|', '|K|', '|P|', 'instance'])[metrics].mean().reset_index()

    summary = df_inst.groupby(['|N|', '|K|', '|P|'], as_index=False)[metrics].mean()
    summary = summary.sort_values(by=['|K|', '|P|', '|N|'])

    rename_map = {
        'objective': 'Obj (Avg)', 'percent_cs_delivery': '% CS Del.',
        'time_to_best_s': 'TTB [s]', 'iter_to_best': 'Iter Best',
        'reheats': 'Reheats', 'improvement_%': 'Impr. %'
    }
    summary = summary.rename(columns=rename_map).round(2)

  
    if 'OUT_DIR' in globals():
        summary.to_csv(os.path.join(OUT_DIR, "alns_internal_scalability.csv"), index=False)
        # save_table_as_image(summary, "alns_internal_scalability.png") 
    
    # Print per verifica
    print(summary.to_string(index=False))

# ============================================================================
# ROBUSTNESS ANALYSIS (4-PANEL PLOT)
# ============================================================================
def plot_alns_robustness(df):
    print("\nGenerating Robustness Analysis Plots...")
    df = df.copy()

    # Mapping Classi
    class_map = {0: 'Small', 1: 'Medium', 2: 'Large'}
    if df['class'].dtype in [int, float]:
        df['class_name'] = df['class'].map(class_map).fillna(df['class'])
    else:
        df['class_name'] = df['class']

   
    df['Label'] = df.apply(lambda x: f"K={int(x['num_cs'])}\nP={int(x['num_parcels'])}", axis=1)
    df = df.sort_values(by=['num_parcels', 'num_cs'])

    # --- 1. CV for each instance (Robustness) ---
    robustness_stats = df.groupby(['class_name', 'Label', 'instance'])['objective'].agg(['mean', 'std']).reset_index()
    # CV = (Std / Mean) * 100
    robustness_stats['cv_%'] = np.where(robustness_stats['mean'] > 0, 
                                        (robustness_stats['std'] / robustness_stats['mean']) * 100, 0)
    label_order = df['Label'].unique()
    robustness_stats['Label'] = pd.Categorical(robustness_stats['Label'], categories=label_order, ordered=True)
    robustness_stats = robustness_stats.sort_values('Label')

    print("\nRobustness Summary (CV% across seeds):")
    print(f"Mean CV: {robustness_stats['cv_%'].mean():.2f}%")
    print(f"Max CV:  {robustness_stats['cv_%'].max():.2f}%")

    # --- Plot setup ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    unique_classes = df['class_name'].unique()
    palette = sns.color_palette("viridis", len(unique_classes))

    # Helper function
    def create_hybrid_plot(ax, data_source, x_col, y_col, hue_col, title, y_label, log_scale=False):
        sns.boxplot(
            data=data_source, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax,
            showmeans=True, meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black"},
            flierprops={"marker": "", "linestyle": "none"},
            boxprops={'alpha': 0.6}
        )
        sns.stripplot(
            data=data_source, x=x_col, y=y_col, hue=hue_col, palette=['black'] * len(unique_classes),
            ax=ax, dodge=True, jitter=True, size=4, alpha=0.6, legend=False
        )
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Instance Class")
        ax.set_ylabel(y_label)
        ax.grid(True, axis='y', alpha=0.3)
        if log_scale: ax.set_yscale('log')
        if ax.get_legend(): ax.get_legend().remove()

    # PANEL 1: Time (Structural + Stochastic Variance)
    create_hybrid_plot(axes[0], df, 'Label', 'time_to_best_s', 'class_name', 
                       "Convergence Speed", "Time to Best (s)", log_scale=True)

    # PANEL 2: Iterations (Structural + Stochastic Variance)
    create_hybrid_plot(axes[1], df, 'Label', 'iter_to_best', 'class_name', 
                       "Search Effort", "Iterations to Best", log_scale=True)

    # PANEL 3: Improvement (Normalized Performance)
    create_hybrid_plot(axes[2], df, 'Label', 'improvement_%', 'class_name', 
                       "Heuristic Effectiveness", "Improvement over Initial (%)")

    # PANEL 4: Stability (CV distribution per instance)
    create_hybrid_plot(axes[3], robustness_stats, 'Label', 'cv_%', 'class_name',
                       "Stability: Coeff. of Variation (CV)", "CV of Objective (%)")
    
    axes[3].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1% Threshold')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(unique_classes)], labels[:len(unique_classes)],
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(unique_classes), title="Class Size")
    plt.suptitle("ALNS Robustness & Scalability Analysis", y=1.05, fontsize=18, fontweight='bold')
    
    if 'OUT_DIR' in globals():
        out_file = os.path.join(OUT_DIR, "plot_alns_robustness_cv1.png")
        plt.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close()
        print(f" -> Plot saved: {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    df = load_data()
    if 'df' in locals() and not df.empty:
        df = process_scalability(df)
        generate_table(df)
        plot_alns_robustness(df)
        print("\n Section 2 ALNS scalability completed.")
    else:
        print("File ALNS not found.")


