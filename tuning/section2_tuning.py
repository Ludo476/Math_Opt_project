import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_scalability")

INPUT_FILE = os.path.join(BASE_RESULTS_DIR, "tuning_results.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "2_tuning_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# =============================================================================
# 1. PREPROCESSING
# =============================================================================
def load_tuning_data():
    print("Loading Tuning Data...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")
        
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip() 
    
    # Conversione numerica
    cols = ['objective', 'time_s', 'time_to_best_s', 'cooling_rate', 'time_limit_config', 'family', 'seed']
    for c in cols:
        if c in df.columns: 
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['time_limit_config'] = df['time_limit_config'].astype(int)
    
    
    df = df.sort_values(['time_limit_config', 'cooling_rate', 'instance'])
    
    return df

# =============================================================================
# 2. ANALYSIS EFFECTIVENESS (Objective Value - Heatmap)
# =============================================================================
def analyze_effectiveness(df):
    print(" Analyzing Effectiveness (Objective Value)...")
    
    df_aggr_seeds = df.groupby(['cooling_rate', 'time_limit_config', 'instance'])['objective'].mean().reset_index()

    # 2. Pivot Table 
    pivot = df_aggr_seeds.pivot_table(
        index='cooling_rate', 
        columns='time_limit_config', 
        values='objective', 
        aggfunc='mean' 
    ).round(2)
    
    pivot.to_csv(os.path.join(OUT_DIR, "tuning_effectiveness_pivot.csv"))
    
    # Heatmap
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", linewidths=.5, cbar_kws={'label': 'Avg Objective'})
    plt.title("Effectiveness: Avg Objective (Aggregated)")
    plt.ylabel("Cooling Rate")
    plt.xlabel("Time Limit (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_effectiveness_heatmap.png"))
    plt.close()

# =============================================================================
# 3. ANALYSIS EFFICIENCY (Time to Best - Boxplots)
# =============================================================================
def analyze_efficiency(df):
    print("Analyzing efficiency (Time to Best)...")
    
    
    # Plot A: TTB vs Cooling Rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x='cooling_rate', 
        y='time_to_best_s', 
        hue='time_limit_config',
        palette="viridis",
        showfliers=False 
    )
    plt.yscale('log') 
    plt.title("Efficiency: Time to Best Distribution")
    plt.ylabel("Time to Best (s) - Log Scale")
    plt.xlabel("Cooling Rate")
    plt.legend(title="Time Limit (s)", loc='upper right')
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_efficiency_vs_cooling.png"))
    plt.close()

    # Plot B: TTB vs Time Limit 
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x='time_limit_config', 
        y='time_to_best_s', 
        hue='cooling_rate',
        palette="magma",
        showfliers=False
    )
    plt.yscale('log')
    plt.title("Efficiency: Time to Best vs Configured Limit")
    plt.ylabel("Time to Best (s) - Log Scale")
    plt.xlabel("Time Limit Config (s)")
    plt.legend(title="Cooling Rate", loc='upper left')
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_efficiency_vs_timelimit.png"))
    plt.close()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    try:
        df = load_tuning_data()
        
        # (Heatmap mean values)
        analyze_effectiveness(df)
        
        # (Boxplot time scalability)
        analyze_efficiency(df)
        
        print("\n Tuning analysis completed successfully.")
        print(f"   Outputs saved in: {OUT_DIR}")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()