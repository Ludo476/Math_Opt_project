import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import ast
import csv
from pandas.plotting import table

# =============================================================================
# CONFIGURATION PATH
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_scalability")

FILE_MILP = os.path.join(BASE_RESULTS_DIR, "final_benchmark_results_milp.csv")
FILE_ALNS = os.path.join(BASE_RESULTS_DIR, "final_benchmark_results_alns.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "3_final_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# =============================================================================
# 1. LOADER
# =============================================================================
def load_csv_robust(filepath):
    if not os.path.exists(filepath):
        print(f"no file: {filepath}")
        return pd.DataFrame()

    print(f"   -> Reading {os.path.basename(filepath)}...")
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try: 
            raw_header = next(reader)
            header = [h.strip() for h in raw_header] #
        except: return pd.DataFrame()
        
        h_idx = -1
        for i, h in enumerate(header):
            if 'history' in h.lower(): h_idx = i
            
        expected = len(header)
        for row in reader:
            if not row: continue
            if len(row) > expected and h_idx != -1:
                extra = len(row) - expected
                row[h_idx] = ",".join(row[h_idx : h_idx + 1 + extra])
                del row[h_idx + 1 : h_idx + 1 + extra]
            if len(row) == expected:
                rows.append(row)
                
    return pd.DataFrame(rows, columns=header)

def parse_history_safe(x):
    try:
        if pd.isna(x) or str(x).strip() == "": return []
        x = str(x).replace('""', '"').replace("'", '"')
        if x.startswith('"') and x.endswith('"'): x = x[1:-1]
        return json.loads(x)
    except: return []

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



# =============================================================================
# 2. PREPROCESSING
# =============================================================================
def load_and_align_data():
    print("loading dataset...")
    df_m = load_csv_robust(FILE_MILP)
    df_a = load_csv_robust(FILE_ALNS)
    
    if df_m.empty or df_a.empty: return pd.DataFrame(), pd.DataFrame()

    # Rename MILP
    df_m = df_m.rename(columns={'time_to_best_s': 'ttb_s', 'percent_cs_delivery': 'percent_cs'})

    # Convert Numeric
    cols_num = ['objective', 'time_s', 'ttb_s', 'percent_cs', 'class', 'family', 'gap']
    for df in [df_m, df_a]:
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'history' in df.columns:
            df['history'] = df['history'].apply(parse_history_safe)

    # Size Label and sort index
    def get_specs(row):
        f = int(row['family'])
        c = int(row['class'])
        
        # Family -> Nodes (|N|)
        n_map = {0: 24, 1: 36, 2: 44, 3: 44}
        N = n_map.get(f, 24)
        
        # Map Class ->  (|K|, |P|)
        # Class 0: (50, 30)
        # Class 1: (75, 50)
        # Class 2: (100, 75)
        # Class 6: (50, 30) (in the first dataset and run I didn't have it so I renamed for global scalaility as class 6)
        kp_map = {
            0: (50, 30),
            1: (75, 50),
            2: (100, 75),
            
        }
        
        if 'num_cs' in row and pd.notnull(row['num_cs']):
            K = int(row['num_cs'])
            P = int(row['num_parcels'])
        else:
            K, P = kp_map.get(c, (0, 0)) # Default 0,0 
        
        # Sort index: first for P (parcel), then for N (stations)
        s_idx = P * 1000 + N 
        
        # Label plots
        label = f"K={K}, P={P}\n(N={N})"
        return N, K, P, s_idx, label

    for df in [df_m, df_a]:
        df[['|N|', '|K|', '|P|', 'Sort_Idx', 'Size_Label']] = df.apply(get_specs, axis=1, result_type='expand')

    return df_m, df_a

# =============================================================================
# 3. GAP, MATCH RATE, TTB
# =============================================================================
def generate_comparison_stats(df_m, df_a):
    print("generating comparison statistics...")
    
    # 1. Aggregate ALNS for instance (among seeds)
    
    alns_stats = df_a.groupby(['Sort_Idx', '|N|', '|K|', '|P|', 'instance']).agg({
        'objective': ['min', 'mean', 'max'], 
        'ttb_s': 'mean',
        'time_s': 'mean'
    })
    # (es: objective_min, objective_mean...)
    alns_stats.columns = ['_'.join(col) for col in alns_stats.columns]
    alns_stats = alns_stats.reset_index()
    
    # 2. Union with MILP (Target)
    
    milp_ref = df_m[['instance', 'objective', 'ttb_s', 'time_s']].rename(
        columns={'objective': 'MILP_Obj', 'ttb_s': 'MILP_TTB', 'time_s': 'MILP_TotalTime'}
    )
    
    merged = alns_stats.merge(milp_ref, on='instance')
    
    # 3. Compute GAP (%)
    # Formula: (Value ALNS - Value MILP) / Value MILP * 100
    # Gap Best: how much the best seed is far from MILP?
    merged['Gap_Best'] = (merged['objective_min'] - merged['MILP_Obj']) / merged['MILP_Obj'] * 100
    # Gap Mean: how much ALNS is far from MILP in mean?
    merged['Gap_Mean'] = (merged['objective_mean'] - merged['MILP_Obj']) / merged['MILP_Obj'] * 100
    # Gap Worst:how much the worst seed is far from MILP?
    merged['Gap_Worst'] = (merged['objective_max'] - merged['MILP_Obj']) / merged['MILP_Obj'] * 100
    
    # 4.compute SPEEDUP (Efficiency)
    # how much ALNS is faster than MILP to find a sol?
    merged['TTB Speedup'] = merged['MILP_TTB'] / (merged['ttb_s_mean'] + 0.001)

    # 5. MATCH RATE (optimality percentage)
    # how many ALNS found a value <= MILP (tolerate 0.01%)
    match_percentages = []
    for inst in merged['instance']:
        milp_val = df_m[df_m['instance'] == inst]['objective'].values[0]
        alns_vals = df_a[df_a['instance'] == inst]['objective'].values
        tolerance = 1e-4
        matches = np.sum(alns_vals <= milp_val * (1 + tolerance))
        match_percentages.append(matches / len(alns_vals) * 100)
    
    merged['Match Rate'] = match_percentages
    
    # mean over metrics
    final_table = merged.groupby(['Sort_Idx', '|K|', '|P|', '|N|']).agg({
        'MILP_Obj': 'mean',          #  MILP mean cost
        'objective_min': 'mean',     #  ALNS mean cost (Best seed)
        'Gap_Best': 'mean',      #  mean Gap best seed
        'Gap_Mean': 'mean',      # mean GAP mean seeds
        'Gap_Worst': 'mean',     # mean GAP worst seed
        'Match Rate': 'mean',    # % mean successes
        'MILP_TTB': 'mean',          # TTB  MILP mean
        'ttb_s_mean': 'mean',        # TTB  ALNS mean
        'TTB Speedup': 'mean'    # mean Speedup
    }).reset_index().sort_values('Sort_Idx')
    
    final_table = final_table.rename(columns={'objective_min': 'ALNS_Best', 'ttb_s_mean': 'ALNS_TTB'})
    
    cols_out = [
        '|K|', '|P|', '|N|', 
        'Match Rate',    # reliability
        'Gap_Best',      # max quality
        'Gap_Mean',      # mean quality
        'Gap_Worst',     # robustness (worst case)
        'TTB Speedup',   # relative efficiency
        'MILP_TTB',          # MILP TTB
        'ALNS_TTB'           # ALNS TTB
    ]
    
    path = os.path.join(OUT_DIR, "final_comparison_table.csv")
    final_table[cols_out].round(2).to_csv(path, index=False)
    save_table_as_image(final_table[cols_out].round(2), "final_comparison_table_img.png")
    
    print("\n" + "="*80)
    print(" COMPARISON SUMMARY (Top Rows)")
    print("="*80)
    print(final_table[cols_out].head().to_string(index=False))
    print(f"\n   -> Stats saved in: {path}")
    
    return final_table, merged

# =============================================================================
# 4. PLOTS
# =============================================================================
def generate_plots(comp_table, df_m, df_a):
    print(" Generating plots...")
    comp_table['Label'] = comp_table.apply(lambda x: f"K={int(x['|K|'])}, P={int(x['|P|'])}\n(N={int(x['|N|'])})", axis=1)
    
    # A. TTB Comparison (Log Scale)
    plt.figure(figsize=(10, 6))
    plot_data = comp_table.melt(id_vars='Label', value_vars=['MILP_TTB', 'ALNS_TTB_Avg'], var_name='Solver', value_name='Time')
    sns.barplot(data=plot_data, x='Label', y='Time', hue='Solver', palette=['gray', 'green'])
    plt.yscale('log')
    plt.title("Efficiency Comparison: Time to Best Solution")
    plt.ylabel("Time (s) - Log Scale")
    plt.xlabel("Instance Configuration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_ttb_comparison.png"))
    plt.close()


# =============================================================================
# 5. (3x3 Scalability Grid)
# =============================================================================
def repair_history(history_list, final_time, final_obj, initial_cost=None):
    
    if not history_list or not isinstance(history_list, list):
        return [0, final_time], [final_obj, final_obj]
    try:
        history_list.sort(key=lambda x: x[0])
        t, v = zip(*history_list)
        t, v = list(t), list(v)
    except:
        return [0, final_time], [final_obj, final_obj]
    
    if t[0] > 0:
        start_val = initial_cost if pd.notnull(initial_cost) else v[0] * 1.05 
        t.insert(0, 0.0)
        v.insert(0, start_val)
    if t[-1] < final_time:
        t.append(final_time)
        v.append(v[-1])
    return t, v

# =============================================================================
# HISTORIES
# =============================================================================
def generate_plots(comp_table, df_m, df_a):
    print(" Generating plots...")
    comp_table['Size_Label'] = comp_table.apply(
        lambda x: f"K={int(x['|K|'])}, P={int(x['|P|'])}\n(N={int(x['|N|'])})", 
        axis=1)
    
    # --- A. TTB COMPARISON (Standard) ---
    plt.figure(figsize=(10, 6))
    plot_data = comp_table.melt(
        id_vars='Size_Label', 
        value_vars=['MILP_TTB', 'ALNS_TTB'], 
        var_name='Solver', 
        value_name='Time')

    plot_data['Solver'] = plot_data['Solver'].replace({'MILP_TTB':'MILP', 'ALNS_TTB':'ALNS'})
    sns.barplot(data=plot_data, x='Size_Label', y='Time', hue='Solver', palette=['gray', 'green'])
    plt.yscale('log')
    plt.title("Efficiency Comparison: Time to Best Solution")
    plt.ylabel("Time (s) - Log Scale")
    plt.xlabel("Instance Configuration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_ttb_comparison.png"))
    plt.close()

    # --- B. MATRIX CONVERGENCE PLOTS (3 Files: Inst 0, 1, 2) ---
    rows_map = {
        0: "K=50, P=30",
        1: "K=75, P=50", 
        2: "K=100, P=75"
    }
    cols_map = {
        0: "N=24", 
        1: "N=36", 
        2: "N=44"
    }
    # loop for number of instance (0, 1, 2)
    for inst_idx in range(3):
        print(f"   -> Generation matrix for instance {inst_idx}...")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        
        for r_cls in range(3):
            for c_fam in range(3):
                ax = axes[r_cls, c_fam]
            
                # Filter for class, family and instance id (es: "_inst_0.json")
                suffix = f"_inst_{inst_idx}.json"
                
                subset_m = df_m[
                    (df_m['class'] == r_cls) & 
                    (df_m['family'] == c_fam) & 
                    (df_m['instance'].str.endswith(suffix))
                ]
                
                if not subset_m.empty:
                    m_row = subset_m.iloc[0]
                    inst_name = m_row['instance']
                    
                    # --- PLOT MILP ---
                    ft_m = m_row['time_s'] if pd.notnull(m_row['time_s']) else 3600.0
                    t_m, v_m = repair_history(m_row['history'], ft_m, m_row['objective'])
                    ax.step(t_m, v_m, where='post', color='black', linewidth=2, label='MILP')
                    ax.axhline(m_row['objective'], color='black', ls=':', alpha=0.5)

                    # --- PLOT ALNS ---
                    # Corresponding instance
                    a_rows = df_a[df_a['instance'] == inst_name]
                    for i, (_, row) in enumerate(a_rows.iterrows()):
                        ft_a = row['time_s'] if pd.notnull(row['time_s']) else 600.0
                        init_c = row['initial_cost'] if 'initial_cost' in row else None
                        t_a, v_a = repair_history(row['history'], ft_a, row['objective'], initial_cost=init_c)
                        ax.step(t_a, v_a, where='post', alpha=0.6, linewidth=1.5, label='ALNS' if i==0 else "")
                    
                    ax.set_xscale('log')
                    ax.grid(True, alpha=0.3)
                    
                    # Titles
                    if r_cls == 0:
                        ax.set_title(f"Network Size: {cols_map[c_fam]}", fontsize=14, fontweight='bold')
                    if c_fam == 0:
                        ax.set_ylabel(f"Load: {rows_map[r_cls]}\nObjective", fontsize=12, fontweight='bold')
                    # X Label 
                    if r_cls == 2:
                        ax.set_xlabel("Time (s)", fontsize=10)    
                    # Legend
                    if r_cls == 0 and c_fam == 2:
                        ax.legend(loc='upper right', fontsize='small')
                else:
                    # if the instance does not exists
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                    ax.axis('off')

        
        fig.suptitle(f"Convergence Matrix - Random Seed Instance {inst_idx}", fontsize=16)
        
        # Save
        plt.savefig(os.path.join(OUT_DIR, f"matrix_convergence_inst_{inst_idx}.png"), dpi=150)
        plt.close()

    print("   -> Matrices saved.")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df_m, df_a = load_and_align_data()
    if not df_m.empty and not df_a.empty:
        comp_tab, merged_raw = generate_comparison_stats(df_m, df_a)
        generate_plots(comp_tab, df_m, df_a)
        print("\n Section 3 comparison completed.")
    else:
        print("missing data.")