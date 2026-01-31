import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from pandas.plotting import table
import numpy as np

# =============================================================================
# 1. CONFIGURATION AND PATH
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_scalability")

FILE_BASE = os.path.join(BASE_RESULTS_DIR, "global_scalability_results_milp_base.csv")
FILE_CB   = os.path.join(BASE_RESULTS_DIR, "global_scalability_results_milp_callback.csv")
FILE_CUTS = os.path.join(BASE_RESULTS_DIR, "global_scalability_results_milp_cuts.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "1_milp_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

TIME_LIMIT = 3600.0
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# =============================================================================
# UTILS: SAVE TABLE 
# =============================================================================
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

def print_pretty_table(title, df):
    print("\n" + "="*80)
    print(f" {title.upper()}")
    print("="*80)
    # Hide index in console output too
    print(df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("-" * 80 + "\n")

# =============================================================================
# UTILS: SAVE TABLE (HIERARCHICAL)
# =============================================================================
def save_hierarchical_table(df, filename, title_text):
    if df.empty: return

    # 1. DataFrame
    solver_short_map = {
        'MILP_Base': 'Sim', 'Simple': 'Sim',
        'MILP_Cuts': 'Cut', 'Cuts': 'Cut',
        'MILP_Callback': 'Cbk', 'Callback': 'Cbk'
    }
    
    new_cols = []
    for col in df.columns:
        if col in ['Size (N)', '|P|', '|K|']:
            new_cols.append(('', col))
        else:
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                metric, solver = parts
                s_short = solver_short_map.get(solver, solver[:3])
                metric = metric.replace('integrality_gap_root', 'Root Gap').replace('presolve_time', 'Presolve').capitalize()
                metric = metric.replace('Time_to_best_s', 'TTB (s)')
                new_cols.append((metric, s_short))
            else:
                new_cols.append((col, ''))

    df.columns = pd.MultiIndex.from_tuples(new_cols)
    
    # 2. Setup Plot
    w = max(12, len(df.columns) * 1.1) 
    h = max(4, len(df) * 0.4 + 2.5)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis('off')

    # 3. Data table
    header_top = [c[0] for c in df.columns]
    header_sub = [c[1] for c in df.columns]
    data_rows = df.values.tolist()
    final_data = [header_top, header_sub] + data_rows
    
    tab = ax.table(
        cellText=final_data, 
        loc='center', 
        cellLoc='center', 
        bbox=[0, 0, 1, 1])
    
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    
    # 4. Styling 
    num_cols = len(df.columns)
    
    # Colori
    HEADER_BG = "#404040"  
    SUB_BG = "#d9d9d9"     
    
    for (row, col), cell in tab.get_celld().items():
        cell.set_height(0.08)
       
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=14)
            cell.set_facecolor(HEADER_BG)
            
            # to unify the cells
            cell.set_edgecolor(HEADER_BG) 
            cell.set_linewidth(1)
            
            txt = header_top[col]
            if col > 0 and header_top[col-1] == txt and txt != '':
                cell.get_text().set_text("")

        # --- SUB HEADER ---
        elif row == 1:
            cell.set_text_props(weight='bold', color='#333333', fontsize=11)
            cell.set_facecolor(SUB_BG)
            cell.set_edgecolor('white') 
           
            if header_top[col] == '': 
                cell.set_facecolor(HEADER_BG)
                cell.set_edgecolor(HEADER_BG) 
                cell.get_text().set_text(header_sub[col])
                cell.set_text_props(weight='bold', color='white', fontsize=14)

        # --- DATA ---
        else:
            real_row_idx = row - 2
            cell.set_facecolor('#f4f4f4' if real_row_idx % 2 else 'white')
            cell.set_edgecolor('#dddddd')
            
            val = cell.get_text().get_text()
            try:
                f_val = float(val)
                if f_val.is_integer():
                    cell.get_text().set_text(f"{int(f_val)}")
                elif abs(f_val) < 0.01 and f_val != 0:
                    cell.get_text().set_text(f"{f_val:.1e}")
                else:
                    cell.get_text().set_text(f"{f_val:.2f}")
            except: pass

    plt.title(title_text.upper(), y=1.02, fontsize=16, fontweight='bold', color='#333333')
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"   -> Image saved: {filename}")

# =============================================================================
# 2. LOADERS
# =============================================================================
def assign_group_columns(row):
    fam = int(row['family'])
    nodes_map = {0: 24, 1: 36, 2: 44, 3: 44} 
    N = nodes_map.get(fam, 24)
    raw_K = int(row['num_cs']) if pd.notnull(row.get('num_cs')) else 0
    raw_P = int(row['num_parcels']) if pd.notnull(row.get('num_parcels')) else 0
    return N, raw_K, raw_P

def get_conditional_n_group(row):
    """
    - if P >= 160 (big instances): group N=36 and N=44 with label ">=36"
    - if P < 160 (small instances): keep distinct (24, 36, 44)
    """
    n_val = row['|N|']
    p_val = row['|P|']
    
    if p_val >= 160:
        if n_val >= 36:
            return ">=36"
        else:
            return str(n_val)
    else:
        return str(n_val)

def load_data():
    print("Loading MILP data...")
    dfs = []
    for fpath, name in [(FILE_BASE, 'MILP_Base'), (FILE_CB, 'MILP_Callback'), (FILE_CUTS, 'MILP_Cuts')]:
        if os.path.exists(fpath):
            try:
                d = pd.read_csv(fpath)
                d['solver'] = name
                dfs.append(d)
            except: pass
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    oom_mask = df['status'].astype(str).str.contains("OutOfMemory|Time", case=False, na=False)
    df.loc[oom_mask, 'time_s'] = TIME_LIMIT
    df.loc[oom_mask, 'gap'] = 1.00 
    df.loc[df['time_s'] > TIME_LIMIT, 'time_s'] = TIME_LIMIT

    new_cols = [
        'integrality_gap_root', 'presolve_time', 'num_vars', 'num_constrs', 'percent_cs_delivery', 'time_to_best_s'
    ]

    cols = ['time_s', 'gap', 'nodes', 'objective', 'root_bound_after_cuts', 'num_cs', 'num_parcels']
    for c in cols + new_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'objective' in df.columns and 'root_bound_after_cuts' in df.columns:
        valid_mask = (df['objective'].abs() > 1e-9) & (df['root_bound_after_cuts'].notna())
        
        #  |Obj - Bound| / |Obj|
        calculated_gap = (
            (df.loc[valid_mask, 'objective'] - df.loc[valid_mask, 'root_bound_after_cuts']).abs() 
            / df.loc[valid_mask, 'objective'].abs())
        
        df.loc[valid_mask, 'integrality_gap_root'] = calculated_gap

    if 'percent_cs_delivery' in df.columns:
        if df['percent_cs_delivery'].max() <= 1.0:
            df['percent_cs_delivery'] = df['percent_cs_delivery'] * 100.0

    df[['|N|', '|K|', '|P|']] = df.apply(assign_group_columns, axis=1, result_type='expand')

    df['N_Group'] = df.apply(get_conditional_n_group, axis=1)
    
    df = df.sort_values(by=['|P|', '|N|'])
    return df

# =============================================================================
# 3. TABLE GENERATION (CLEAN & SHORT NAMES)
# =============================================================================
def generate_tables(df):
    print("Generating Tables...")

    # --- Pre-processing ---
    df_agg = df.copy()
    
    if 'root_bound_after_cuts' in df_agg.columns:
        df_agg = df_agg.rename(columns={'root_bound_after_cuts': 'Root Bound'})

    df_agg['Disp_P'] = df_agg['|P|'].apply(lambda x: ">=160" if x >= 160 else str(int(x)))
    df_agg['Disp_K'] = df_agg['|K|'].apply(lambda x: ">=200" if x >= 200 else str(int(x)))
    df_agg['sort_P'] = df_agg['|P|'].apply(lambda x: 9999 if x >= 160 else x)

    solver_map = {
        'MILP_Base': 'Simple',
        'MILP_Cuts': 'Cuts',
        'MILP_Callback': 'Callback'
    }
    df_agg['solver_short'] = df_agg['solver'].map(solver_map).fillna(df_agg['solver'])

    # ---------------------------
    # A. Detail Tables
    # ---------------------------
    detail_metrics = {
        'time_s': 'Time(s)', 
        'gap': 'Gap', 
        'nodes': 'Nodes', 
        'presolve_time': 'Presolve',
        'percent_cs_delivery': '% Del',
        'integrality_gap_root': 'Root Gap',
        'time_to_best_s': 'TTB'
    }
    
    cols_to_use = [c for c in detail_metrics.keys() if c in df_agg.columns]
    
    for solver_orig, solver_clean in solver_map.items():
        subset = df_agg[df_agg['solver'] == solver_orig]
        if subset.empty: continue
        
        tbl = subset.groupby(['sort_P', 'N_Group', 'Disp_P', 'Disp_K'], sort=True)[cols_to_use].mean().reset_index()
        tbl = tbl.drop(columns=['sort_P']).rename(columns={'Disp_P': '|P|', 'Disp_K': '|K|', 'N_Group': '|N|'})
        tbl = tbl.rename(columns=detail_metrics)

        cols_final = ['|N|', '|P|', '|K|'] + [detail_metrics[c] for c in cols_to_use]
        tbl = tbl[cols_final]

        print_pretty_table(f"Analysis: {solver_clean}", tbl)
        save_table_as_image(tbl.round(2), f"table_detail_{solver_clean}.png")

    # ---------------------------
    # B. Comparison Summary (Time & Gap)
    # ---------------------------
    pivot = df_agg.pivot_table(
        index=['sort_P', 'N_Group', 'Disp_P', 'Disp_K'], 
        columns='solver_short', 
        values=['time_s', 'gap'], 
        aggfunc='mean'
    )
    
    # 1. Flattening
    pivot.columns = [f"{c[0]}_{c[1]}" for c in pivot.columns]

    # 2. Reset Index and order rows
    pivot = pivot.reset_index().sort_values(by=['sort_P', 'N_Group'])
    
    # 3. Rename index columns
    pivot = pivot.drop(columns=['sort_P']).rename(columns={'Disp_P': '|P|', 'Disp_K': '|K|', 'N_Group': 'Size (N)'})

    new_cols = []
    for c in pivot.columns:
        if 'time_s' in c:
            new_cols.append(c.replace('time_s', 'Time (s)'))
        elif 'gap' in c:
            new_cols.append(c.replace('gap', 'Gap (%)'))
        else:
            new_cols.append(c)
    pivot.columns = new_cols

    cols_time = sorted([c for c in pivot.columns if 'Time' in c])
    cols_gap  = sorted([c for c in pivot.columns if 'Gap' in c])
    
    # final order
    final_cols = ['Size (N)', '|P|', '|K|'] + cols_time + cols_gap

    pivot = pivot[final_cols]
    save_hierarchical_table(pivot, "table_milp_comparison_summary.png", "Performance Comparison")

    # ---------------------------
    # C. Technical Comparison (Hierarchical)
    # ---------------------------
    tech_cols = ['nodes', 'integrality_gap_root', 'presolve_time', 'time_to_best_s']
    tech_cols = [c for c in tech_cols if c in df_agg.columns]
    
    if tech_cols:
        pivot_tech = df_agg.pivot_table(
            index=['sort_P', 'N_Group', 'Disp_P', 'Disp_K'], 
            columns='solver_short', 
            values=tech_cols, 
            aggfunc='mean'
        )
        
        # Flattening 
        pivot_tech.columns = [f"{c[0]}_{c[1]}" for c in pivot_tech.columns]
        
        pivot_tech = pivot_tech.reset_index().sort_values(by=['sort_P', 'N_Group'])
        pivot_tech = pivot_tech.drop(columns=['sort_P']).rename(columns={'Disp_P': '|P|', 'Disp_K': '|K|', 'N_Group': 'Size (N)'})
        
        # grouping for metric
        sorted_cols = sorted([c for c in pivot_tech.columns if c not in ['Size (N)', '|P|', '|K|']])
        priority = ['presolve_time', 'nodes', 'integrality_gap_root', 'time_to_best_s']
        
        final_col_order = ['Size (N)', '|P|', '|K|']
        for p in priority:
            final_col_order.extend([c for c in sorted_cols if c.startswith(p)])
        
        final_col_order = [c for c in final_col_order if c in pivot_tech.columns]
        pivot_tech = pivot_tech[final_col_order]
        save_hierarchical_table(pivot_tech, "table_technical_compact.png", "Technical Metrics Analysis")


    # ---------------------------
    # D. Cuts Analysis (Grouped)
    # ---------------------------
    cut_cols = [c for c in df.columns if c.startswith('cuts_VI')]
    if cut_cols:
        df_cuts = df_agg[df_agg['solver'] == 'MILP_Callback'].copy()
        if not df_cuts.empty:
            cut_summary = df_cuts.groupby(['sort_P', 'N_Group', 'Disp_P', 'Disp_K'], sort=True)[cut_cols].mean().reset_index()
            
            vi_groups = {
                'VI1': ['cuts_VI1'],
                'VI2-VI3': ['cuts_VI2', 'cuts_VI3'],
                'VI4-VI6':  ['cuts_VI4', 'cuts_VI5', 'cuts_VI6'],
                'VI7-VI9': ['cuts_VI7', 'cuts_VI8', 'cuts_VI9']
            }
            
            for group_name, cols in vi_groups.items():
                valid = [c for c in cols if c in cut_summary.columns]
                if valid:
                    cut_summary[group_name] = cut_summary[valid].sum(axis=1)

            cut_summary['Total'] = cut_summary[cut_cols].sum(axis=1)
            cut_summary = cut_summary.drop(columns=['sort_P']).rename(columns={'Disp_P': '|P|', 'Disp_K': '|K|', 'N_Group': '|N|'})

            base_cols = ['|N|', '|P|', '|K|']
            group_cols = [g for g in vi_groups.keys() if g in cut_summary.columns]
            ordered_cols = base_cols + group_cols + ['Total'] 

            final_cols = [c for c in ordered_cols if c in cut_summary.columns]
            cut_summary = cut_summary[final_cols]
            
            print_pretty_table("Cuts Analysis", cut_summary)
            cut_summary.round(1).to_csv(os.path.join(OUT_DIR, "cuts_vi_analysis.csv"), index=False)
            save_table_as_image(cut_summary.round(1), "table_cuts_vi_analysis.png")

# =============================================================================
# 4. PLOTS
# =============================================================================
def generate_plots(df):
    print("Generating Plots...")
    df_plot = df.copy()

    df_plot = df[df['|P|'] < 160].copy()

    if df_plot.empty:
        print("Warning: No data available for plots after filtering P < 160.")
        return
    df_plot['N_numeric'] = pd.to_numeric(df_plot['N_Group'])
    df_N_agg = df_plot.groupby(['N_numeric', 'solver'])['time_s'].mean().reset_index()
    # Plot 1: Scalability vs N
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_N_agg, x='N_numeric', y='time_s', hue='solver', style='solver', 
                 marker='o', markersize=9, linewidth=3, errorbar=None)
    plt.yscale('log')
    plt.xticks([24, 36, 44])
    plt.axhline(TIME_LIMIT, linestyle='--', color='red', alpha=0.5, label='Time Limit')
    plt.title("Scalability vs Network Size (|N|)", fontweight='bold')
    plt.ylabel("Avg Time (s) (log-scale)")
    plt.xlabel("Network Size (|N|)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scalability_vs_N_clean.png"), dpi=200)
    plt.close()

    # Plot 2: Scalability vs P,K (Small Instances)
    df_pk = df_plot[(df_plot['|P|'] < 160)].copy()
    if df_pk.empty: return
 
    solver_map = {'MILP_Base': 'Simple', 'MILP_Cuts': 'Cuts', 'MILP_Callback': 'Callback'}
    df_pk['solver'] = df_pk['solver'].map(solver_map).fillna(df_pk['solver'])

    df_pk['PK_Label'] = df_pk.apply(lambda x: f"P={int(x['|P|'])}\nK={int(x['|K|'])}", axis=1)
    df_pk = df_pk.sort_values(by=['|P|', '|K|'])
    
    plt.figure(figsize=(12, 7))
    ax=sns.pointplot(
            data=df_pk, 
            x='PK_Label',
            y='time_s', 
            hue='solver',  
            dodge=0.25,       # <--- not to overlap
            errorbar='sd',    # <--- std
            capsize=.1,       # <--- bar style
            markers=['o', 's', '^'], # Marker for solver
            linestyles=['-', '--', '-.'], # Styles linea 
            )

    ADD_VAR_LABELS = True #to add num_var and num_constr to the plot
    
    if ADD_VAR_LABELS:
        model_stats = df_pk.groupby('PK_Label')[['num_vars', 'num_constrs']].mean()
        
        unique_labels = df_pk['PK_Label'].unique()
        model_stats = model_stats.reindex(unique_labels)
        
        for idx, label in enumerate(unique_labels):
            if label in model_stats.index:
                row = model_stats.loc[label]
                # Check se nan
                if pd.isna(row['num_vars']): continue

                n_vars = int(row['num_vars'])
                n_cons = int(row['num_constrs'])
                
                txt_v = f"{n_vars/1000:.1f}k" if n_vars > 1000 else str(n_vars)
                txt_c = f"{n_cons/1000:.1f}k" if n_cons > 1000 else str(n_cons)
                info_text = f"Vars: {txt_v}\nCons: {txt_c}"
                
                ax.text(
                    idx, 0.95, info_text, 
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='top', 
                    fontsize=10, color='#444444', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8)
                )

    plt.axhline(TIME_LIMIT, linestyle='--', color='red', alpha=0.5, label='Time Limit')
    plt.title("Impact of Load (P, K) on Runtime (Log Scale)", fontweight='bold')
    plt.ylabel("Time (s) - Log Scale")
    plt.xlabel("Instance Class (Parcels, Crowdshippers)")
    plt.yscale('log')
    plt.xticks(rotation=0)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.legend(title='Method', loc='center left')
    plt.savefig(os.path.join(OUT_DIR, "scalability_vs_PK_variance.png"), dpi=200)
    plt.close()

    sns.set_context("paper", font_scale=1.4)

# ============================================================================
#PLOT WITH SHADOWS FOR VARIANCE
# ============================================================================
def plot_scalability_with_var(df_input):
    print(" Generating Scalability Plots (Small + Large)...")

    df_plot = df_input.copy()
    solver_map = {'MILP_Base': 'Simple', 'MILP_Cuts': 'Cuts', 'MILP_Callback': 'Callback'}
    df_plot['solver'] = df_plot['solver'].map(solver_map).fillna(df_plot['solver'])

    def create_label(row):
        p_val = row['|P|']
        k_val = row['|K|']
        
        if p_val >= 160:
            return "Large\n(≥160)"
        else:
            return f"P={int(p_val)}\nK={int(k_val)}"

    df_plot['Instance_Label'] = df_plot.apply(create_label, axis=1)

    df_plot['sort_key_P'] = df_plot['|P|']
    df_plot['sort_key_K'] = df_plot['|K|']
    

    df_plot = df_plot.sort_values(by=['sort_key_P', 'sort_key_K'])

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)


    plt.figure(figsize=(14, 8))
    
    sns.lineplot(
        data=df_plot, 
        x='Instance_Label', 
        y='time_s', 
        hue='solver', 
        style='solver', 
        markers=True, 
        dashes=False, 
        linewidth=3,
        sort=False 
    )
    
    plt.axhline(y=TIME_LIMIT, color='red', linestyle='--', alpha=0.6, label='Time Limit')
    
    plt.yscale('log') 
    plt.title('Scalability Trend (Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Instance Size', fontsize=13)
    plt.ylabel('Time (s) - Log Scale', fontsize=13)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(title='Method')
    
    filename_line = "scalability_trend_all.png"
    plt.savefig(os.path.join(OUT_DIR, filename_line), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"   -> Plot saved: {filename_line}")

    # Plot 3: Faceted Barplot (Aggregated)
    def get_load_label(row):
        p_val = int(row['|P|'])
        k_val = int(row['|K|'])
        if p_val >= 160: return "Large\n(≥160)"
        return f"P={p_val}\nK={k_val}"
    
    df_plot['Load'] = df_plot.apply(get_load_label, axis=1)
    df_plot['sort_helper'] = df_plot['|P|'].apply(lambda x: 9999 if x >= 160 else x)
    df_plot = df_plot.sort_values(by=['sort_helper', '|K|'])

    g = sns.catplot(data=df_plot, kind="bar", x="Load", y="time_s", hue="solver", 
                    col="|N|", col_wrap=3, height=5, aspect=1.1, 
                    palette="viridis", edgecolor="black", errorbar=None)
    
    g.fig.suptitle("Impact of Parcel Density (P) at fixed Network Size (N)", y=1.02, fontweight='bold')
    
    for ax in g.axes.flat:
        ax.axhline(TIME_LIMIT, c='r', ls='--', lw=1.5)
        ax.set_yscale('log')
        ax.set_ylim(1, TIME_LIMIT * 1.5)

    g.savefig(os.path.join(OUT_DIR, "impact_of_load_P.png"), bbox_inches='tight', dpi=150)
    plt.close()

# MAIN
if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        generate_tables(df)
        generate_plots(df)
        print("\n Section 1 Analysis Completed.")
        plot_scalability_with_var(df)
        print(f"   Outputs saved in: {OUT_DIR}")
    else: print(" No data found.")