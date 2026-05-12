"""
analysis.py — Run after all 160 experiments complete.
Pulls data from the API and generates paper-ready charts + tables.

Batches covered:
  Batch 1-3  : Multi-seed HPA / MPC+LSTM / MPC+Prophet  (90 runs)
  Batch 4-6  : Cold-start sensitivity                    (30 runs)
  Batch 7    : FH-OPT A/B (use_fh_opt=false vs true)    (40 runs)

Usage:
    python analysis.py
    python analysis.py --api http://localhost:8000
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
# Only policies that actually have runs in paste.txt
POLICIES  = ["hpa", "lstm", "prophet"]
WORKLOADS = ["smooth", "bursty", "bimodal", "diurnal_burst", "flash_crowd", "slow_ramp_up"]
METRICS   = ["sla_pct", "avg_latency_ms", "avg_replicas", "total_cost"]

COLORS = {
    "hpa":            "#e7298a",
    "lstm":           "#1a9e77",
    "prophet":        "#7570b3",
    "lstm_fh_off":    "#1a9e77",
    "lstm_fh_on":     "#52c27e",
    "prophet_fh_off": "#7570b3",
    "prophet_fh_on":  "#b3a9e8",
}
LABELS = {
    "hpa":            "HPA (baseline)",
    "lstm":           "MPC+LSTM",
    "prophet":        "MPC+Prophet",
    "lstm_fh_off":    "LSTM  FH-OPT OFF",
    "lstm_fh_on":     "LSTM  FH-OPT ON",
    "prophet_fh_off": "Prophet  FH-OPT OFF",
    "prophet_fh_on":  "Prophet  FH-OPT ON",
}

OUT = Path("paper_figures")
OUT.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ci95(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=len(arr) - 1) * stats.sem(arr))


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ── 1. Fetch all completed runs ───────────────────────────────────────────────
def fetch_runs(api_base: str) -> pd.DataFrame:
    rows, offset, limit = [], 0, 200
    while True:
        r = requests.get(
            f"{api_base}/v1/runs",
            params={"limit": limit, "offset": offset, "status": "completed"},
        )
        r.raise_for_status()
        data = r.json()
        batch = data["runs"]
        if not batch:
            break
        rows.extend(batch)
        offset += limit
        if offset >= data["total"]:
            break

    records = []
    for run in rows:
        s = run.get("summary", {})
        if not s:
            continue

        policy     = run["policy"]
        forecaster = run["forecaster"]  # "none" / "lstm" / "prophet"
        use_fh_opt = run.get("use_fh_opt", False)

        # config key:  hpa  |  lstm  |  prophet
        # (multi-seed runs from Batch 1-3 — no fh_opt field)
        if policy == "hpa":
            config = "hpa"
        else:
            config = forecaster          # "lstm" or "prophet"

        # fh_key:  lstm_fh_off | lstm_fh_on | prophet_fh_off | prophet_fh_on
        # (only populated for Batch 7 rows)
        fh_key = None
        if policy == "mpc" and "use_fh_opt" in run:
            suffix = "fh_on" if use_fh_opt else "fh_off"
            fh_key = f"{forecaster}_{suffix}"

        records.append({
            "policy":        policy,
            "forecaster":    forecaster,
            "config":        config,
            "fh_key":        fh_key,
            "use_fh_opt":    use_fh_opt,
            "workload":      run["workload"],
            "seed":          run["seed"],
            "cold_start_s":  run["cold_start_s"],
            "sla_pct":       s.get("sla_pct", 0),
            "avg_latency_ms": s.get("avg_latency_ms", 0),
            "avg_replicas":  s.get("avg_replicas", 0),
            "total_cost":    s.get("total_cost", 0),
            "peak_replicas": s.get("peak_replicas", 0),
        })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} completed runs")
    print(f"  Configs:   {sorted(df['config'].unique())}")
    print(f"  Workloads: {sorted(df['workload'].unique())}")
    print(f"  fh_keys:   {sorted(df['fh_key'].dropna().unique())}")
    return df


# ── 2. Aggregate mean ± CI across seeds (Batch 1-3) ──────────────────────────
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed runs — excludes cold-start sensitivity & FH-OPT rows."""
    base = df[df["cold_start_s"] == 120.0].copy()
    # Exclude Batch 7 rows (they have fh_key set)
    base = base[base["fh_key"].isna()]

    grp = base.groupby(["config", "workload"])
    rows = []
    for (cfg, wl), g in grp:
        row = {"config": cfg, "workload": wl, "n_seeds": len(g)}
        for m in METRICS:
            row[f"{m}_mean"]   = g[m].mean()
            row[f"{m}_std"]    = g[m].std()
            row[f"{m}_ci95"]   = ci95(g[m].values)
            row[f"{m}_median"] = g[m].median()
        rows.append(row)
    return pd.DataFrame(rows)


# ── 3. Table 1 — SLA% mean ± CI across workloads ─────────────────────────────
def table_sla(agg: pd.DataFrame):
    configs = [c for c in POLICIES if c in agg["config"].unique()]
    wls     = [w for w in WORKLOADS if w in agg["workload"].unique()]

    pivot_mean = agg.pivot(index="config", columns="workload", values="sla_pct_mean")
    pivot_ci   = agg.pivot(index="config", columns="workload", values="sla_pct_ci95")
    pivot_mean = pivot_mean.reindex(index=configs, columns=wls)
    pivot_ci   = pivot_ci.reindex_like(pivot_mean)

    display = pivot_mean.copy().astype(object)
    for r in display.index:
        for c in display.columns:
            m  = pivot_mean.loc[r, c]
            ci = pivot_ci.loc[r, c]
            display.loc[r, c] = "—" if pd.isna(m) else f"{m:.2f} ± {ci:.2f}"

    display.index   = [LABELS.get(i, i) for i in display.index]
    display.columns = [w.replace("_", " ").title() for w in display.columns]
    print("\n=== TABLE 1: SLA Violation % (mean ± 95% CI) ===")
    print(display.to_string())
    display.to_csv(OUT / "table1_sla.csv")


# ── 4. Figure 1 — SLA grouped bar chart ──────────────────────────────────────
def fig_sla_bars(agg: pd.DataFrame):
    configs = [c for c in POLICIES if c in agg["config"].unique()]
    wls     = [w for w in WORKLOADS if w in agg["workload"].unique()]
    n_wl, n_cfg = len(wls), len(configs)
    width = 0.8 / n_cfg
    x     = np.arange(n_wl)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cfg in enumerate(configs):
        sub   = agg[agg["config"] == cfg].set_index("workload")
        means = [sub.loc[w, "sla_pct_mean"] if w in sub.index else np.nan for w in wls]
        cis   = [sub.loc[w, "sla_pct_ci95"] if w in sub.index else 0      for w in wls]
        offset = (i - n_cfg / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9,
               label=LABELS[cfg], color=COLORS[cfg], alpha=0.85)
        ax.errorbar(x + offset, means, yerr=cis,
                    fmt="none", color="black", capsize=3, linewidth=1.2)

    ax.axhline(5, color="red", linestyle="--", linewidth=1.5, label="5% SLA target")
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ").title() for w in wls], fontsize=11)
    ax.set_ylabel("SLA Violation %", fontsize=13)
    ax.set_xlabel("Workload", fontsize=13)
    ax.set_title("SLA Violation % by Policy & Workload\n(mean ± 95% CI across 5 seeds)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    despine(ax)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_sla_bars.png", dpi=180)
    plt.close()
    print("Saved fig1_sla_bars.png")


# ── 5. Figure 2 — Cold-start sensitivity (all 3 policies) ────────────────────
def fig_coldstart(df: pd.DataFrame):
    cs_runs = df[df["cold_start_s"].isin([30, 60, 120, 180, 300])].copy()
    cs_runs = cs_runs[cs_runs["fh_key"].isna()]   # exclude FH-OPT rows
    if cs_runs.empty:
        print("No cold-start sensitivity data — skipping fig2")
        return

    grp = cs_runs.groupby(["config", "cold_start_s"])["sla_pct"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg in ["hpa", "lstm", "prophet"]:
        sub = grp[grp["config"] == cfg]
        if sub.empty:
            continue
        ax.plot(sub["cold_start_s"], sub["mean"], marker="o",
                color=COLORS[cfg], label=LABELS[cfg], linewidth=2)
        ax.fill_between(sub["cold_start_s"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        color=COLORS[cfg], alpha=0.15)

    ax.axhline(5, color="red", linestyle="--", linewidth=1.5, label="5% SLA target")
    ax.set_xlabel("Cold Start Duration (s)", fontsize=13)
    ax.set_ylabel("SLA Violation %", fontsize=13)
    ax.set_title("Cold-Start Sensitivity: SLA Violation vs Cold Start Duration",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    despine(ax)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_coldstart.png", dpi=180)
    plt.close()
    print("Saved fig2_coldstart.png")


# ── 6. Figure 3 — SLA heatmap ─────────────────────────────────────────────────
def fig_heatmap(agg: pd.DataFrame):
    configs = [c for c in POLICIES if c in agg["config"].unique()]
    wls     = [w for w in WORKLOADS if w in agg["workload"].unique()]
    pivot   = agg.pivot(index="config", columns="workload", values="sla_pct_mean")
    pivot   = pivot.reindex(index=configs, columns=wls)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=30)
    ax.set_xticks(range(len(wls)))
    ax.set_yticks(range(len(configs)))
    ax.set_xticklabels([w.replace("_", " ").title() for w in wls], fontsize=11)
    ax.set_yticklabels([LABELS.get(c, c) for c in configs], fontsize=11)
    for i in range(len(configs)):
        for j in range(len(wls)):
            val = pivot.iloc[i, j]
            txt = f"{val:.1f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, color="black" if val < 15 else "white")
    plt.colorbar(im, ax=ax, label="SLA Violation %")
    ax.set_title("SLA Violation % Heatmap — Policy × Workload (mean across 5 seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_heatmap.png", dpi=180)
    plt.close()
    print("Saved fig3_heatmap.png")


# ── 7. Figure 4 — FH-OPT A/B comparison (THE main contribution chart) ────────
def fig_fhopt_ab(df: pd.DataFrame):
    """
    Grouped bar chart: FH-OPT OFF vs ON, for LSTM and Prophet,
    across diurnal_burst and flash_crowd — 5 seeds each.
    Shows SLA%, avg_latency_ms, total_cost side-by-side.
    """
    fh_df = df[df["fh_key"].notna()].copy()
    if fh_df.empty:
        print("No FH-OPT A/B data (Batch 7) — skipping fig4")
        return

    fh_keys  = ["lstm_fh_off", "lstm_fh_on", "prophet_fh_off", "prophet_fh_on"]
    wls      = ["diurnal_burst", "flash_crowd"]
    metrics  = [("sla_pct", "SLA Violation %"),
                ("avg_latency_ms", "Avg Latency (ms)"),
                ("total_cost", "Total Cost")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel) in zip(axes, metrics):
        grp = fh_df.groupby(["fh_key", "workload"])[metric].agg(["mean", "std"]).reset_index()
        n_wl, n_key = len(wls), len(fh_keys)
        width = 0.8 / n_key
        x     = np.arange(n_wl)

        for i, fk in enumerate(fh_keys):
            sub   = grp[grp["fh_key"] == fk].set_index("workload")
            means = [sub.loc[w, "mean"] if w in sub.index else np.nan for w in wls]
            stds  = [sub.loc[w, "std"]  if w in sub.index else 0      for w in wls]
            offset = (i - n_key / 2 + 0.5) * width
            style  = "--" if "fh_on" in fk else "-"
            ax.bar(x + offset, means, width * 0.9,
                   label=LABELS[fk], color=COLORS[fk], alpha=0.85,
                   linestyle=style, edgecolor="white")
            ax.errorbar(x + offset, means, yerr=stds,
                        fmt="none", color="black", capsize=3, linewidth=1.0)

        if metric == "sla_pct":
            ax.axhline(5, color="red", linestyle="--", linewidth=1.2, label="5% target")
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace("_", " ").title() for w in wls], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        despine(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("FH-OPT OFF vs ON: SLA, Latency, Cost\n(mean ± std across 5 seeds × 2 workloads)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_fhopt_ab.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved fig4_fhopt_ab.png")


# ── 8. Figure 5 — FH-OPT SLA improvement per workload (delta chart) ──────────
def fig_fhopt_delta(df: pd.DataFrame):
    """
    Shows SLA delta (FH-OFF minus FH-ON) per forecaster per workload.
    Positive = FH-OPT improved SLA. This is the clearest single-panel proof.
    """
    fh_df = df[df["fh_key"].notna()].copy()
    if fh_df.empty:
        print("No FH-OPT A/B data — skipping fig5")
        return

    wls = ["diurnal_burst", "flash_crowd"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, forecaster in zip(axes, ["lstm", "prophet"]):
        off_key = f"{forecaster}_fh_off"
        on_key  = f"{forecaster}_fh_on"
        grp = fh_df.groupby(["fh_key", "workload"])["sla_pct"].agg(["mean", "std"]).reset_index()

        deltas, errs = [], []
        for wl in wls:
            off_row = grp[(grp["fh_key"] == off_key) & (grp["workload"] == wl)]
            on_row  = grp[(grp["fh_key"] == on_key)  & (grp["workload"] == wl)]
            if off_row.empty or on_row.empty:
                deltas.append(np.nan); errs.append(0)
            else:
                delta = float(off_row["mean"].values[0]) - float(on_row["mean"].values[0])
                err   = np.sqrt(float(off_row["std"].values[0])**2 +
                                float(on_row["std"].values[0])**2)
                deltas.append(delta); errs.append(err)

        colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
        bars = ax.bar([w.replace("_", " ").title() for w in wls], deltas,
                      color=colors, alpha=0.85, width=0.5)
        ax.errorbar(range(len(wls)), deltas, yerr=errs,
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_title(f"MPC+{forecaster.upper()}: SLA Δ (OFF − ON)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("SLA Improvement (pp)", fontsize=11)
        ax.text(0.5, 0.97, "↑ Positive = FH-OPT reduced violations",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="gray")
        despine(ax)

        # annotate bars
        for bar, d in zip(bars, deltas):
            if not np.isnan(d):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + 0.05 * max(abs(d), 0.1),
                        f"{d:+.2f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("FH-OPT Improvement: SLA Violation Reduction (OFF minus ON)\n"
                 "Green = FH-OPT wins, Red = no improvement",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig5_fhopt_delta.png", dpi=180)
    plt.close()
    print("Saved fig5_fhopt_delta.png")


# ── 9. Statistical test table — FH-OPT paired Wilcoxon ──────────────────────
def table_fhopt_stats(df: pd.DataFrame):
    """
    Paired Wilcoxon signed-rank test: FH-OFF vs FH-ON per forecaster × workload.
    p < 0.05 confirms FH-OPT is statistically significant.
    """
    fh_df = df[df["fh_key"].notna()].copy()
    if fh_df.empty:
        print("No FH-OPT data for statistical test")
        return

    rows = []
    for forecaster in ["lstm", "prophet"]:
        for wl in ["diurnal_burst", "flash_crowd"]:
            off = fh_df[(fh_df["fh_key"] == f"{forecaster}_fh_off") &
                        (fh_df["workload"] == wl)]["sla_pct"].values
            on  = fh_df[(fh_df["fh_key"] == f"{forecaster}_fh_on") &
                        (fh_df["workload"] == wl)]["sla_pct"].values
            if len(off) < 2 or len(on) < 2:
                continue
            # align by seed if possible, else use raw arrays
            n = min(len(off), len(on))
            try:
                stat, p = stats.wilcoxon(off[:n], on[:n])
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "Forecaster":   forecaster.upper(),
                "Workload":     wl.replace("_", " ").title(),
                "SLA OFF mean": f"{off.mean():.2f}%",
                "SLA ON mean":  f"{on.mean():.2f}%",
                "Δ (pp)":       f"{off.mean() - on.mean():+.2f}",
                "W statistic":  f"{stat:.1f}" if not np.isnan(stat) else "—",
                "p-value":      f"{p:.4f}" if not np.isnan(p) else "—",
                "Significant":  "✓" if (not np.isnan(p) and p < 0.05) else "✗",
            })

    tbl = pd.DataFrame(rows)
    print("\n=== TABLE 2: FH-OPT Statistical Significance (Wilcoxon signed-rank) ===")
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT / "table2_fhopt_stats.csv", index=False)


# ── 10. Summary table ─────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame):
    # Only Batch 1-3 (no cold-start variants, no FH-OPT rows)
    base = df[(df["cold_start_s"] == 120.0) & (df["fh_key"].isna())]
    grp  = base.groupby("config")
    rows = []
    for cfg, g in grp:
        rows.append({
            "Policy":       LABELS.get(cfg, cfg),
            "N runs":       len(g),
            "N seeds":      g["seed"].nunique(),
            "SLA mean %":   f"{g['sla_pct'].mean():.2f}",
            "SLA std":      f"{g['sla_pct'].std():.2f}",
            "SLA 95% CI":   f"±{ci95(g['sla_pct'].values):.2f}",
            "Latency mean": f"{g['avg_latency_ms'].mean():.1f}",
            "Cost mean":    f"{g['total_cost'].mean():.0f}",
            "AvgReplicas":  f"{g['avg_replicas'].mean():.2f}",
        })
    tbl = pd.DataFrame(rows).set_index("Policy")
    print("\n=== SUMMARY TABLE (Batch 1-3, all workloads) ===")
    print(tbl.to_string())
    tbl.to_csv(OUT / "summary_table.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    df = fetch_runs(args.api)
    if df.empty:
        print("No completed runs found. Run experiments first.")
        sys.exit(1)

    df.to_csv(OUT / "all_runs_raw.csv", index=False)
    print(f"Raw data saved to {OUT}/all_runs_raw.csv")

    agg = aggregate(df)
    agg.to_csv(OUT / "aggregated.csv", index=False)

    # Tables
    table_sla(agg)
    summary_table(df)
    table_fhopt_stats(df)

    # Figures
    fig_sla_bars(agg)      # Fig 1: Main SLA comparison
    fig_coldstart(df)      # Fig 2: Cold-start sensitivity
    fig_heatmap(agg)       # Fig 3: SLA heatmap
    fig_fhopt_ab(df)       # Fig 4: FH-OPT A/B grouped bars  ← NEW
    fig_fhopt_delta(df)    # Fig 5: FH-OPT delta chart       ← NEW

    print(f"\nAll outputs saved to ./{OUT}/")
    print("Files:", sorted(f.name for f in OUT.iterdir()))
