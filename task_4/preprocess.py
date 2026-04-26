import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# ── 1.1 Load & melt wide → long ──────────────────────────────────────────────
raw = pd.read_csv(HERE / "global_inflation_data.csv", encoding="utf-8-sig")

year_cols = [c for c in raw.columns if c.isdigit()]
id_vars   = ["country_name", "indicator_name"]

long = raw.melt(
    id_vars=id_vars,
    value_vars=year_cols,
    var_name="Year",
    value_name="Inflation",
)
long.columns = ["Country", "Indicator", "Year", "Inflation"]

# ── 1.2 Basic cleaning ────────────────────────────────────────────────────────
target_indicator = "Annual average inflation (consumer prices) rate"
long = long[long["Indicator"] == target_indicator].copy()

long["Year"]      = long["Year"].astype(int)
long["Inflation"] = pd.to_numeric(long["Inflation"], errors="coerce")

missing_count = long["Inflation"].isna().sum()
print(f"Missing Inflation values: {missing_count}")

# ── 1.3 Outlier capping (map only) ───────────────────────────────────────────
cap_99 = long["Inflation"].quantile(0.99)
print(f"99th percentile cap: {cap_99:.2f}")

long["Inflation_Map_Capped"] = long["Inflation"].clip(upper=cap_99)

# ── 1.4 Country-level summary ─────────────────────────────────────────────────
def safe_mean(series, y0, y1):
    subset = series[(series.index >= y0) & (series.index <= y1)]
    return subset.mean() if not subset.empty else np.nan

def safe_max(series, y0, y1):
    subset = series[(series.index >= y0) & (series.index <= y1)]
    return subset.max() if not subset.empty else np.nan

records = []
for country, grp in long.groupby("Country"):
    s = grp.set_index("Year")["Inflation"].dropna()

    pre_covid_avg      = safe_mean(s, 2017, 2019)
    covid_avg          = safe_mean(s, 2020, 2022)
    covid_impact_main  = covid_avg - pre_covid_avg
    covid_impact_peak  = safe_max(s, 2020, 2022) - pre_covid_avg

    covid_window = s[(s.index >= 2020) & (s.index <= 2022)]
    peak_covid_year = int(covid_window.idxmax()) if not covid_window.empty and covid_window.notna().any() else np.nan

    long_run_mean = safe_mean(s, 1980, 2024)
    long_run_max  = safe_max(s,  1980, 2024)

    records.append({
        "Country":           country,
        "Pre_COVID_Avg":     pre_covid_avg,
        "COVID_Avg":         covid_avg,
        "COVID_Impact_Main": covid_impact_main,
        "COVID_Impact_Peak": covid_impact_peak,
        "Peak_COVID_Year":   peak_covid_year,
        "Long_Run_Mean":     long_run_mean,
        "Long_Run_Max":      long_run_max,
    })

summary = pd.DataFrame(records)

# ── 1.5 标记 Hyperinflation Outlier（仅用于显示层，不删除数据）────────────────
HYPERINFLATION_THRESHOLD = 1000  # Pre_COVID_Avg 超过此值视为 hyperinflation outlier

def label_outlier(row):
    pre = row["Pre_COVID_Avg"]
    if pd.isna(pre):
        return 0, ""
    if pre > HYPERINFLATION_THRESHOLD:
        return 1, f"Pre-COVID avg inflation {pre:,.0f}% — hyperinflation outlier, excluded from main chart for readability"
    return 0, ""

summary[["Hyperinflation_Outlier", "Outlier_Note"]] = summary.apply(
    label_outlier, axis=1, result_type="expand"
)

outliers = summary[summary["Hyperinflation_Outlier"] == 1][["Country", "Pre_COVID_Avg", "COVID_Avg", "COVID_Impact_Main"]]
print(f"\nHyperinflation outliers (Pre_COVID_Avg > {HYPERINFLATION_THRESHOLD}%):")
print(outliers.to_string(index=False))

# ── 1.6 Merge summary fields back into long ──────────────────────────────────
long = long.merge(
    summary[["Country", "Pre_COVID_Avg", "COVID_Avg", "Peak_COVID_Year"]],
    on="Country",
    how="left",
)

# ── 1.7 Output ────────────────────────────────────────────────────────────────
long.to_csv(HERE / "inflation_long.csv", index=False)
summary.to_csv(HERE / "inflation_country_summary.csv", index=False)

print(f"\nDone.")
print(f"  inflation_long.csv          → {len(long):,} rows")
print(f"  inflation_country_summary.csv → {len(summary):,} rows")
print(f"\nLong table sample:")
print(long.head())
print(f"\nSummary sample:")
print(summary.head())

print(f"\nCOVID Impact ranking (bottom 10):")
print(summary[['Country', 'COVID_Impact_Main']].sort_values('COVID_Impact_Main').head(10))
print(f"\nCOVID Impact ranking (top 10):")
print(summary[['Country', 'COVID_Impact_Main']].sort_values('COVID_Impact_Main').tail(10))