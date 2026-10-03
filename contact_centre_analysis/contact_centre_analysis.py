"""Contact-centre KPI analysis with reproducible synthetic demo data."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = ["date", "offered", "answered", "abandoned", "sla_answered",
            "aht_seconds", "asa_seconds", "csat_score", "repeat_calls",
            "agent_hours", "available_hours"]
COUNTS = ["offered", "answered", "abandoned", "sla_answered", "repeat_calls"]
NONNEG = COUNTS + ["aht_seconds", "asa_seconds", "agent_hours", "available_hours"]
LOG = logging.getLogger("contact_centre")

def synthetic_data(days=90, seed=42):
    """Create illustrative data only. Never treat these figures as real performance."""
    if days < 7:
        raise ValueError("days must be at least 7")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    weekday = np.where(dates.dayofweek >= 5, .62, 1.0)
    offered = np.maximum(1, np.round(rng.normal(2400, 180, days) * weekday *
                                      np.linspace(.96, 1.06, days))).astype(int)
    abandon_rate = np.clip(rng.normal(.055, .012, days), .015, .12)
    abandoned = np.round(offered * abandon_rate).astype(int)
    answered = offered - abandoned
    sla_answered = np.minimum(answered, np.round(
        answered * np.clip(rng.normal(.79, .055, days), .55, .96))).astype(int)
    aht = np.clip(rng.normal(610, 45, days), 450, 800).round(1)
    asa = np.clip(rng.normal(95, 28, days) + abandon_rate * 220, 15, 240).round(1)
    csat = np.clip(rng.normal(4.25, .22, days) - (asa - 80) / 500, 3.2, 4.9).round(2)
    repeat = np.minimum(answered, np.round(answered *
        np.clip(rng.normal(.105, .018, days), .04, .18))).astype(int)
    agent_hours = np.maximum(1, offered * rng.normal(.245, .018, days))
    available_hours = np.maximum(agent_hours, offered * rng.normal(.31, .02, days))
    return pd.DataFrame({
        "date": dates, "offered": offered, "answered": answered,
        "abandoned": abandoned, "sla_answered": sla_answered,
        "aht_seconds": aht, "asa_seconds": asa, "csat_score": csat,
        "repeat_calls": repeat, "agent_hours": agent_hours.round(1),
        "available_hours": available_hours.round(1)})

def load_data(input_file=None, days=90, seed=42):
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_csv(path)
    else:
        LOG.warning("Using synthetic demonstration data, not real operational results.")
        df = synthetic_data(days, seed)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED).sort_values("date")
    if df.empty:
        raise ValueError("No valid rows remain after parsing dates and numeric fields.")
    if (df[NONNEG] < 0).any().any():
        raise ValueError("Counts, durations and hours cannot be negative.")
    if (df[COUNTS] % 1 != 0).any().any():
        raise ValueError("Call count fields must be whole numbers.")
    if ((df.csat_score < 1) | (df.csat_score > 5)).any():
        raise ValueError("csat_score must be between 1 and 5.")
    for col in ["answered", "abandoned"]:
        if (df[col] > df.offered).any():
            raise ValueError(f"{col} cannot exceed offered.")
    if (df.answered + df.abandoned > df.offered).any():
        raise ValueError("answered + abandoned cannot exceed offered.")
    if (df.sla_answered > df.answered).any() or (df.repeat_calls > df.answered).any():
        raise ValueError("sla_answered and repeat_calls cannot exceed answered.")
    if df.date.duplicated().any():
        LOG.warning("Duplicate dates found. Aggregating daily rows.")
        df = df.groupby("date", as_index=False).agg({
            "offered":"sum", "answered":"sum", "abandoned":"sum",
            "sla_answered":"sum", "aht_seconds":"mean", "asa_seconds":"mean",
            "csat_score":"mean", "repeat_calls":"sum",
            "agent_hours":"sum", "available_hours":"sum"})
    return df.reset_index(drop=True)

def analyse(df):
    daily = df.copy()
    daily["answer_rate_pct"] = np.where(daily.offered > 0, daily.answered / daily.offered * 100, np.nan)
    daily["abandon_rate_pct"] = np.where(daily.offered > 0, daily.abandoned / daily.offered * 100, np.nan)
    daily["service_level_pct"] = np.where(daily.answered > 0, daily.sla_answered / daily.answered * 100, np.nan)
    daily["repeat_rate_pct"] = np.where(daily.answered > 0, daily.repeat_calls / daily.answered * 100, np.nan)
    daily["occupancy_proxy_pct"] = np.where(daily.available_hours > 0,
        daily.agent_hours / daily.available_hours * 100, np.nan)
    daily["offered_7d_avg"] = daily.offered.rolling(7, min_periods=1).mean()
    daily["asa_7d_avg"] = daily.asa_seconds.rolling(7, min_periods=1).mean()
    offered, answered = df.offered.sum(), df.answered.sum()
    summary = pd.DataFrame([{
        "period_start": df.date.min().date().isoformat(),
        "period_end": df.date.max().date().isoformat(),
        "days": int(df.date.nunique()), "offered_calls": int(offered),
        "answered_calls": int(answered), "abandoned_calls": int(df.abandoned.sum()),
        "answer_rate_pct": answered / offered * 100 if offered else np.nan,
        "abandon_rate_pct": df.abandoned.sum() / offered * 100 if offered else np.nan,
        "service_level_pct": df.sla_answered.sum() / answered * 100 if answered else np.nan,
        "weighted_aht_seconds": np.average(df.aht_seconds, weights=df.answered) if answered else np.nan,
        "weighted_asa_seconds": np.average(df.asa_seconds, weights=df.answered) if answered else np.nan,
        "average_csat": df.csat_score.mean(),
        "repeat_rate_pct": df.repeat_calls.sum() / answered * 100 if answered else np.nan,
        "occupancy_proxy_pct": df.agent_hours.sum() / df.available_hours.sum() * 100
            if df.available_hours.sum() else np.nan}])
    weekly = df.set_index("date").resample("W-MON").agg({
        "offered":"sum", "answered":"sum", "abandoned":"sum", "sla_answered":"sum",
        "repeat_calls":"sum", "aht_seconds":"mean", "asa_seconds":"mean",
        "csat_score":"mean", "agent_hours":"sum", "available_hours":"sum"}).reset_index()
    weekly["abandon_rate_pct"] = np.where(weekly.offered > 0, weekly.abandoned / weekly.offered * 100, np.nan)
    weekly["service_level_pct"] = np.where(weekly.answered > 0, weekly.sla_answered / weekly.answered * 100, np.nan)
    weekly["repeat_rate_pct"] = np.where(weekly.answered > 0, weekly.repeat_calls / weekly.answered * 100, np.nan)
    return daily, weekly, summary

def charts(daily, out):
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("daily_volume.png", ["offered", "offered_7d_avg"], "Daily contact volume", "Calls"),
        ("service_performance.png", ["abandon_rate_pct", "service_level_pct"], "Service performance", "Percent"),
        ("average_speed_of_answer.png", ["asa_seconds", "asa_7d_avg"], "Average speed of answer", "Seconds")]
    for filename, cols, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(11, 5))
        for col in cols:
            ax.plot(daily.date, daily[col], label=col.replace("_", " "))
        ax.set(title=title, xlabel="Date", ylabel=ylabel)
        ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(out / filename, dpi=150); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyse daily contact-centre KPIs.")
    parser.add_argument("--input", help="CSV file with the documented columns.")
    parser.add_argument("--days", type=int, default=90, help="Synthetic history length (default 90).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input, args.days, args.seed)
    daily, weekly, summary = analyse(df)
    df.to_csv(out / "cleaned_daily_data.csv", index=False)
    daily.to_csv(out / "daily_kpis.csv", index=False)
    weekly.to_csv(out / "weekly_kpis.csv", index=False)
    summary.to_csv(out / "summary_kpis.csv", index=False)
    charts(daily, out)
    k = summary.iloc[0]
    report = [
        "# Contact Centre KPI Summary", "",
        f"Period: {k.period_start} to {k.period_end} ({int(k.days)} days)", "",
        "> Synthetic data is illustrative only. It does not describe a real operation.", "",
        f"- Offered calls: {k.offered_calls:,.0f}",
        f"- Answered calls: {k.answered_calls:,.0f}",
        f"- Abandoned calls: {k.abandoned_calls:,.0f}",
        f"- Answer rate: {k.answer_rate_pct:.1f}%",
        f"- Abandon rate: {k.abandon_rate_pct:.1f}%",
        f"- Service level: {k.service_level_pct:.1f}%",
        f"- Weighted AHT: {k.weighted_aht_seconds / 60:.2f} minutes",
        f"- Weighted ASA: {k.weighted_asa_seconds:.1f} seconds",
        f"- Average CSAT: {k.average_csat:.2f} / 5",
        f"- Repeat-call rate: {k.repeat_rate_pct:.1f}%",
        f"- Occupancy proxy: {k.occupancy_proxy_pct:.1f}%", "",
        "## Definitions and cautions", "",
        "- Service level is calculated as sla_answered / answered. Align the denominator with your official definition.",
        "- AHT and ASA are weighted by answered volume in the overall summary.",
        "- Repeat-call rate is repeat_calls / answered. Formal repeat analysis needs an agreed repeat window and contact/customer key.",
        "- Occupancy is a proxy: confirm agent_hours and available_hours have matching populations and definitions.",
        "- Review data-quality issues and metric definitions before using results in formal reporting."]
    (out / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("CONTACT CENTRE KPI SUMMARY")
    print(f"Period: {k.period_start} to {k.period_end}")
    print(f"Offered: {k.offered_calls:,.0f} | Abandon rate: {k.abandon_rate_pct:.1f}%")
    print(f"Service level: {k.service_level_pct:.1f}% | Weighted AHT: {k.weighted_aht_seconds / 60:.2f} min")
    print(f"Weighted ASA: {k.weighted_asa_seconds:.1f}s | CSAT: {k.average_csat:.2f}/5")
    print(f"Outputs: {out.resolve()}")

if __name__ == "__main__":
    main()
