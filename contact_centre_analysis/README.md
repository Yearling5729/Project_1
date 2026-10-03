# Contact Centre Stats Analysis

A small Python project for analysing daily contact-centre performance. It validates input data, calculates operational KPIs, produces daily and weekly CSV outputs, saves charts, and writes a short Markdown report.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python contact_centre_analysis.py
```

The default run creates 90 days of seeded synthetic data. The figures are illustrative and do not describe a real operation.

## Outputs

The `outputs/` directory contains:
- `cleaned_daily_data.csv`: validated source data
- `daily_kpis.csv`: daily rates and rolling averages
- `weekly_kpis.csv`: weekly totals and rates
- `summary_kpis.csv`: overall period KPIs
- `daily_volume.png`: offered calls and 7-day average
- `service_performance.png`: abandon rate and service level
- `average_speed_of_answer.png`: ASA and 7-day average
- `analysis_report.md`: summary and metric caveats

## Use your own data

Export daily aggregates to CSV with these columns:

| Column | Meaning |
|---|---|
| `date` | Date for the daily row |
| `offered` | Calls offered |
| `answered` | Calls answered |
| `abandoned` | Calls abandoned |
| `sla_answered` | Calls answered within the service-level threshold |
| `aht_seconds` | Average handling time in seconds |
| `asa_seconds` | Average speed of answer in seconds |
| `csat_score` | Average CSAT score, from 1 to 5 |
| `repeat_calls` | Repeat-call count under your agreed definition |
| `agent_hours` | Productive agent hours |
| `available_hours` | Available staffed hours for the same population and period |

Run:

```bash
python contact_centre_analysis.py --input daily_contact_centre.csv --output-dir outputs
```

## KPI definitions and caveats

- Answer rate = answered / offered.
- Abandon rate = abandoned / offered.
- Service level = calls answered within threshold / answered calls. Confirm this denominator matches your organisation's official definition.
- Overall AHT and ASA are weighted by answered volume.
- Repeat-call rate = repeat calls / answered calls. Formal analysis needs a defined repeat window and suitable customer/contact key.
- Occupancy proxy = agent hours / available hours. Confirm both fields use matching definitions and populations.

The script checks required columns, parses dates, validates numeric values, checks count relationships, and aggregates duplicate dates. Investigate data-quality issues before using outputs for formal reporting.

Use aggregated, non-identifiable data. Do not add names, telephone numbers, account numbers or call transcripts to this example.
