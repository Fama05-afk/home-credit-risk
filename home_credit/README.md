# Home Credit dbt project

This is the dbt project that transforms the raw Home Credit Kaggle CSV files into the feature mart used for model training. See the [main project README](../README.md) for the full pipeline, results, and how this fits into the rest of the repo.

## Layers

```
models/
├── staging/        # 7 models: cleaning, column renaming
├── intermediate/    # 9 models: feature engineering (ratios, aggregations)
└── mart/            # 1 model: final table with 100+ features
```

## Commands

```bash
# Run the full pipeline
dbt run

# Run by layer
dbt run --select staging.*
dbt run --select intermediate.*
dbt run --select mart.*

# Tests
dbt test

# Check the connection and config
dbt debug
```

## Reproducibility

The `ORDER BY loan_id` clause in the mart model is non negotiable. Without it, row order can shift between runs, which silently changes the train/val/test split and makes model comparisons invalid.

## Connection

DuckDB, configured in `profiles.yml`. Database file: `../data/dbt_output/dev.duckdb`.
