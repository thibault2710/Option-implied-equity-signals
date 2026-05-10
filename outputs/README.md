# Outputs Directory

Bulk generated outputs are ignored in the public repository.

Curated public outputs are kept under:

- `outputs/public_2010_2023/final_results_pack.md`
- `outputs/public_2010_2023/final_tables/`
- `outputs/public_2010_2023/final_figures/`

Full diagnostic outputs, intermediate tables, and charts are not included. They can be regenerated locally by running the public pipeline after the required licensed data and processed panel are available:

```bash
python scripts/public/run_public_analysis.py
```

In a fresh public clone without licensed data, inspect the curated final outputs above rather than running the analysis pipeline.

The ignored diagnostic folders include:

- `outputs/public_2010_2023/tables/`
- `outputs/public_2010_2023/charts/`
- `outputs/sample_2010_2023/`
