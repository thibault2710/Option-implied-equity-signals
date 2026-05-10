# Final Results Pack: Public 2010-2023 Pipeline

## Main Interpretation

Low call-minus-put implied volatility identifies bottom-tail underperformance within the optionable-stock universe. The evidence is best interpreted as a negative-selection signal rather than a symmetric long-short sentiment factor. Results are gross of transaction costs.

## Headline Table

| strategy | annualized_return | annualized_volatility | sharpe_ratio | raw_t_stat | nw_t_stat | ff5_mom_alpha | ff5_mom_alpha_tstat | n_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bottom Decile U-B EW All | 7.57% | 6.82% | 1.11 | 4.16 | 4.00 | 6.40% | 5.07 | 168 |
| Bottom Decile U-B EW $100M+ | 7.73% | 5.20% | 1.49 | 5.56 | 5.87 | 6.29% | 5.85 | 168 |
| Bottom Quintile U-B EW All | 5.42% | 3.60% | 1.51 | 5.64 | 5.40 | 4.51% | 5.90 | 168 |
| Bottom Quintile U-B EW $100M+ | 4.92% | 3.14% | 1.57 | 5.87 | 5.93 | 3.99% | 5.61 | 168 |
| Q5-Q1 EW All | 5.35% | 5.32% | 1.01 | 3.77 | 4.47 | 5.45% | 4.33 | 168 |
| Q5-Q1 EW $100M+ | 5.56% | 4.64% | 1.20 | 4.48 | 4.81 | 5.42% | 4.43 | 168 |

## Value-Weighted Robustness Table

| strategy | annualized_return | annualized_volatility | sharpe_ratio | raw_t_stat | nw_t_stat | ff5_mom_alpha | ff5_mom_alpha_tstat | n_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bottom Decile U-B VW All | 9.04% | 9.97% | 0.91 | 3.39 | 3.81 | 5.08% | 2.93 | 168 |
| Bottom Decile U-B VW $100M+ | 8.27% | 9.11% | 0.91 | 3.40 | 3.53 | 4.20% | 2.49 | 168 |

## Decile and Leg Decomposition

| portfolio_or_leg | annualized_ew_return | source |
| --- | --- | --- |
| D1 | 3.96% | decile_returns |
| D2 | 8.26% | decile_returns |
| D3 | 12.20% | decile_returns |
| D4 | 13.86% | decile_returns |
| D5 | 13.04% | decile_returns |
| D6 | 13.27% | decile_returns |
| D7 | 13.71% | decile_returns |
| D8 | 14.12% | decile_returns |
| D9 | 13.64% | decile_returns |
| D10 | 9.28% | decile_returns |
| Universe | 11.53% | bottom_tail_returns |
| Bottom decile | 3.96% | bottom_tail_returns |
| Top decile | 9.28% | bottom_tail_returns |
| Universe - Bottom | 7.57% | bottom_tail_returns |
| Top - Universe | -2.25% | bottom_tail_returns |
| Top - Bottom | 5.32% | bottom_tail_returns |

## Factor Alpha Summary

| portfolio | alpha_annualized | alpha_tstat | r_squared | n_months |
| --- | --- | --- | --- | --- |
| IV Spread Bottom Quintile U-B EW All | 4.51% | 5.90 | 0.261 | 168 |
| IV Spread Bottom Decile U-B EW MktCap100M | 6.29% | 5.85 | 0.358 | 168 |
| IV Spread Bottom Quintile U-B EW MktCap100M | 3.99% | 5.61 | 0.244 | 168 |
| IV Spread Bottom Decile U-B EW All | 6.40% | 5.07 | 0.321 | 168 |
| IV Spread Q5-Q1 EW MktCap100M | 5.42% | 4.43 | 0.147 | 168 |
| IV Spread Q5-Q1 EW All | 5.45% | 4.33 | 0.180 | 168 |
| IV Spread Bottom Decile U-B VW All | 5.08% | 2.93 | 0.505 | 168 |
| IV Spread Bottom Decile U-B VW MktCap100M | 4.20% | 2.49 | 0.507 | 168 |

## Final Figures

- Figure 1: `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_figures/01_main_cumulative_performance.png` - Cumulative growth of $1 for universe-minus-bottom-tail IV-spread strategies, 2010-2023.
- Figure 2: `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_figures/02_decile_returns.png` - Annualized equal-weighted returns by IV-spread decile, 2010-2023.
- Figure 3: `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_figures/03_factor_alpha_headline.png` - Annualized FF5+Momentum alphas for selected IV-spread strategies, 2010-2023.

## Public Audit Status

| status | count |
| --- | --- |
| PASS | 67 |
| WARN | 0 |
| FAIL | 0 |
| INFO | 1 |

## Reproducibility Note

These public final outputs are generated from the standalone public pipeline scripts:

- `scripts/public/04_run_main_results.py`
- `scripts/public/05_run_factor_regressions.py`
- `scripts/public/08_create_final_outputs.py`
- `scripts/public/09_audit_results.py`

Raw WRDS, OptionMetrics, and CRSP data are not included in the public GitHub repository. Users with the required data access can regenerate raw and processed files locally.

## Output File Inventory

Final public tables:

- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_tables/audit_status_table_2010_2023.csv`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_tables/decile_leg_table_2010_2023.csv`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_tables/figure_inventory_2010_2023.csv`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_tables/headline_table_2010_2023.csv`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_tables/public_final_pack_validation.csv`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_tables/value_weighted_table_2010_2023.csv`

Final public figures:

- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_figures/01_main_cumulative_performance.png`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_figures/02_decile_returns.png`
- `/Users/thibaulteelen/Documents/thib/Booth/Quant projects/Options Project/options_implied_signals/outputs/public_2010_2023/final_figures/03_factor_alpha_headline.png`
