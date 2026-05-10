# Option-Implied Volatility Spreads and Bottom-Tail Underperformance in U.S. Equities, 2010–2023

*Sample: signal months 2010-01 through 2023-12; forward returns 2010-02 through 2024-01.*

---

## Abstract

This project investigates whether option-implied signals predict subsequent stock returns within the optionable U.S. equity universe. The core signal is the IV spread, defined as the at-the-money call implied volatility minus the at-the-money put implied volatility for the same name on the same day. Using OptionMetrics IvyDB volatility surfaces, CRSP daily and monthly returns, and the official WRDS CRSP–OptionMetrics linking table, the analysis builds a monthly signal panel of 447,629 stock-months covering signal months 2010-01 through 2023-12 and forward returns 2010-02 through 2024-01. The headline finding is that stocks in the lowest decile of IV spread—names whose put IV is unusually high relative to their call IV—subsequently underperform the broader optionable universe. The equal-weighted strategy that buys the universe and sells the bottom IV-spread decile earns 7.57% annualized over 168 months, with a raw t-statistic of 4.16, a Newey-West t-statistic of 4.00, and a Fama-French five-factor plus momentum (FF5+MOM) alpha of 6.40% (t = 5.07). Restricting the universe to stocks with market capitalization above $100 million produces 7.73% annualized and a 6.29% FF5+MOM alpha (t = 5.85). The decile pattern is not smoothly monotonic: D1 returns 3.96% while D2–D9 cluster between 8% and 14% and D10 returns 9.28%. The cleaner interpretation is bottom-tail negative selection rather than a monotonic ranking factor. All numbers are gross of transaction costs. Average one-way turnover in the bottom decile is roughly 71%, and a rough cost sensitivity that charges both legs of the long-short shows that net returns turn slightly negative around 50 basis points per side. The result is therefore best framed as evidence of bottom-tail negative selection rather than a directly tradeable monthly long-short factor. A more practical re-framing is a long-only exclusion screen that simply removes the bottom IV-spread decile from a broad equal-weighted optionable book: this version improves the all-stock universe return from 11.53% to 12.37% annualized (an improvement of 0.84%, NW t = 4.00) at one-way turnover of about 8.5% per month, and survives realistic cost assumptions over the bands tested.

<div style="page-break-after: always;"></div>

## 1. Introduction

Options markets aggregate information about the cross-section of forward-looking risk in ways that the stock tape does not. Demand for downside protection, dealer hedging pressure, and informed trading on private information about earnings or corporate events can all show up first in the term structure and skew of implied volatility. A large body of empirical work has interpreted relative differences between call and put implied volatility for the same underlying — often called the IV spread — as a measure of asymmetric demand or informed sentiment about the stock.

The intuition is straightforward. When the put-side implied volatility for a name is high relative to the call-side implied volatility, the option market is paying up for downside protection or expressing relative pessimism about the underlying. If that pessimism is at least partially informed, the underlying stock should subsequently underperform.

This project tests the hypothesis directly. The IV spread is defined with the call leg first, so a low or negative IV spread corresponds to put IV being elevated relative to call IV. The question is whether stocks in the low IV-spread tail subsequently underperform the optionable universe. The full 2010–2023 sample, audited end-to-end, indicates that they do, and that this underperformance survives standard factor controls. The effect is concentrated in the bottom decile rather than spread evenly across the IV-spread distribution, which has direct implications for how the result should be framed.

The signal itself is established in the literature. Bali and Hovakimian (2009) document a positive cross-sectional relation between the call-minus-put implied volatility spread and next-month stock returns and interpret it as a jump-risk proxy. Cremers and Weinbaum (2010) show that deviations from put–call parity, measured through paired call and put IVs, predict future stock returns. An, Ang, Bali, and Cakici (2014) show that monthly changes in call and put implied volatilities also predict the cross-section of returns. Xing, Zhang, and Zhao (2010) document related predictability for an OTM-put-vs-ATM-call skew measure. The broader options-information literature, including Pan and Poteshman (2006) on signed option volume, Garleanu, Pedersen, and Poteshman (2009) on demand-based option pricing, and Carr and Wu (2009) and Bollerslev, Tauchen, and Zhou (2009) on the variance risk premium, supplies the structural intuition for why call-minus-put IV could capture downside concern, put demand, or negative information. Closer in spirit to the bottom-tail framing here, Atilgan, Bali, Demirtas, and Gunaydin (2020) document a left-tail return-momentum effect in the cross-section. The contribution of this project is not a new signal. It is an applied, audited 2010–2023 out-of-sample re-evaluation of the IV-spread predictor, with a deliberate bottom-tail framing, an honest treatment of trading costs, and a practical re-interpretation of the signal as a long-only negative-selection screen.

## 2. Data

The empirical work draws on three primary data sources, all accessed through WRDS.

OptionMetrics IvyDB provides the daily volatility surface used to construct option-implied signals. From this surface, the project extracts at-the-money call implied volatility (`iv_atm_call`), at-the-money put implied volatility (`iv_atm_put`), and out-of-the-money put implied volatility (`iv_otm_put`) at the standard 30-day horizon. The full-sample raw volatility surface contains roughly 45.4 million daily rows.

CRSP daily and monthly stock files supply realized returns and market capitalizations. Daily returns are used to compute trailing 21-day realized variance, annualized, for each `permno`. Monthly returns are used to evaluate next-month performance of portfolios formed on the IV spread.

The OptionMetrics-to-CRSP link is built from the official WRDS CRSP–OptionMetrics historical link (`wrdsapps_link_crsp_optionm.opcrsphist`). This source provides time-aware mappings from `secid` to `permno` with `sdate` and `edate` validity windows and a quality `score`. A static CUSIP-based bridge was built earlier as a fallback, but the canonical implementation in `src/linking.py` uses the official time-aware linker. `src/signals.py` retains only a deprecated wrapper.

Fama-French factors and the Carhart momentum factor are loaded from local Ken French CSVs and aligned by `return_month`, since signals constructed in calendar month *t* are matched to returns in month *t+1*.

The final monthly signal panel covers 168 contiguous return months from 2010-02 to 2024-01, with 447,629 monthly stock observations. The audited file inventory passes all 40 checks (raw files, linker, daily and monthly panels, return alignment, formula identities, dynamic month counts) with zero warnings or failures.

## 3. Signal Construction

For each `secid`-day in the volatility surface, the daily signal panel records:

- `iv_atm_call`: at-the-money call implied volatility at the 30-day horizon
- `iv_atm_put`: at-the-money put implied volatility at the 30-day horizon
- `iv_otm_put`: out-of-the-money put implied volatility at the 30-day horizon

From these, three options-implied signals are computed each day:

```
iv_spread       = iv_atm_call - iv_atm_put
iv_skew         = iv_otm_put  - iv_atm_call
implied_var     = iv_atm_call ^ 2
```

Realized variance is computed from CRSP daily returns as the trailing 21-day variance times 252. The variance risk premium follows as `vrp = implied_var - realized_var`. All formula identities are verified at the daily and monthly stages of the audit.

The monthly signal panel is built by taking, for each `permno`, the latest available daily signal value within calendar month *t* and matching it to that name's CRSP return in month *t+1*. This avoids look-ahead bias by construction. Cross-sectional z-scores within each signal-month are computed for each signal, and a composite signal is defined as `(z(iv_spread) − z(iv_skew) − z(vrp)) / 3`. The audit verifies that the signal date strictly precedes the return date in every panel row.

The IV spread is the focus of the final analysis because it is empirically the strongest standalone signal in the leaderboard (Section 7.3) and because its sign convention has a clean intuitive interpretation: low IV spread means high put IV relative to call IV, which is consistent with negative information or elevated demand for downside protection.

## 4. Methodology

Each signal-month, stocks are sorted by IV spread cross-sectionally into deciles and quintiles. The same sort is repeated under several universe filters: all optionable stocks; stocks with market capitalization above $100 million; above $500 million; and above $1 billion. Equal-weighted and value-weighted versions are both produced, but the cleaner equal-weighted construction is the focus of the headline results.

Two related portfolio definitions are considered. The first is the conventional long-short Q5 minus Q1 quintile sort, reported for completeness and for comparison with a more standard cross-sectional design. The second, and the one that frames the headline interpretation, is the universe-minus-bottom-tail leg, denoted `U-B`. This portfolio takes the equal-weighted return of the full optionable universe and subtracts the equal-weighted return of the bottom IV-spread decile (or quintile). A positive value of U-B is equivalent to the bottom tail underperforming the broader universe. This U-B framing is preferred because the decile leg decomposition (Section 6) shows that almost all of the dispersion in subsequent returns comes from the bottom decile being unusually weak, not from the top decile being unusually strong.

Mean returns are tested with raw and Newey-West (HAC) t-statistics with four lags. Factor regressions of the U-B and Q5–Q1 portfolios on CAPM, FF3, FF5, and FF5+MOM are estimated with HAC standard errors. Long-short returns are aligned to factor data by `return_month`, not by `signal_month`, and the dependent variable is the long-short return rather than its excess over the risk-free rate, since the long-short portfolio is self-financing.

<div style="page-break-after: always;"></div>

## 5. Headline Results

Within the equal-weighted, all-optionable-stock specification, the universe-minus-bottom-decile portfolio earns 7.57% annualized over 168 months. The raw t-statistic is 4.16 and the Newey-West HAC t-statistic with four lags is 4.00. The Fama-French five-factor plus momentum alpha is 6.40% per year with a t-statistic of 5.07. Restricting to stocks above $100 million in market capitalization, both the raw return and the alpha tighten further: 7.73% gross return (raw t = 5.56, NW t = 5.87) and a 6.29% FF5+MOM alpha (t = 5.85).

The bottom-quintile and Q5–Q1 specifications point in the same direction. The U-B bottom-quintile equal-weighted portfolio earns 5.42% (NW t = 5.40) with a 4.51% FF5+MOM alpha (t = 5.90). The standard Q5–Q1 equal-weighted portfolio earns 5.35% (NW t = 4.47) with a 5.45% alpha (t = 4.33).

**Table 1. Headline 2010–2023 results.** Equal-weighted strategies on the IV-spread signal. U-B denotes universe minus bottom tail. NW t-statistics use HAC standard errors with four lags.

| Strategy | Ann. return | Raw t-stat | NW t-stat | FF5+MOM alpha | Alpha t-stat | N months |
|---|---:|---:|---:|---:|---:|---:|
| Bottom Decile U-B EW All | 7.57% | 4.16 | 4.00 | 6.40% | 5.07 | 168 |
| Bottom Decile U-B EW $100M+ | 7.73% | 5.56 | 5.87 | 6.29% | 5.85 | 168 |
| Bottom Quintile U-B EW All | 5.42% | 5.64 | 5.40 | 4.51% | 5.90 | 168 |
| Bottom Quintile U-B EW $100M+ | 4.92% | 5.87 | 5.93 | 3.99% | 5.61 | 168 |
| Q5–Q1 EW All | 5.35% | 3.77 | 4.47 | 5.45% | 4.33 | 168 |
| Q5–Q1 EW $100M+ | 5.56% | 4.48 | 4.81 | 5.42% | 4.43 | 168 |

**Figure 1. Main Cumulative Performance.**

![Figure 1](../outputs/sample_2010_2023/final_figures/01_main_cumulative_performance.png)

*Figure 1. Cumulative growth of $1 invested in the universe-minus-bottom-tail IV-spread strategies, 2010–2023. Both the all-optionable and the $100M+ versions accumulate steadily over the sample, with no single year driving the performance.*

*Source: `outputs/sample_2010_2023/final_figures/01_main_cumulative_performance.png`.*

<div style="page-break-after: always;"></div>

## 6. Decile Pattern and Leg Decomposition

A natural concern with any signal that produces a strong long-short return is whether the underlying decile pattern is consistent with a smooth, monotonic factor or whether the entire effect is concentrated in a single tail. The decile evidence here is informative.

Annualized equal-weighted returns by decile (sorted from lowest to highest IV spread) are: D1 = 3.96%, D2 = 8.26%, D3 = 12.20%, D4 = 13.86%, D5 = 13.04%, D6 = 13.27%, D7 = 13.71%, D8 = 14.12%, D9 = 13.64%, D10 = 9.28%. The optionable universe earns 11.53% annualized over the same 168 months on an equal-weighted basis.

D1 is a clear negative outlier relative to the rest of the distribution. D10 is also somewhat below the universe and the interior deciles. The middle of the distribution, from D3 through D9, is essentially flat, with adjacent-decile differences typically within a few percentage points. The Spearman rank correlation between decile rank and decile mean return is 0.49, and only six of the nine adjacent-decile differences are positive in the equal-weighted construction. This pattern is not a clean monotonic ranking signal.

The leg decomposition makes the asymmetry explicit. Universe minus bottom decile earns 7.57% annualized, top minus universe earns −2.25%, and top minus bottom earns 5.32%. The bottom-tail leg dominates: the entire 5.32% top-minus-bottom long-short spread, and more, is attributable to the bottom decile's underperformance rather than the top decile's outperformance. This is the empirical basis for framing the result as bottom-tail negative selection rather than as a high-IV-spread alpha.

**Table 2. Decile and leg decomposition, equal-weighted, 2010–2023.**

| Portfolio / Leg | Annualized EW return |
|---|---:|
| D1 (lowest IV spread) | 3.96% |
| D2 | 8.26% |
| D3 | 12.20% |
| D4 | 13.86% |
| D5 | 13.04% |
| D6 | 13.27% |
| D7 | 13.71% |
| D8 | 14.12% |
| D9 | 13.64% |
| D10 (highest IV spread) | 9.28% |
| Universe | 11.53% |
| Universe − Bottom | 7.57% |
| Top − Universe | −2.25% |
| Top − Bottom | 5.32% |

**Figure 2. IV-Spread Decile Returns.**

![Figure 2](../outputs/sample_2010_2023/final_figures/02_decile_returns.png)

*Figure 2. Annualized equal-weighted returns by IV-spread decile. D1 is the clear negative outlier; the interior of the distribution is roughly flat and D10 also sits modestly below the universe.*

*Source: `outputs/sample_2010_2023/final_figures/02_decile_returns.png`.*

**Figure 3. Leg Decomposition.**

![Figure 3](../outputs/sample_2010_2023/final_figures/03_leg_decomposition.png)

*Figure 3. Annualized returns for the universe, the bottom IV-spread decile, the top decile, and the relative-return legs. The picture is dominated by the bottom decile's underperformance, with no symmetric top-tail outperformance.*

*Source: `outputs/sample_2010_2023/final_figures/03_leg_decomposition.png`.*

<div style="page-break-after: always;"></div>

## 7. Robustness

### 7.1 Factor controls

Fama-French five-factor plus momentum regressions confirm that the bottom-tail underperformance is not absorbed by standard factor exposures. For the all-stock equal-weighted U-B decile portfolio, the FF5+MOM alpha is 6.40% per year with t = 5.07. The strategy loads negatively on SMB (β = −0.28, t = −4.17), positively on RMW (β = 0.19, t = 2.14), and positively on momentum (β = 0.15, t = 3.89). The market beta is essentially zero (β = 0.01, t = 0.31). The same qualitative picture holds in the $100M+ specification, where the alpha is 6.29% (t = 5.85) with a negative SMB loading (β = −0.25, t = −6.41) and a positive momentum loading (β = 0.10, t = 3.65). The negative SMB loading is consistent with the bottom IV-spread decile being smaller on average than the universe (Section 8) and with the U-B leg therefore being long large names and short small names on net. The full factor anatomy is shown in Appendix Figure A2.

**Table 3. FF5+MOM factor anatomy of the IV-spread bottom-decile U-B strategies.** Alphas are annualized; t-statistics are HAC.

| Portfolio | Alpha | α t | β Mkt-RF | t | β SMB | t | β HML | t | β RMW | t | β CMA | t | β MOM | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-B EW All | 6.40% | 5.07 | 0.012 | 0.31 | −0.277 | −4.17 | 0.101 | 1.76 | 0.191 | 2.14 | −0.094 | −0.86 | 0.148 | 3.89 |
| U-B EW $100M+ | 6.29% | 5.85 | 0.045 | 1.79 | −0.249 | −6.41 | 0.025 | 0.59 | 0.142 | 2.24 | −0.040 | −0.50 | 0.100 | 3.65 |

### 7.2 Subperiods

The strategy is not concentrated in a single market regime, but neither is it uniform across time. Annualized U-B all-stock equal-weighted returns by subperiod are 5.86% in 2010–2013 (NW t = 3.02), 7.40% in 2014–2017 (NW t = 2.71), 4.31% in 2018–2020 (NW t = 0.98), and 13.08% in 2021–2023 (NW t = 2.85). The 2018–2020 subperiod is the weakest both in magnitude and in significance, and 2020 in particular is a noisy year for the all-stock specification. Excluding calendar 2020 raises the full-sample annualized return to 9.00% with a Newey-West t of 5.43.

The honest reading is that the result holds across subperiods on average, but with meaningful variation in significance and magnitude. The 2020 dispersion deserves to be flagged rather than excluded from the headline.

**Table 4. Subperiod performance of the bottom-decile U-B EW all-stock strategy.**

| Subperiod | Ann. return | NW t-stat | N months |
|---|---:|---:|---:|
| 2010–2013 | 5.86% | 3.02 | 47 |
| 2014–2017 | 7.40% | 2.71 | 48 |
| 2018–2020 | 4.31% | 0.98 | 36 |
| 2021–2023 | 13.08% | 2.85 | 36 |
| 2010–2017 | 6.64% | 3.84 | 95 |
| 2018–2023 | 8.70% | 2.31 | 72 |
| Excluding 2020 | 9.00% | 5.43 | 156 |

**Figure 4. Subperiod Performance.**

![Figure 4](../outputs/sample_2010_2023/final_figures/05_subperiod_performance.png)

*Figure 4. Annualized universe-minus-bottom-decile returns across major subperiods. Performance is positive in every subperiod and is strongest in 2021–2023; the 2018–2020 window is the weakest.*

*Source: `outputs/sample_2010_2023/final_figures/05_subperiod_performance.png`.*

### 7.3 Other signals

IV skew (`iv_otm_put − iv_atm_call`), the variance risk premium (`vrp`), and the composite signal `(z(iv_spread) − z(iv_skew) − z(vrp)) / 3` were all tested as standalone alternatives and as comparison points. IV skew and VRP are weaker standalone signals across the leaderboard, with NW t-statistics that frequently fall below 1 in equal-weighted Q5–Q1 specifications. The composite has a positive but weaker performance than IV spread itself. IV spread therefore drives the final results, and the headline interpretation is built around it. Appendix Figure A1 ranks all tested strategies by Newey-West t-statistic.

### 7.4 Signal extensions and holding-period decay

To check whether the baseline IV-spread *level* signal could be improved by simple feature engineering, four families of related signals were tested: changes in IV spread (one-month and three-month), put-IV change relative to call-IV change, level-plus-change combinations, persistent bottom-decile membership, and long-side call-strength variants. Three additional diagnostics were also produced: holding-period decay over months t+1 through t+3, lower-frequency (quarterly and semiannual) exclusion screens, and a split between newly-bottom and persistent-bottom names. The full tables are in the appendix and in the source CSVs; the summary points are below.

**Extensions do not displace the baseline.** The strongest universe-minus-bottom-tail strategies in the extension leaderboard are still the baseline `iv_spread_level`. The $100M+ equal-weighted U-B quintile earns 4.92% with NW t = 5.93, and the $100M+ U-B decile earns 7.73% with NW t = 5.87. The all-stock U-B quintile earns 5.42% with NW t = 5.40. Change-based variants (one-month change, three-month change, relative put pressure) and combinations (level-plus-change) appear in the leaderboard but do not clearly dominate the baseline on return, t-statistic, factor alpha, and turnover jointly. The recommendation embedded in the diagnostics file is to keep the baseline unchanged.

**No reliable long-side signal.** A symmetric set of *top-decile-minus-universe* sorts was run on the same extension grid. The best long-side result is roughly 0.33% annualized with NW t = 0.28. The conclusion is asymmetry: low IV spread identifies likely underperformers, but high IV spread does not identify reliable winners. This is consistent with the leg decomposition in Section 6 (top minus universe = −2.25%) and reinforces the bottom-tail framing of the paper.

**Predictability is short-horizon and concentrated in month t+1.** The all-stock U-B return is 7.57% at t+1 (NW t = 4.00), 1.95% at t+2-only (NW t = 0.84), and 3.93% at t+3-only (NW t = 2.15). In the $100M+ universe the picture is firmer further out: 7.73% at t+1 (NW t = 5.87), 3.70% at t+2-only (NW t = 2.31), and 4.37% at t+3-only (NW t = 3.44). The cumulative three-month $100M+ spread, scaled to an annual rate, is 5.16% with NW t = 4.22. The pattern is consistent with a short-horizon information component rather than a slowly-decaying risk premium, and the $100M+ result carries more into months two and three than the all-stock version.

**Newly-bottom names drive the result.** Splitting the bottom decile into stocks that are *newly* in the bottom (not in the bottom decile last month) versus *persistently* in the bottom (in the bottom for two consecutive months) is informative. In the all-stock universe, universe-minus-newly-bottom earns 8.10% (NW t = 4.57) while universe-minus-persistent-bottom earns 4.90% (NW t = 1.31). In the $100M+ universe, both legs are economically meaningful but the newly-bottom leg is sharper: 7.57% with NW t = 6.19, versus 6.80% with NW t = 2.70 for the persistent leg. This is consistent with the underperformance reflecting fresh negative option-market information or short-lived informed trading rather than a chronic-risk premium on names that always sit in the high-put-IV tail.

**Quarterly exclusion is a defensible compromise; semiannual is not.** The long-only exclusion improvement decays with rebalancing frequency. In the all-stock universe the monthly improvement is 0.84% (NW t = 4.00), the quarterly improvement is 0.57% (NW t = 2.67), and the semiannual improvement is 0.29% (NW t = 1.19). In the $100M+ universe the monthly improvement is 0.86% (NW t = 5.87), quarterly is 0.61% (NW t = 4.61), and semiannual is 0.32% (NW t = 1.99). Quarterly exclusion preserves the economic sign and a meaningful share of the statistical strength while lowering the rebalancing frequency; semiannual exclusion weakens markedly. For a manager who wants to rebalance less than monthly, quarterly is the supportable choice.

**Figure 5. IV-Spread Holding-Period Decay.**

![Figure 5](../outputs/sample_2010_2023/charts/iv_spread_holding_period_decay.png)

*Figure 5. Annualized universe-minus-bottom-decile IV-spread returns at the t+1, t+2-only, and t+3-only horizons, all-stock and $100M+ equal-weighted. Predictability is concentrated in t+1; the $100M+ specification carries more into months 2 and 3.*

*Source: `outputs/sample_2010_2023/charts/iv_spread_holding_period_decay.png`.*

The overall reading of the extensions is straightforward. The baseline low-IV-spread bottom-tail signal remains the cleanest core finding. Change-based, persistence, and long-side variants are useful as robustness checks but do not replace it. The signal is asymmetric, short-horizon, and concentrated in newly-arriving bottom-tail names, and quarterly rebalancing is the most defensible lower-frequency variant for the long-only application.

<div style="page-break-after: always;"></div>

## 8. Bottom-Tail Characteristics and Economic Interpretation

If the result reflects bottom-tail negative selection, the bottom decile should look characteristically different from the universe, and in ways that are consistent with downside-risk concerns. The data are consistent with this reading.

Average market capitalization in the universe is roughly $9.49 billion, while the bottom IV-spread decile averages roughly $1.02 billion, a difference of about $8.47 billion. The bottom decile is therefore meaningfully smaller than the universe, which motivates the $100M+ robustness check; reassuringly, the result is if anything sharper inside that filter.

The bottom decile is also more volatile, both implied and realized, and it has unusually elevated put IV. ATM call IV averages 0.91 in the bottom decile versus 0.66 in the universe; ATM put IV averages 1.36 in the bottom decile versus 0.61 in the universe; IV skew averages 0.39 versus 0.05; trailing realized variance averages 0.43 versus 0.28; and the variance risk premium averages 1.05 versus 0.71. By construction, the IV spread itself averages −0.45 in the bottom decile versus +0.05 in the universe.

Taken together, the bottom IV-spread decile is populated by smaller, more volatile names with sharply elevated put-side option premia. This is the kind of cross-section in which one would expect downside-risk concerns or asymmetric protection demand to be priced, and it is the part of the cross-section that subsequently underperforms. Appendix Figure A3 visualises these characteristic differences.

**Table 5. Selected characteristics: bottom decile versus universe, all-stock specification.**

| Characteristic | Universe mean | Bottom decile mean | Bottom − Universe |
|---|---:|---:|---:|
| Market cap (USD millions) | 9,493.37 | 1,019.88 | −8,473.49 |
| ATM call IV | 0.6622 | 0.9064 | +0.2442 |
| ATM put IV | 0.6134 | 1.3585 | +0.7451 |
| IV spread | 0.0488 | −0.4521 | −0.5009 |
| IV skew | 0.0535 | 0.3856 | +0.3322 |
| Realized variance (annualized) | 0.2796 | 0.4349 | +0.1553 |
| Variance risk premium | 0.7072 | 1.0524 | +0.3452 |

## 9. Sector Exposure

A simple sector overlay shows that the bottom IV-spread decile is overrepresented in Finance / Real Estate (16.09% of bottom-decile stock-months versus 13.96% of the universe) and in the Unknown sector category (5.98% versus 3.22%). It is underrepresented in Retail (4.27% versus 6.67%), Services (16.53% versus 18.01%), and Wholesale (2.07% versus 2.63%). Manufacturing is essentially proportionally represented.

The Unknown overrepresentation is partly an artifact of incomplete sector labelling, and the Finance / Real Estate concentration is a known sensitivity for any signal that captures downside-risk pricing. Both observations are useful diagnostics rather than the central result, and both motivate the use of the $100M+ universe filter and the equal-weighted construction. Sector-neutral sorts have been explored and broadly preserve the bottom-tail underperformance pattern. Appendix Figure A4 shows the full sector overrepresentation profile.

<div style="page-break-after: always;"></div>

## 10. Implementation and Transaction Costs

Reported returns and alphas are gross of transaction costs. A naive monthly long-short implementation of the IV-spread strategy is turnover-intensive: average one-way bottom-decile turnover is 70.70% per month in the all-stock equal-weighted construction, and 72.84% per month in the $100M+ construction.

A rough sensitivity analysis applies a flat per-side cost to *both legs* of the universe-minus-bottom-decile portfolio, where each leg incurs cost in proportion to its monthly turnover and the cost is annualized. The resulting drag is roughly twice the single-leg drag at the same per-side rate. At 10 basis points per side the all-stock strategy nets to roughly 5.89% annualized; at 25 basis points it nets to roughly 3.36%; at 50 basis points it falls to roughly −0.86%. The $100M+ specification follows a similar profile (5.99%, 3.38%, and −0.96% at 10, 25, and 50 bps respectively). These figures are not a substitute for a calibrated trading-cost model — they ignore borrow costs, market-impact concavity, capacity, and any liquidity-aware execution — but they make the implementation point clearly. The full long-short cost-sensitivity profile is shown in Appendix Figure A5.

The honest implication is that the IV spread bottom-tail signal is informative as a return-predictability and risk-management diagnostic, and is plausibly tradeable at very low cost levels, but a naive monthly long-short implementation deteriorates rapidly at realistic execution costs. Lower-frequency rebalancing, tighter liquidity screens, and cost-aware portfolio construction would be necessary for a serious implementation. This motivates the long-only exclusion specification developed in Section 11, which replaces the explicit short leg of the U-B portfolio with a one-sided membership rule (drop the bottom IV-spread decile from the equal-weighted book). That construction has materially lower turnover, retains a statistically and economically meaningful return improvement over the full universe, and is the more practical way to use the signal in an actual portfolio.

**Table 6. Long-short cost sensitivity, per-side rate applied to both legs.** "Cost (bps)" is the per-side rate; the annualized drag is approximately 2 × cost × turnover × 12.

| Strategy | Cost (bps, per side) | One-way turnover | Net ann. return |
|---|---:|---:|---:|
| Bottom decile U-B EW All | 10 | 70.70% | 5.89% |
| Bottom decile U-B EW All | 25 | 70.70% | 3.36% |
| Bottom decile U-B EW All | 50 | 70.70% | −0.86% |
| Bottom decile U-B EW $100M+ | 10 | 72.84% | 5.99% |
| Bottom decile U-B EW $100M+ | 25 | 72.84% | 3.38% |
| Bottom decile U-B EW $100M+ | 50 | 72.84% | −0.96% |

<div style="page-break-after: always;"></div>

## 11. Long-Only Application: IV Spread as a Negative-Selection Screen

The decile pattern documented in Section 6 and the cost sensitivity in Section 10 together motivate a one-sided use of the signal. Instead of buying the universe and selling the bottom decile as a self-financing long-short, a long-only manager can simply drop the bottom IV-spread decile from a broad equal-weighted optionable book. The short leg is removed and replaced with a membership rule on the long side. This re-frames the IV spread as a negative-selection screen rather than as a tradeable factor.

The all-stock equal-weighted full universe earns 11.53% annualized over the 168-month sample, with an annualized volatility of 21.58% and a Sharpe ratio of 0.53. Excluding the bottom IV-spread decile every month raises the annualized return to 12.37%, the volatility falls slightly to 21.27%, and the Sharpe ratio rises to 0.58. The annualized improvement of 0.84% has a Newey-West t-statistic of 4.00 on the difference portfolio. Excluding the bottom quintile rather than the bottom decile produces a larger gross improvement of 1.36% (NW t = 5.40). Inside the $100M+ market-cap filter the picture is the same: the full universe earns 11.92% with a Sharpe of 0.57, the bottom-decile exclusion earns 12.78% with a Sharpe of 0.62, and the 0.86% improvement has a Newey-West t-statistic of 5.87. Bottom-quintile exclusion in $100M+ improves returns by 1.23%.

The factor regression of the difference portfolio on FF5 plus momentum gives an annualized alpha of 0.71% with a t-statistic of 5.07 for the all-stock bottom-decile exclusion-minus-universe spread, and 0.70% with a t-statistic of 5.85 inside the $100M+ universe. The factor anatomy mirrors Section 7: the difference portfolio inherits the same negative SMB loading and modest positive momentum exposure as the U-B long-short, because the construction is mechanically the same shape, just without the explicit short.

The implementation case rests on turnover and cost. Where the U-B long-short rebalances the bottom decile in full each month and incurs roughly 71% one-way turnover on the short leg, the long-only exclusion only changes the *membership* of the long book. The membership-change approximation gives an average one-way turnover of about 8.46% per month for the all-stock bottom-decile exclusion and 8.82% per month for the $100M+ exclusion — roughly an order of magnitude lower than the long-short construction. The cost convention here uses the same per-side rate as Section 10, but charges only the *single long leg* of the exclusion portfolio (there is no explicit short leg). Under this single-leg accounting, the all-stock bottom-decile exclusion improvement nets to approximately 0.59% annualized at 25 basis points per side and 0.34% annualized at 50 basis points per side. Both remain positive, and the long-only construction never crosses zero over the cost bands tested in this analysis. The drag here is mechanically smaller than in the long-short version of Section 10 because there is one leg to charge instead of two, on top of the much lower turnover. Appendix Figure A6 shows the cumulative improvement of the exclusion portfolio over the full universe across the sample.

Two qualifications belong on the page. First, this is a membership-change turnover approximation, not a full transaction-cost model: it does not model normal benchmark-driven rebalancing of the long leg, value-weight drift, securities-lending costs, or market impact, and it should not be read as a calibrated trading-cost analysis. Second, the long-only screen does not generate alpha against FF5+MOM as an *absolute* portfolio; the exclusion portfolio's standalone FF5+MOM alpha is slightly negative and statistically insignificant. The economic content is in the difference portfolio, which captures the value of avoiding the bottom IV-spread tail relative to a passive equal-weighted optionable benchmark.

The interpretation is therefore the practical re-framing of the result. The IV spread is most useful here as a long-only exclusion or risk-management overlay integrated into normal rebalancing, not as a standalone high-turnover long-short strategy. The original long-short evidence remains the cleaner statistical test of the cross-sectional effect; the long-only exclusion screen is the more implementable application.

**Table 7. Long-only IV-spread exclusion screen, equal-weighted, 2010–2023.** Improvement is the annualized difference between the exclusion portfolio and the full-universe portfolio. NW t-statistics use HAC standard errors with four lags. Difference-portfolio FF5+MOM alphas are annualized.

| Universe | Portfolio | Ann. return | Sharpe | Improvement | NW t-stat | FF5+MOM α (diff) | α t-stat |
|---|---|---:|---:|---:|---:|---:|---:|
| All | Full universe | 11.53% | 0.53 | — | — | — | — |
| All | Exclude bottom decile | 12.37% | 0.58 | 0.84% | 4.00 | 0.71% | 5.07 |
| All | Exclude bottom quintile | 12.89% | 0.60 | 1.36% | 5.40 | 1.13% | 5.90 |
| $100M+ | Full universe | 11.92% | 0.57 | — | — | — | — |
| $100M+ | Exclude bottom decile | 12.78% | 0.62 | 0.86% | 5.87 | 0.70% | 5.85 |
| $100M+ | Exclude bottom quintile | 13.15% | 0.63 | 1.23% | 5.92 | 1.00% | 5.61 |

**Table 8. Approximate one-way membership turnover and rough cost-net improvement for the bottom-decile exclusion screen.** Cost-net improvement applies the per-side cost rate from Section 10 to the *single long leg* of the exclusion portfolio (annualized drag ≈ cost × turnover × 12).

| Universe | Portfolio | One-way turnover | Net improvement at 25 bps per side | Net improvement at 50 bps per side |
|---|---|---:|---:|---:|
| All | Exclude bottom decile | 8.46% | 0.59% | 0.34% |
| $100M+ | Exclude bottom decile | 8.82% | 0.60% | 0.34% |

**Figure 6. Long-Only Exclusion Cumulative Returns.**

![Figure 6](../outputs/sample_2010_2023/charts/long_only_exclusion_cumulative_returns.png)

*Figure 6. Cumulative growth of $1 for the equal-weighted full universe and the bottom-decile-exclusion portfolio, all-stock and $100M+ versions, 2010–2023.*

*Source: `outputs/sample_2010_2023/charts/long_only_exclusion_cumulative_returns.png`.*

**Figure 7. Long-Only Cost Sensitivity.**

![Figure 7](../outputs/sample_2010_2023/charts/long_only_exclusion_cost_sensitivity.png)

*Figure 7. Net annualized improvement of the long-only exclusion screen versus the full universe under the per-side cost convention from Section 10 applied to the single long leg of the exclusion portfolio.*

*Source: `outputs/sample_2010_2023/charts/long_only_exclusion_cost_sensitivity.png`.*

<div style="page-break-after: always;"></div>

## 12. Audit and Reproducibility

The pipeline is organized as a sequence of numbered scripts that move from raw data to final tables. The major stages are (i) WRDS access tests and raw OptionMetrics and CRSP pulls, (ii) construction of the official time-aware OptionMetrics-CRSP link, (iii) daily IV-signal construction, (iv) realized variance and VRP computation, (v) the monthly signal panel, (vi) cross-sectional quintile and decile sorts, (vii) market-cap and transformation robustness, (viii) factor regressions, (ix) bottom-tail diagnostics, (x) full-sample 2010–2023 expansion with sample-specific outputs, (xi) the research audit, and (xii) the final results pack and figure inventory.

The full-sample audit ran 40 checks across raw files, the official linker, the daily IV panel, the daily VRP panel, the monthly panel, the bottom-tail returns and summary, and the factor regression summary. All 40 checks passed, with 0 warnings and 0 failures. Verified items include: required columns present at each stage; no duplicate `permno`-date rows in the daily panels and no duplicate `permno`-`signal_month` rows in the monthly panel; signal date strictly precedes return date; `return_month = signal_month + 1` for every row; the IV spread, IV skew, implied variance, and VRP formulas reproduce exactly from their inputs; realized variance is non-negative; the dynamic month count equals 168 for signals 2010-01 to 2023-12 and returns 2010-02 to 2024-01; the main `iv_spread_adj` U-B EW row is present in the bottom-tail summary; and all expected factor models are present in the regression output.

One interim issue surfaced and was fixed during the expansion. The earlier `scripts/30_compare_2010_2023_vs_2018_2023.py` selected the first matching `tail`/`universe`/`weighting`/`leg` row in the bottom-tail summary, which in the full-sample summary corresponded to `composite_signal` rather than `iv_spread_adj`. The reconciliation note in `reconcile_bottom_tail_final_summary.md` documents the fix, and the headline numbers reported here come from a strict `signal == iv_spread_adj` filter on `bottom_tail_summary_2010_2023.csv` that matches the directly recomputed monthly returns.

## 13. Limitations

Several limitations should be kept in view when interpreting the results.

The reported long-short returns and alphas are gross of transaction costs, securities lending costs, and market impact. Even the rough cost sensitivity in Section 10 makes clear that a naive monthly long-short implementation is sensitive to cost assumptions, which is the central reason the long-only exclusion screen in Section 11 is presented as the more practical application.

Turnover is high, around 70% one-way per month for the bottom decile. This is consistent with a tail signal whose membership churns each month, but it amplifies the implementation problem.

The IV-spread signal is not a smooth monotonic factor across the cross-section. Most of the predictability sits in the lowest decile, and the headline U-B framing is a deliberate choice to reflect that. A reader who interprets the result as a generic high-minus-low IV-spread alpha will overstate what the data show.

The bottom decile is meaningfully smaller than the universe, and the negative SMB loading is partly a mechanical consequence of being long the universe and short the bottom tail. The $100M+ result is reported precisely to address this concern, and it is if anything stronger.

Sector classification is incomplete. A nontrivial share of bottom-decile observations is labelled Unknown, which limits the precision of sector-neutral robustness checks. A more complete industry assignment would be a useful refinement.

The result is within the optionable-stock universe and is conditional on having a valid daily volatility surface for the underlying. It is not a statement about the entire CRSP universe, and it should not be extrapolated beyond stocks for which the option market is liquid enough to provide reliable IV inputs.

Future work could extend the sample beyond 2023, evaluate the signal under lower-frequency rebalancing, build a calibrated transaction-cost-aware portfolio construction that internalizes turnover, impose stronger liquidity screens, and explore conditioning information that interacts with the IV spread (for example, IV-spread changes, IV-spread term structure, or post-news windows).

## 14. Conclusion

Within the optionable U.S. equity universe over 2010–2023, stocks in the lowest decile of the call-minus-put implied volatility spread underperform the broader optionable universe in the following month. The equal-weighted universe-minus-bottom-decile strategy earns 7.57% annualized over 168 months with a Newey-West t of 4.00 and a Fama-French five-factor plus momentum alpha of 6.40% (t = 5.07), and the result tightens slightly inside the $100 million market-cap filter. The signal is robust statistically, but it is not a smoothly monotonic ranking factor: almost all of the predictability comes from the bottom decile, and the cleaner reading is bottom-tail negative selection rather than a high-minus-low IV-spread alpha. Performance is positive across major subperiods, although weakest in 2018–2020. The bottom decile is smaller, more volatile, and carries sharply elevated put IV, consistent with the option market pricing downside risk that is at least partially realized in subsequent returns. The direct long-short implementation has high turnover and is gross of trading costs; rough cost sensitivity shows that it is competitive at low cost levels but compresses quickly as costs rise. The more practical final application is therefore the long-only exclusion screen of Section 11, which simply removes the bottom IV-spread decile from a broad equal-weighted optionable book, raises annualized return by about 0.84% (NW t = 4.00; 0.86% with NW t = 5.87 inside the $100M+ universe), turns over roughly an order of magnitude less than the long-short version, and remains positive over the cost bands tested. Read as a whole, the evidence supports treating IV spread as a bottom-tail warning signal and as a long-only negative-selection or risk-management overlay, not as a standalone tradeable factor. Additional extension tests (Section 7.4) show that simple IV-spread levels remain more reliable than change-based or long-side call-strength variants, and that the signal is strongest over the next month, especially for newly-entering bottom-decile stocks.

<div style="page-break-after: always;"></div>

# Appendix

## Appendix A. Supporting Figures

These figures are referenced from the body but moved here to keep the main narrative compact. Each is sourced from the full-sample results pipeline.

**Figure A1. Signal Leaderboard.**

![Figure A1](../outputs/sample_2010_2023/final_figures/04_signal_leaderboard.png)

*Figure A1. Top full-sample strategies ranked by Newey-West t-statistic. IV-spread bottom-tail strategies dominate the leaderboard; IV skew, VRP, and the composite are weaker as standalone signals.*

*Source: `outputs/sample_2010_2023/final_figures/04_signal_leaderboard.png`.*

**Figure A2. FF5+Momentum Factor Betas.**

![Figure A2](../outputs/sample_2010_2023/final_figures/06_factor_betas.png)

*Figure A2. FF5+Momentum factor betas for selected IV-spread strategies. The negative SMB loading and the moderate positive momentum loading are consistent with the bottom decile being smaller-cap and recently weaker than the universe.*

*Source: `outputs/sample_2010_2023/final_figures/06_factor_betas.png`.*

**Figure A3. Bottom-Tail Characteristics.**

![Figure A3](../outputs/sample_2010_2023/final_figures/07_bottom_tail_characteristics.png)

*Figure A3. Bottom-decile characteristics relative to the optionable-stock universe. The bottom IV-spread decile is smaller, more volatile, and carries unusually elevated put IV.*

*Source: `outputs/sample_2010_2023/final_figures/07_bottom_tail_characteristics.png`.*

**Figure A4. Sector Exposure.**

![Figure A4](../outputs/sample_2010_2023/final_figures/09_sector_exposure.png)

*Figure A4. Average sector overrepresentation in the bottom IV-spread decile. Finance / Real Estate and Unknown are the two overrepresented categories.*

*Source: `outputs/sample_2010_2023/final_figures/09_sector_exposure.png`.*

**Figure A5. Long-Short Transaction-Cost Sensitivity.**

![Figure A5](../outputs/sample_2010_2023/final_figures/08_transaction_cost_sensitivity.png)

*Figure A5. Net annualized returns for the long-short construction under the per-side cost convention applied to both legs. Both strategies remain positive at 10 bps per side, are markedly compressed at 25 bps, and turn slightly negative at 50 bps.*

*Source: `outputs/sample_2010_2023/final_figures/08_transaction_cost_sensitivity.png`.*

**Figure A6. Long-Only Exclusion Cumulative Improvement.**

![Figure A6](../outputs/sample_2010_2023/charts/long_only_exclusion_improvement_cumulative.png)

*Figure A6. Cumulative annualized improvement of the bottom-decile-exclusion portfolio over the full-universe portfolio.*

*Source: `outputs/sample_2010_2023/charts/long_only_exclusion_improvement_cumulative.png`.*

**Figure A7. Signal-Extensions Leaderboard.**

![Figure A7](../outputs/sample_2010_2023/charts/signal_extensions_leaderboard.png)

*Figure A7. Universe-minus-bottom-decile signal extensions ranked by Newey-West t-statistic. Baseline `iv_spread_level` strategies remain at the top of the leaderboard.*

*Source: `outputs/sample_2010_2023/charts/signal_extensions_leaderboard.png`.*

**Figure A8. Signal-Extensions Decile Returns.**

![Figure A8](../outputs/sample_2010_2023/charts/signal_extensions_decile_returns.png)

*Figure A8. Annualized decile returns for the tested IV-spread extension signals (level, one-month change, three-month change, relative put pressure, level-plus-change combination).*

*Source: `outputs/sample_2010_2023/charts/signal_extensions_decile_returns.png`.*

**Figure A9. Lower-Frequency Long-Only Exclusion.**

![Figure A9](../outputs/sample_2010_2023/charts/iv_spread_lower_frequency_exclusion.png)

*Figure A9. Annualized improvement of the long-only bottom-decile exclusion at monthly, quarterly, and semiannual rebalancing frequencies, all-stock and $100M+ equal-weighted.*

*Source: `outputs/sample_2010_2023/charts/iv_spread_lower_frequency_exclusion.png`.*

**Figure A10. Newly-Bottom vs Persistent-Bottom.**

![Figure A10](../outputs/sample_2010_2023/charts/iv_spread_new_vs_persistent_bottom.png)

*Figure A10. Universe-minus-newly-bottom and universe-minus-persistent-bottom annualized returns and Newey-West t-statistics, all-stock and $100M+ equal-weighted.*

*Source: `outputs/sample_2010_2023/charts/iv_spread_new_vs_persistent_bottom.png`.*

<div style="page-break-after: always;"></div>

## Appendix B. Additional Robustness Tables

The body tables (Tables 1–8) report the headline equal-weighted decile and quintile results, the U-B and Q5–Q1 specifications in the all-stock and $100M+ universes, and the long-only exclusion screen. Additional robustness is available in the source CSVs but not reproduced here in full. In particular:

- Value-weighted versions of the U-B decile, U-B quintile, and Q5–Q1 strategies are produced for every universe filter. Source: `bottom_tail_summary_2010_2023.csv`, weighting field set to `vw`.
- Larger market-cap universe filters at $500M and $1B are produced for every U-B and Q5–Q1 specification. Source: `bottom_tail_summary_2010_2023.csv`, universe field set to `mktcap_500m` or `mktcap_1b`.
- Signal-transformation robustness (raw, percentile rank, winsorized z-score) is produced for the Q5–Q1 quintile sorts. Source: `robustness_quintile_summary.csv` and the per-strategy CSV files referenced from `alpha_anatomy_signal_leaderboard.csv`.
- Decile-rank decomposition (D10−D1 and intermediate adjacent-decile contrasts) is in `alpha_anatomy_subperiods.csv` and `alpha_anatomy_leg_decomposition.csv`.
- Factor-regression specifications across CAPM, FF3, FF5, and FF5+MOM are stored in `factor_regression_summary_2010_2023.csv` for every selected portfolio.
- Long-only exclusion variants beyond bottom-decile and bottom-quintile (including symmetric bottom-and-top exclusions, larger universe filters, value-weighted versions) are stored in `long_only_exclusion_summary_2010_2023.csv` and `long_only_exclusion_improvement_2010_2023.csv`.
- Signal-extensions diagnostics — one-month and three-month changes in IV spread, relative put-IV pressure, level-plus-change combinations, persistent-bottom membership, and long-side call-strength variants — are stored in `signal_extensions_sort_summary_2010_2023.csv`, `signal_extensions_long_only_summary_2010_2023.csv`, `signal_extensions_factor_summary_2010_2023.csv`, `signal_extensions_turnover_2010_2023.csv`, and `signal_extensions_cost_sensitivity_2010_2023.csv`. The summary recommendation is in `signal_extensions_summary_report.md`.
- Holding-period decay (t+1, t+2-only, t+3-only, cumulative-3m, cumulative-6m) is in `iv_spread_holding_period_decay_summary_2010_2023.csv`.
- Newly-bottom versus persistent-bottom decomposition is in `iv_spread_new_vs_persistent_bottom_summary_2010_2023.csv`.
- Lower-frequency (monthly, quarterly, semiannual) exclusion screens are in `iv_spread_lower_frequency_exclusion_summary_2010_2023.csv` and `iv_spread_lower_frequency_exclusion_costs_2010_2023.csv`. The summary is in `iv_spread_holding_persistence_summary_report.md`.

## Appendix C. Figure Inventory

**Table C1. Figure inventory.** Body and appendix figures with file paths.

| # | Title | File path | Location |
|---|---|---|---|
| 1 | Main Cumulative Performance | `outputs/sample_2010_2023/final_figures/01_main_cumulative_performance.png` | Body §5 |
| 2 | IV-Spread Decile Returns | `outputs/sample_2010_2023/final_figures/02_decile_returns.png` | Body §6 |
| 3 | Leg Decomposition | `outputs/sample_2010_2023/final_figures/03_leg_decomposition.png` | Body §6 |
| 4 | Subperiod Performance | `outputs/sample_2010_2023/final_figures/05_subperiod_performance.png` | Body §7.2 |
| 5 | IV-Spread Holding-Period Decay | `outputs/sample_2010_2023/charts/iv_spread_holding_period_decay.png` | Body §7.4 |
| 6 | Long-Only Exclusion Cumulative Returns | `outputs/sample_2010_2023/charts/long_only_exclusion_cumulative_returns.png` | Body §11 |
| 7 | Long-Only Cost Sensitivity | `outputs/sample_2010_2023/charts/long_only_exclusion_cost_sensitivity.png` | Body §11 |
| A1 | Signal Leaderboard | `outputs/sample_2010_2023/final_figures/04_signal_leaderboard.png` | Appendix A |
| A2 | FF5+Momentum Factor Betas | `outputs/sample_2010_2023/final_figures/06_factor_betas.png` | Appendix A |
| A3 | Bottom-Tail Characteristics | `outputs/sample_2010_2023/final_figures/07_bottom_tail_characteristics.png` | Appendix A |
| A4 | Sector Exposure | `outputs/sample_2010_2023/final_figures/09_sector_exposure.png` | Appendix A |
| A5 | Long-Short Transaction-Cost Sensitivity | `outputs/sample_2010_2023/final_figures/08_transaction_cost_sensitivity.png` | Appendix A |
| A6 | Long-Only Exclusion Cumulative Improvement | `outputs/sample_2010_2023/charts/long_only_exclusion_improvement_cumulative.png` | Appendix A |
| A7 | Signal-Extensions Leaderboard | `outputs/sample_2010_2023/charts/signal_extensions_leaderboard.png` | Appendix A |
| A8 | Signal-Extensions Decile Returns | `outputs/sample_2010_2023/charts/signal_extensions_decile_returns.png` | Appendix A |
| A9 | Lower-Frequency Long-Only Exclusion | `outputs/sample_2010_2023/charts/iv_spread_lower_frequency_exclusion.png` | Appendix A |
| A10 | Newly-Bottom vs Persistent-Bottom | `outputs/sample_2010_2023/charts/iv_spread_new_vs_persistent_bottom.png` | Appendix A |

## Appendix D. Audit and Reproducibility Summary

**Table D1. Full-sample audit status.**

| Status | Count |
|---|---:|
| PASS | 40 |
| WARN | 0 |
| FAIL | 0 |

Sample coverage: signals 2010-01 to 2023-12; forward returns 2010-02 to 2024-01; 168 contiguous return months; 447,629 monthly stock observations.

Verified items include required columns at every stage, absence of duplicate keys, signal-precedes-return alignment, formula identities for IV spread, IV skew, implied variance, and VRP, non-negative realized variance, dynamic month count of 168, presence of the main `iv_spread_adj` row in the bottom-tail summary, and presence of all expected factor-regression models. Source: `outputs/sample_2010_2023/tables/research_audit_summary_2010_2023.csv`.

The reconciliation of the previous comparison-script issue (script 30 selected `composite_signal` instead of `iv_spread_adj`) is documented in `outputs/sample_2010_2023/tables/reconcile_bottom_tail_final_summary.md`.

## Appendix E. Tables Directory

**Table E1. Body and appendix tables.**

| # | Title | Source CSV / MD |
|---|---|---|
| 1 | Headline 2010–2023 results | `final_results_pack.md` §1; `bottom_tail_summary_2010_2023.csv` |
| 2 | Decile and leg decomposition | `alpha_anatomy_leg_decomposition.csv`; `final_results_pack.md` §2 |
| 3 | FF5+MOM factor anatomy | `factor_regression_summary_2010_2023.csv` |
| 4 | Subperiod performance | `alpha_anatomy_subperiods.csv` |
| 5 | Bottom-decile characteristics | `alpha_anatomy_bottom_tail_characteristics.csv` |
| 6 | Long-short cost sensitivity | `alpha_anatomy_transaction_cost_sensitivity.csv` |
| 7 | Long-only exclusion screen | `long_only_exclusion_summary_2010_2023.csv`; `long_only_exclusion_improvement_2010_2023.csv`; `long_only_exclusion_alpha_summary_2010_2023.csv` |
| 8 | Long-only turnover and cost-net improvement | `long_only_exclusion_turnover_2010_2023.csv`; `long_only_exclusion_cost_sensitivity_2010_2023.csv` |
| C1 | Figure inventory | `outputs/sample_2010_2023/tables/final_figure_inventory.csv` |
| D1 | Audit status | `research_audit_summary_2010_2023.csv` |

<div style="page-break-after: always;"></div>

## References (needs verification)

The references below correspond to the in-text citations in Section 1. Author lists, journals, volumes, issue numbers, and page ranges should be confirmed against the original sources before final publication. No citation has been added that is not used in the body.

- Atilgan, Y., Bali, T. G., Demirtas, K. O., and Gunaydin, A. D. (2020). Left-Tail Momentum: Underreaction to Bad News, Costly Arbitrage, and Equity Returns. *Journal of Financial Economics*, 135(3), 725–753.
- An, B.-J., Ang, A., Bali, T. G., and Cakici, N. (2014). The Joint Cross Section of Stocks and Options. *Journal of Finance*, 69(5), 2279–2337.
- Bali, T. G., and Hovakimian, A. (2009). Volatility Spreads and Expected Stock Returns. *Management Science*, 55(11), 1797–1812.
- Bollerslev, T., Tauchen, G., and Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies*, 22(11), 4463–4492.
- Carr, P., and Wu, L. (2009). Variance Risk Premiums. *Review of Financial Studies*, 22(3), 1311–1341.
- Cremers, M., and Weinbaum, D. (2010). Deviations from Put-Call Parity and Stock Return Predictability. *Journal of Financial and Quantitative Analysis*, 45(2), 335–367.
- Garleanu, N., Pedersen, L. H., and Poteshman, A. M. (2009). Demand-Based Option Pricing. *Review of Financial Studies*, 22(10), 4259–4299.
- Pan, J., and Poteshman, A. M. (2006). The Information in Option Volume for Future Stock Prices. *Review of Financial Studies*, 19(3), 871–908.
- Xing, Y., Zhang, X., and Zhao, R. (2010). What Does the Individual Option Volatility Smirk Tell Us about Future Equity Returns? *Journal of Financial and Quantitative Analysis*, 45(3), 641–662.
