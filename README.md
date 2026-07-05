# World Cup 2026 Match Predictor

A from-scratch statistical model that predicts international football match outcomes using a Poisson attack/defense framework with a Dixon-Coles low-score correction, fitted on ~3,600 real international matches (2023–2026) and applied to live 2026 FIFA World Cup Round of 16 fixtures.

Built as an independent portfolio project — data collection, cleaning, feature engineering, model fitting, backtesting, and tournament simulation all done end-to-end in Python.

## Why this project

Most publicly available football datasets are either too broad (150 years of history, including eras of football that bear little resemblance to the modern game) or come with no attached modeling work. This project narrows to a recent, relevant window (2023 onward) and builds a transparent, reproducible probabilistic model on top of it — rather than a black-box classifier — so every prediction can be traced back to an interpretable team strength number.

## Data

Source: international football results dataset (Wikipedia, RSSSF, and federation websites), covering matches from 1872 to present.

**Filtering applied:**
- Restricted to matches from **2023-01-01 onward** — recent form is more predictive than long-run history, given how much the sport changes over decades.
- Restricted to **FIFA member associations only** — the source data includes ~27 non-FIFA entities (CONIFA-style regional/stateless teams such as Tibet, Sápmi, Kernow) which can never appear in an actual World Cup and were excluded.
- Two exact-duplicate match rows (data entry errors) were identified and removed.

**Final training set:** 3,622 matches across 223 FIFA-eligible teams, plus 7 real, unplayed Round of 16 fixtures used as live prediction targets.

## Feature engineering

| Feature | Description |
|---|---|
| Rolling form | Each team's average goals scored/conceded over their prior 5 matches, computed with no data leakage (shifted so a match never sees its own result) |
| Head-to-head record | Historical win rate between the specific pair of teams in a matchup. **Caveat:** median number of prior meetings per pair is 0 — most matchups have no shared history in this window, so this feature carries real signal only for frequently-paired regional rivals |
| Tournament weight | A 4-tier competitiveness weight (World Cup/continental finals = 1.0, qualifiers = 0.85, regional nations leagues = 0.7, friendlies/other = 0.5), used to scale each match's influence when fitting team strength |

## Model

**Poisson attack/defense model** (Dixon & Coles, 1997 framework):

- Each team has an *attack strength* and *defense strength*, estimated relative to a fixed reference team (Mexico, the team with the most matches in the dataset), via maximum likelihood.
- Expected goals for a match = league-average goals × attacking team's attack strength × opposing team's defense strength × home advantage (if applicable).
- A single global **home advantage multiplier** (fitted: **1.26**) and **Dixon-Coles correlation parameter, rho** (fitted: **-0.089**) correct for the known tendency of independent Poisson models to underestimate low-scoring draws (0-0, 1-0, 0-1, 1-1).
- Home advantage is only applied at non-neutral venues — most World Cup knockout matches are neutral, so this correctly switches off for the majority of tournament fixtures.

Fitted via `scipy.optimize` (L-BFGS-B for the 446 team-strength parameters, Nelder-Mead for rho).

## Backtest

| Metric | Model | Coin-flip baseline |
|---|---|---|
| Ranked Probability Score (↓ better) | 0.102 | ~0.24 |
| Log-loss (↓ better) | 0.81 | ~1.10 |
| Correct result (W/D/L) | 62.8% | 33% |

**Important caveat:** this is an **in-sample** backtest — the model's parameters were fitted using all 3,622 matches at once, including information "from the future" relative to any individual match being scored. A fully rigorous evaluation would use a walk-forward design (refitting periodically using only data available at each point in time, then scoring only subsequent matches), which would likely show somewhat weaker — but more honest — performance. This is a known limitation of the current version, not a hidden one.

## Live predictions (2026 World Cup, Round of 16)

Generated for the actual, unplayed fixtures at time of writing:

| Match | Home Win | Draw | Away Win |
|---|---|---|---|
| Brazil vs Norway | 42.1% | 25.4% | 32.5% |
| Mexico vs England | 19.8% | 32.1% | 48.1% |
| Portugal vs Spain | 24.3% | 29.4% | 46.3% |
| United States vs Belgium | 22.4% | 24.6% | 53.0% |
| Argentina vs Egypt | 65.4% | 26.3% | 8.2% |
| Switzerland vs Colombia | 25.3% | 29.8% | 44.9% |
| France vs Morocco | 39.3% | 37.3% | 23.4% |

## Tournament simulation

Extending beyond the immediate round requires the real bracket structure — who plays whom in the quarterfinals is a tournament fact, not something derivable from team strength alone, so this was confirmed against official sources rather than assumed. Using the confirmed bracket, a 20,000-run Monte Carlo simulation gives semifinal-reaching probabilities for this side of the draw:

| Team | Reaches semifinal |
|---|---|
| France | 58.3% |
| Argentina | 50.6% |
| Spain | 46.2% |
| Morocco | 41.7% |
| England | 39.0% |
| Colombia | 27.7% |
| Brazil | 26.6% |
| Portugal | 25.1% |
| Belgium | 22.8% |
| Norway | 20.6% |
| Switzerland | 15.3% |
| Mexico | 13.8% |
| Egypt | 6.5% |
| United States | 5.9% |

Knockout draws are resolved via a simulated penalty shootout (modeled as a 50/50 coin flip — consistent with football analytics literature showing shootout outcomes are largely unpredictable from team strength).

## Known limitations

- Backtest is in-sample; a walk-forward re-evaluation would give a more honest measure of real predictive skill.
- Head-to-head feature is sparse for most team pairs and currently contributes limited signal.
- Tournament competitiveness weights are a reasoned first-pass judgment call, not empirically derived — a natural next step is tuning them against backtest performance.
- Home-advantage and rho are fitted once globally, not per-confederation or per-team, which may understate variation (e.g. some teams may have unusually strong or weak home effects).

## Roadmap

- [ ] Walk-forward backtest for an honest out-of-sample performance measure
- [ ] Extend bracket simulation through semifinals and final once the full draw is known
- [ ] Analytical gradient for the optimizer (should cut fitting time from minutes to seconds)
- [ ] Track predictions against real results as the tournament concludes

## Tech stack

Python, pandas, NumPy, SciPy (`optimize`, `stats`)

## Author

Dominic Mwasya
