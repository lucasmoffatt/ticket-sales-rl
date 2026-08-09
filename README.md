# Q-Learning Dynamic Pricing Agent

A beginner-friendly, recruiter-quality reinforcement learning project that trains a **tabular Q-learning agent** to dynamically price a fixed inventory of tickets over a limited selling period in order to maximize total revenue.

Stack: **Python · NumPy · Pandas · Gymnasium · Plotly · Streamlit · pytest**

---

## 1. Business problem

A venue has a fixed number of tickets and a fixed number of days until the event. Each day it must choose a ticket price. Price too high and demand collapses; price too low and inventory sells out early, leaving money on the table. The objective is to choose prices over time that maximize **total revenue**.

## 2. Why reinforcement learning?

Pricing is a **sequential decision** problem under uncertainty:

- Today's price affects remaining inventory and future opportunities.
- Demand is stochastic.
- The best price depends on *state* (tickets left, time left), not a single static rule.

That maps naturally to RL: observe state → choose price → receive revenue → transition.

This project focuses specifically on **tabular Q-learning** so every concept (Q-values, epsilon-greedy, Bellman updates) stays inspectable and interview-friendly.

## 3. System diagram

```text
Environment → State → Q-Learning Agent → Price → Demand → Revenue → New State
     ↑                                                                  |
     └────────────────────── reward / next obs ─────────────────────────┘
```

```mermaid
flowchart LR
    Env[Environment] --> State[State]
    State --> Agent[QLearningAgent]
    Agent --> Price[Price]
    Price --> Demand[Demand]
    Demand --> Revenue[Revenue]
    Revenue --> NewState[New State]
    NewState --> Env
```

## 4. State space

Observation vector (normalized to `[0, 1]`, no look-ahead):

| Index | Feature | Meaning |
|------:|---------|---------|
| 0 | tickets remaining / initial inventory | Stock left |
| 1 | days remaining / selling days | Time left |
| 2 | previous price / max price | Last price charged |
| 3 | previous sales / initial inventory | Yesterday's sales |

For tabular Q-learning, inventory and time are discretized into:

- Inventory: **Low / Medium / High**
- Time: **Early / Middle / Late**

## 5. Action space

Discrete prices: **$50, $75, $100, $125, $150, $175, $200**

## 6. Reward function

Default:

```text
reward = daily revenue = price × tickets_sold
```

Optional terminal penalty (configurable, default `0`):

```text
if days == 0 and unsold > 0:
    reward -= terminal_inventory_penalty × unsold
```

## 7. Demand simulation

Poisson demand with mean:

```text
λ = base_demand × demand_level × price_factor(price) × urgency_factor(days_remaining)
```

- Higher prices generally reduce expected demand
- Randomness via Poisson sampling
- Optional urgency boost near the event
- Sales capped by remaining inventory in the environment

See [`src/simulation/demand.py`](src/simulation/demand.py).

## 8. Q-learning

Tabular Q-learning implemented **from scratch** (no deep RL library):

- Epsilon-greedy exploration
- Manual Bellman / TD update  
  `Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]`
- Saved Q-table + training reward history in `models/q_learning_agent.json`

See [`src/agents/q_learning_agent.py`](src/agents/q_learning_agent.py).

## 9. Evaluation methodology

The greedy Q-learning policy is evaluated over many episodes with shared seeds (`base_seed + episode_index`).

Metrics:

- Average / median / std total revenue
- Sell-through percentage
- Average selling price
- Average unsold tickets
- Sell-out rate

## 10. Results

Default setting: **100 tickets · 20 days · demand_level=1.0 · 100 eval episodes**  
Training: **3,000 Q-learning episodes**.

| Strategy | Avg revenue | Median | Std | Sell-through | Avg price | Sell-out rate |
|----------|------------:|-------:|----:|-------------:|----------:|--------------:|
| **Q-Learning** | **~$16,758** | ~$16,812 | ~$1,037 | ~98.5% | ~$170 | ~68% |

Re-run evaluation to regenerate exact numbers and charts in [`data/`](data/).

## 11. Key findings

- The learned policy tends to charge **higher prices** when inventory pressure is low and hold/discount when stock is high late in the window (see policy heatmap).
- Near-perfect sell-through is easy in this demand regime; **revenue quality** (price mix over time) is the real objective.
- Because demand is synthetic, results demonstrate the RL pipeline — not a claim about real ticketing markets.

## 12. Limitations

- Demand is synthetic, not fitted to real ticketing data.
- Customers are not strategic or strongly heterogeneous.
- Tabular discretization discards previous-price / previous-sales detail in the Q-table state.
- Single-event, single-product setting only.

## 13. Future improvements

- Finer state discretization or function approximation
- Demand models calibrated from public event data
- Multi-event / seat-tier extensions
- Reward-shaping experiments with the terminal inventory penalty

## 14. How to run

### Setup

```bash
cd rl-dynamic-pricing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tests

```bash
pytest -q
```

### Train

```bash
PYTHONPATH=src python scripts/train_q_learning.py --episodes 3000
```

### Evaluate + charts

```bash
PYTHONPATH=src python scripts/evaluate_q_learning.py --episodes 100
```

Outputs: `data/evaluation_summary.csv`, policy heatmap, training curve, episode traces.

### Streamlit dashboard

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Controls: inventory, selling days, demand level, terminal penalty.  
Runs the trained **Q-learning** policy and shows revenue metrics + interactive charts.

### Random episode smoke test (environment only)

```bash
PYTHONPATH=src python scripts/run_random_episode.py
```

---

## Project structure

```text
rl-dynamic-pricing/
├── README.md
├── requirements.txt
├── app/streamlit_app.py
├── scripts/
│   ├── run_random_episode.py
│   ├── train_q_learning.py
│   └── evaluate_q_learning.py
├── src/
│   ├── environment/dynamic_pricing_env.py
│   ├── simulation/demand.py
│   ├── agents/q_learning_agent.py
│   ├── evaluation/
│   └── visualization/
├── models/q_learning_agent.json
├── data/                         # evaluation outputs (generated)
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_q_learning_analysis.ipynb
│   └── 03_q_learning_evaluation.ipynb
└── tests/
```

## Charts

After evaluation, open:

- `data/q_learning_training.html`
- `data/policy_heatmap.html`
- `data/revenue_summary.html`
- `data/price_over_time.html`
- `data/inventory_over_time.html`
- `data/cumulative_revenue.html`
