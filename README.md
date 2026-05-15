<div align="center">

# ADAPT — Adaptive Duration Approximation for Predictive Timing

**A self-calibrating proactive autoscaler for Kubernetes that treats container cold-start duration as a live measurement, not a static constant.**

[
[
[
[

</div>

***

## The Problem

Standard Kubernetes HPA is reactive — it fires a scaling decision only after overload is detected. For containers with cold-start delays of 120–600 s (ML inference, GPU workloads), that reaction comes too late.

```
12:03 PM  Traffic forecast: 300 RPS arriving in 2 minutes
12:05 PM  HPA detects overload → triggers scale-out
12:07 PM  New replicas ready (2-min cold start)
12:05–07  System overloaded — SLA violated ❌

ADAPT at 12:03 PM: "cold start ≈ 120 s → scale NOW"
12:05 PM  New replicas already ready
          No violation, no wasted headroom ✅
```

The deeper issue: every proactive autoscaler published to date treats cold-start duration as a **static deployment-time constant**, even though real cloud environments exhibit ±30–60% boot-time variance. ADAPT closes this gap.

***

## What ADAPT Does

Three tightly connected components form a closed loop:

| Component | Role |
|-----------|------|
| **ADAPT estimator** | Online EWMA over observed replica graduation events — tracks cold-start duration at runtime instead of hardcoding it |
| **FH-OPT** | Converts the live ADAPT estimate into a dynamic MPC planning horizon `h*(t) = ⌈Δ̂(t)/τ⌉ + ε`, updated every control step |
| **MPC policy** | Multi-objective optimizer balancing SLA protection, cost, and stability; pluggable forecasting backend (Prophet or LSTM) |

The ADAPT estimate updates every time a warming replica batch graduates to active service, so the planning horizon automatically stretches or shrinks as cluster conditions change — no manual tuning required.

***

## Architecture

The diagram below shows the closed-loop data flow across the three components.

![Architecture](https://github.com/user-attachments/assets/d54381ed-913a-4e27-a2c7-c909cd6bdbd9)


At each timestep the loop runs as follows:

```
Forecaster           →  λ̂(t+1 … t+h*)          [RPS forecast vector]
        ↓
ADAPT estimator      →  Δ̂(t)                    [live cold-start estimate]
        ↓
FH-OPT               →  h*(t) = ⌈Δ̂(t)/τ⌉ + ε   [adaptive horizon]
        ↓
MPC policy           →  n*(t) = argmin [ λ_sla·v(t) + λ_cost·n(t)/n_max + λ_stab·|Δn| ]
        ↓
Scaler               →  patch replicas
        ↓
Cold-start tracker   →  observe Δ_obs when replica batch graduates → update ADAPT
```

ADAPT is initialized to the configured prior Δ₀ (default 120 s) and typically converges to within 10% of the true value after 8–10 graduation events.

***

## Results

Evaluated across **3 policies × 6 workload archetypes × 5 random seeds** with Wilcoxon signed-rank significance testing.

| Workload | HPA (reactive) | MPC + Prophet | MPC + LSTM |
|----------|:--------------:|:-------------:|:----------:|
| Smooth | 7.1% | 2.3% | **1.8%** |
| Bursty | 12.4% | 5.6% | **3.2%** |
| Bimodal | 15.3% | 28.7% | **4.1%** |
| Diurnal burst | 18.9% | 6.4% | **4.7%** |
| Flash crowd | 19.2% | 7.1% | **4.9%** |
| Slow ramp-up | 8.3% | 3.1% | **2.4%** |

*SLA violation rate (%) — mean over 5 seeds, 95% CI in paper Table II. Lower is better.*

**Key finding:** MPC + LSTM achieves < 5% SLA violation on all workloads vs 7–19% for reactive HPA. Proactive scaling becomes clearly worthwhile once cold-start duration exceeds ~120 s; LSTM begins to justify its overhead over Prophet at ~180 s.

***

## Figures

### SLA Violation Rate by Policy and Workload

> Per-workload SLA violation rate (%) with 95% CI error bars over 5 seeds.

![sla_bars](https://github.com/user-attachments/assets/a5d09808-3fa9-4d32-a1ac-b72d464b30e5)


***

### ADAPT Convergence

> ADAPT estimate Δ̂(t) converging to the ground-truth cold-start of 120 s on a diurnal-burst trace. Shaded region shows ±1 SD from Welford online variance.

![cold_start](https://github.com/user-attachments/assets/a6fb7f2a-7732-4f0f-a01a-e4001b7c64ee)


***

### FH-OPT Ablation

> Paired SLA violation rates with FH-OPT enabled vs fixed horizon `hf = 2`, across 5 seeds on diurnal-burst and flash-crowd workloads.

![fh-opt ablation](https://github.com/user-attachments/assets/5379c605-a707-417d-8c2b-8478b4421d64)


***

### Cold-Start Sensitivity

> SLA violation rate (%) as a function of cold-start duration (30–300 s) for each policy. HPA degrades sharply beyond 60 s; MPC + LSTM remains robust to 180 s.

![heatmap](https://github.com/user-attachments/assets/c0f653f3-d525-4e9e-90f8-55520194db0e)


***

### Cost vs SLA Trade-off (Pareto Frontier)

> Total cost (replica-minutes) vs mean SLA violation rate (%) for all policy-workload configurations. MPC variants dominate HPA on the Pareto frontier.

![latency_cost](https://github.com/user-attachments/assets/25aab028-4bac-44cf-8e8a-500f7697beb9)

***

## Grafana Dashboard

> Live experiment monitoring — ADAPT estimate, active/warming replica counts, SLA violation rate, and per-step latency in a single view.

![Grafana](https://github.com/user-attachments/assets/3a17e15e-92f9-4667-9005-1fe684696437)

***

## Repository Layout

```
autoscalingresearch/
├── README.md
├── Makefile                        # make reproduce → runs all experiments
├── paper/
│   ├── ADAPT_Conference.pdf        # Camera-ready paper
│   ├── ADAPT_Conference.tex        # Main LaTeX source
│   ├── sections/                   # Per-section .tex files
│   └── figures/                    # All vector plots + source CSVs
├── src/
│   ├── adapt/
│   │   ├── estimator.py            # EWMA cold-start estimator + Welford variance
│   │   └── fh_opt.py               # FH-OPT horizon derivation
│   ├── forecasters/
│   │   ├── base.py                 # Pluggable interface
│   │   ├── prophet_forecaster.py
│   │   └── lstm_forecaster.py
│   ├── optimizer/
│   │   └── mpc_policy.py           # Multi-objective MPC decision loop
│   └── simulator/
│       ├── simulator.py            # Discrete-time M/M/1 simulator
│       └── workloads.py            # Six synthetic workload generators
├── notebooks/
│   ├── paper_figures/              # All figures used in the paper
│   └── 04_visualization.ipynb     # Generates all paper figures
├── configs/
│   └── default.yaml                # All hyperparameters in one file
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
└── tests/
    └── ...                         # pytest suite (>80 % coverage)
```

***

## Quick Start

### Option 1 — Reproduce all paper results (recommended)

```bash
git clone https://github.com/Himanshu21035/autoscalingresearch
cd autoscalingresearch
make reproduce
# Runs 3 policies × 6 workloads × 5 seeds = 90 experiments
# Output: results/ directory with CSVs and all 5 figures
# Expected time: ~30 minutes on an 8-core machine
```

### Option 2 — Docker Compose (includes Prometheus + Grafana)

```bash
docker compose up --build
# Simulator runs at localhost:8000
# Grafana dashboard at localhost:3000  (admin / admin)
# Prometheus at localhost:9090
```

### Option 3 — Run a single experiment manually

```bash
pip install -r requirements.txt

python -m src.simulator.run \
  --policy mpc_lstm \
  --workload diurnal_burst \
  --cold_start 120 \
  --seed 42
```

***

## Configuration

All experiment parameters live in `configs/default.yaml`:

```yaml
simulator:
  tau: 60               # Decision timestep (seconds)
  sla_latency_ms: 500   # L* — SLA threshold
  capacity_per_replica: 20  # RPS per replica

adapt:
  alpha: 0.3            # EWMA smoothing factor
  delta_min: 30         # Minimum cold-start bound (s)
  delta_max: 600        # Maximum cold-start bound (s)
  delta_0: 120          # Prior before first observation

fh_opt:
  epsilon: 1            # Safety buffer (timesteps)

mpc:
  lambda_sla: 10.0      # SLA penalty weight
  lambda_cost: 1.0      # Cost penalty weight
  lambda_stab: 0.5      # Stability penalty weight
  gamma: 1.1            # Forecast margin

evaluation:
  seeds: [42, 123, 456, 789, 1337]
  workloads: [smooth, bursty, bimodal, diurnal_burst, flash_crowd, slow_rampup]
  cold_starts: [30, 60, 120, 180, 300]
```

***

## Workloads

Six synthetic archetypes, each generated across 5 random seeds (500 steps ≈ 8.3 hours):

| Workload | Characteristics |
|----------|-----------------|
| `smooth` | Slow sinusoidal ramp, low variance |
| `bursty` | Poisson arrivals with periodic spikes |
| `bimodal` | Two distinct load levels with random switching |
| `diurnal_burst` | Daily pattern with a sharp morning peak |
| `flash_crowd` | Sudden 3× spike of short duration |
| `slow_rampup` | Monotone increase over the full trace |

Train / validation / test split: 70% / 10% / 20%. All policies are evaluated on the held-out test split only.

***

## Comparison with Prior Work

| System | Multi-forecast | Multi-policy | Dynamic cold start | Public data | Open source |
|--------|:-:|:-:|:-:|:-:|:-:|
| Google Autopilot | ✗ | ✗ | ✗ | ✗ | ✗ |
| AWS Predictive Scaling | ✗ | ✗ | ✗ | ✗ | ✗ |
| MPC Cloud (Rajkumar et al.) | ✗ | ✗ | ✗ | ✗ | ✓ |
| Showar | ✗ | ∼ | ✗ | ✗ | ✓ |
| Survey (47 systems) | ∼ | ∼ | ✗ | ∼ | ✗ |
| **ADAPT (this work)** | **✓** | **✓** | **✓** | **✓** | **✓** |

***

## Limitations

- All results are from a simulator, not a live Kubernetes cluster. The M/M/1 latency model is a reasonable approximation for CPU-bound services but underestimates tail latency for GPU-bound inference.
- Cold-start duration is fixed per run rather than drawn from a distribution, which makes ADAPT's job easier than in real deployments. The 120 s / 180 s crossover thresholds should be treated as indicative, not universal.
- ADAPT goes stale during long over-provisioned periods (no graduation events → no estimate update). A periodic decay toward the prior would address this and is left as future work.

***

## Paper

> Himanshu Singh. **ADAPT: A Self-Calibrating Proactive Autoscaler for Container Orchestration.** Department of Computer Engineering, J.C. Bose University of Science and Technology, India.


Full paper PDF is in the `paper/` directory.

***

## Future Directions

- Real Kubernetes cluster validation (replace simulator execution layer with `kubernetes` Python client)
- KEDA queue-aware baseline for fairer comparison on message-queue workloads
- Periodic ADAPT decay toward prior during quiet periods
- Multi-service cold-start co-optimization
- Uncertainty-aware MPC using the Welford variance already tracked by ADAPT

***

## License

MIT — see [LICENSE](LICENSE).