# Reproduce

This repository is designed to reproduce the main ADAPT experiments and figures from the paper.

## Requirements

- Python 3.11+
- `pip` or Conda
- Optional: Docker and Docker Compose

## Reproduce Full Results

Clone the repository and run:

```bash
git clone https://github.com/Himanshu21035/autoscaling_research
cd autoscalingresearch
make reproduce
```

This runs all main experiments across policies, workloads, and seeds, and stores the outputs in the results directory.

## Run with Docker

```bash
docker compose up --build
```

This starts the simulator and monitoring stack.

## Run a Single Experiment

```bash
pip install -r requirements.txt

python -m src.simulator.run \
  --policy mpc_lstm \
  --workload diurnal_burst \
  --cold_start 120 \
  --seed 42
```

## Outputs

The reproduction pipeline generates:

- Experiment result CSV files
- Aggregated metrics
- Paper figures

## Notes

- All experiments are deterministic for a fixed seed.
- Default settings are stored in `configs/default.yaml`.
- The main evaluation uses 3 policies, 6 workloads, and 5 random seeds.