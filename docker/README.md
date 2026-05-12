# Step 10B — Docker + Grafana

One-command demo stack: FastAPI + Prometheus + Grafana.

## Quick Start

```bash
cd docker

# 1. Copy env (edit if needed)
cp .env.example .env

# 2. Build and start all services
docker compose up --build -d

# 3. Wait ~20s for health checks, then open:
#    API Swagger:   http://localhost:8000/docs
#    Prometheus:    http://localhost:9090
#    Grafana:       http://localhost:3000  (admin / autoscaler)
```

## Services

| Service    | URL                        | Purpose                        |
|------------|----------------------------|--------------------------------|
| API        | http://localhost:8000/docs | FastAPI Swagger UI             |
| Prometheus | http://localhost:9090      | Metrics store + query engine   |
| Grafana    | http://localhost:3000      | Policy comparison dashboard    |

## Populate the Dashboard

The dashboard shows live data as experiments run. To seed it:

```bash
# Option A — submit via API (runs in background, Grafana updates live)
curl -X POST http://localhost:8000/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"policy":"hpa","forecaster":"none","workload":"diurnal_burst"}'

curl -X POST http://localhost:8000/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"policy":"mpc","forecaster":"lstm","workload":"diurnal_burst"}'

# Option B — run smoke test (pushes all 4 configs at once)
docker exec autoscaler-api python scripts/smoke_test_step9.py
```

## Dashboard Panels

- **KPI row** — replicas, RPS, latency, SLA violated (live gauges)
- **Replicas over time** — all 4 policies overlaid
- **RPS vs Capacity** — shows over/under provisioning
- **Latency comparison** — with SLA threshold line
- **ADAPT estimate** — cold-start learning curve
- **Utilisation** — capacity efficiency per policy

## Prometheus Metrics

Scraped from `GET /v1/metrics/prometheus` every 10s:

```
autoscaler_replicas          {policy, forecaster}
autoscaler_rps               {policy, forecaster}
autoscaler_latency_ms        {policy, forecaster}
autoscaler_sla_violated      {policy, forecaster}
autoscaler_cost              {policy, forecaster}
autoscaler_utilisation       {policy, forecaster}
autoscaler_adapt_estimate_s  {policy, forecaster}
```

## Tear Down

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop + delete all data
```
