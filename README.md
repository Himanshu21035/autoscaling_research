# Predictive Autoscaling with Cold Start Co-Optimization

**Phase 1 Research Project** — ML Systems

## What This Is
A discrete-time simulation framework evaluating forecasting + optimization method 
combinations for cloud autoscaling, with explicit cold start delay modeling.

**Core contribution:** ADAPT-MPC — a Model Predictive Controller that estimates 
cold start delay dynamically (via EMA) instead of using a fixed constant.

## Quick Start
```bash
git 
cd autoscaler-research
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
