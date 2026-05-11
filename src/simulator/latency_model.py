# src/simulator/latency_model.py

# Maximum latency returned when system is at/over capacity
MAX_LATENCY_MS = 10_000.0


def mm1_latency(
    rps: float,
    capacity: float,
    base_latency_ms: float = 50.0,
) -> float:
    """
    M/M/1 queueing model: latency = base / (1 - utilization)
    
    Args:
        rps: current request rate (requests/second)
        capacity: total system capacity (requests/second)
        base_latency_ms: service time with zero load (ms)
    
    Returns:
        latency in milliseconds
    
    Note: As rps → capacity, latency → infinity.
          Returns MAX_LATENCY_MS when rps >= capacity.
    """
    if capacity <= 0:
        return MAX_LATENCY_MS

    utilization = rps / capacity

    if utilization >= 1.0:
        return MAX_LATENCY_MS

    # M/M/1: E[T] = service_time / (1 - rho)
    latency = base_latency_ms / (1.0 - utilization)

    # Cap at MAX to avoid numerical explosion near saturation
    return min(latency, MAX_LATENCY_MS)