"""
Routing algorithm — the brain of the KV-cache-aware router.

Given a set of healthy replicas and a prefix hash, selects the best
replica using a scoring function that balances:

  score(replica) = cache_hit_bonus - load_penalty

Cache hit bonus: if a replica has the prefix cached, it gets +CACHE_BONUS.
Load penalty:    in_flight * LOAD_WEIGHT — penalises busy replicas.

This means:
  - A replica with a cache hit but moderate load beats a cold idle replica.
  - A replica with a cache hit but extreme load may lose to a cold replica.

The crossover point (load_penalty > cache_bonus) is intentional: we don't
want to pile all traffic onto one replica just because it has a warm cache.

Fallback: if no cache hit exists anywhere, we route to the least-loaded
healthy replica (standard least-connections load balancing).
"""

import logging
import random
from typing import Optional

from backend_pool import BackendPool, Replica
from prefix_tracker import PrefixTracker

logger = logging.getLogger("kv-router.routing")

# Tunable constants
CACHE_HIT_BONUS = 100.0   # score bonus for a confirmed cache hit
LOAD_WEIGHT     = 10.0    # score penalty per in-flight request


def select_replica(
    prefix_hash: str,
    pool: BackendPool,
    tracker: PrefixTracker,
) -> Optional[Replica]:
    """
    Choose the best replica for a request with this prefix hash.
    Returns None if no healthy replicas are available.
    """
    healthy = pool.all_healthy()
    if not healthy:
        return None

    # Which replicas have this prefix cached?
    cached_replicas = set(tracker.replicas_with_prefix(prefix_hash))

    if cached_replicas:
        logger.debug(
            "prefix=%s cache_hit on replicas=%s",
            prefix_hash, cached_replicas
        )

    # Score every healthy replica
    scored = []
    for r in healthy:
        hit_bonus = CACHE_HIT_BONUS if r.id in cached_replicas else 0.0
        load_penalty = r.in_flight * LOAD_WEIGHT
        score = hit_bonus - load_penalty
        scored.append((score, r))

    # Among replicas with the same top score, pick randomly to avoid
    # thundering-herd on a single warmed replica
    best_score = max(s for s, _ in scored)
    candidates = [r for s, r in scored if s == best_score]
    chosen = random.choice(candidates)

    route_type = "CACHE_HIT" if chosen.id in cached_replicas else "MISS"
    logger.debug(
        "route=%s replica=%s score=%.1f",
        route_type, chosen.id, best_score
    )

    return chosen, route_type == "CACHE_HIT"
