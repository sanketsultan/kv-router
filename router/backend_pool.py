"""
Backend pool — manages the set of live LLM inference replicas.

Tracks:
  - Health (periodic /health polling)
  - In-flight request count (used for load-aware routing)
  - Running average TTFT per replica (for observability)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger("kv-router.pool")


@dataclass
class Replica:
    id: str           # e.g. "backend-0"
    url: str          # e.g. "http://localhost:8001"
    healthy: bool = True
    in_flight: int = 0

    # Rolling average TTFT (exponential moving average, alpha=0.2)
    _ema_ttft_ms: float = 500.0
    _alpha: float = 0.2

    # Stats
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def update_ttft(self, ttft_ms: float) -> None:
        self._ema_ttft_ms = self._alpha * ttft_ms + (1 - self._alpha) * self._ema_ttft_ms

    @property
    def avg_ttft_ms(self) -> float:
        return round(self._ema_ttft_ms, 1)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 3) if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "healthy": self.healthy,
            "in_flight": self.in_flight,
            "avg_ttft_ms": self.avg_ttft_ms,
            "total_requests": self.total_requests,
            "cache_hit_rate": self.cache_hit_rate,
        }


class BackendPool:
    def __init__(self, replicas: list[dict], health_interval: float = 10.0):
        """
        replicas: list of {"id": "backend-0", "url": "http://..."}
        """
        self._replicas: dict[str, Replica] = {
            r["id"]: Replica(id=r["id"], url=r["url"])
            for r in replicas
        }
        self._health_interval = health_interval
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        )
        asyncio.create_task(self._health_loop())
        logger.info("Backend pool started with %d replicas", len(self._replicas))

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_healthy(self) -> list[Replica]:
        return [r for r in self._replicas.values() if r.healthy]

    def get(self, replica_id: str) -> Optional[Replica]:
        return self._replicas.get(replica_id)

    def acquire(self, replica_id: str) -> None:
        """Increment in-flight counter before forwarding request."""
        r = self._replicas.get(replica_id)
        if r:
            r.in_flight += 1
            r.total_requests += 1

    def release(self, replica_id: str, ttft_ms: float, cache_hit: bool) -> None:
        """Decrement in-flight and update stats after request completes."""
        r = self._replicas.get(replica_id)
        if r:
            r.in_flight = max(0, r.in_flight - 1)
            r.update_ttft(ttft_ms)
            if cache_hit:
                r.cache_hits += 1
            else:
                r.cache_misses += 1

    def snapshot(self) -> list[dict]:
        return [r.to_dict() for r in self._replicas.values()]

    # ------------------------------------------------------------------
    # Health polling
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        while True:
            await asyncio.gather(*[
                self._check(r) for r in self._replicas.values()
            ])
            await asyncio.sleep(self._health_interval)

    async def _check(self, replica: Replica) -> None:
        url = f"{replica.url}/health"
        try:
            async with self._session.get(url) as resp:
                was_healthy = replica.healthy
                replica.healthy = resp.status == 200
                if not was_healthy and replica.healthy:
                    logger.info("Replica %s recovered", replica.id)
                elif was_healthy and not replica.healthy:
                    logger.warning("Replica %s became unhealthy", replica.id)
        except Exception:
            if replica.healthy:
                logger.warning("Replica %s unreachable", replica.id)
            replica.healthy = False
