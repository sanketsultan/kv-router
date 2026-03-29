"""
Prefix tracker — the core data structure of the KV-cache-aware router.

Maintains a mapping of:
    prefix_hash -> {replica_id: last_seen_timestamp}

When a request arrives, we hash its canonical prefix (system prompt +
first user message). If a replica has seen this prefix before (i.e. it
computed KV blocks for it), we route there to get a cache hit.

Eviction mirrors how vLLM's internal prefix cache works: LRU with a
configurable max-entries-per-replica bound.
"""

import hashlib
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class PrefixEntry:
    replica_id: str
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float  = field(default_factory=time.monotonic)
    hit_count: int    = 0


class PrefixTracker:
    """
    Thread-safe LRU tracker of which replicas have which prefix cached.

    Each replica maintains its own LRU of up to `max_prefixes_per_replica`
    prefix hashes. When a replica evicts a prefix (due to its own memory
    pressure), we don't hear about it directly — we rely on TTL to age
    out entries that haven't been reconfirmed by actual traffic.
    """

    def __init__(
        self,
        max_prefixes_per_replica: int = 2000,
        ttl_seconds: float = 1800.0,  # 30 min — matches vLLM default eviction window
        prefix_length_chars: int = 2000,  # how much of the prompt we fingerprint
    ):
        self.max_prefixes = max_prefixes_per_replica
        self.ttl = ttl_seconds
        self.prefix_length = prefix_length_chars

        # replica_id -> OrderedDict[prefix_hash, PrefixEntry]  (LRU order)
        self._store: dict[str, OrderedDict[str, PrefixEntry]] = defaultdict(OrderedDict)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def canonical_prefix(self, messages: list[dict]) -> str:
        """
        Extract the stable, cacheable prefix from a chat message list.

        The prefix is: everything except the last user message, plus the
        first `prefix_length_chars` chars of the last user message.
        This mirrors how vLLM identifies reusable KV blocks.
        """
        parts = []
        for i, msg in enumerate(messages):
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if i == len(messages) - 1 and role == "user":
                # Truncate the final user turn — the rest is unique per request
                content = content[: self.prefix_length]
            parts.append(f"{role}:{content}")
        return "\n".join(parts)

    def hash_prefix(self, messages: list[dict]) -> str:
        prefix = self.canonical_prefix(messages)
        return hashlib.sha256(prefix.encode()).hexdigest()[:16]

    def record_hit(self, prefix_hash: str, replica_id: str) -> None:
        """Call this after routing a request to `replica_id`."""
        with self._lock:
            lru = self._store[replica_id]
            if prefix_hash in lru:
                entry = lru.pop(prefix_hash)  # remove to re-insert at end (LRU update)
                entry.last_seen = time.monotonic()
                entry.hit_count += 1
            else:
                entry = PrefixEntry(replica_id=replica_id)
                if len(lru) >= self.max_prefixes:
                    lru.popitem(last=False)  # evict oldest
            lru[prefix_hash] = entry

    def replicas_with_prefix(self, prefix_hash: str) -> list[str]:
        """
        Return replica IDs that have this prefix cached, ordered by
        most-recently-confirmed first. Expired entries are pruned.
        """
        now = time.monotonic()
        result = []
        with self._lock:
            for replica_id, lru in self._store.items():
                if prefix_hash in lru:
                    entry = lru[prefix_hash]
                    if now - entry.last_seen <= self.ttl:
                        result.append((replica_id, entry.last_seen))
                    else:
                        # TTL expired — drop silently
                        del lru[prefix_hash]

        # Most recently confirmed cache hit first
        result.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in result]

    def stats(self) -> dict:
        """Return per-replica prefix counts for the /metrics endpoint."""
        with self._lock:
            now = time.monotonic()
            out = {}
            for replica_id, lru in self._store.items():
                live = sum(
                    1 for e in lru.values()
                    if now - e.last_seen <= self.ttl
                )
                out[replica_id] = {"cached_prefixes": live}
            return out

    def evict_replica(self, replica_id: str) -> None:
        """Call when a replica is removed from the pool."""
        with self._lock:
            self._store.pop(replica_id, None)
