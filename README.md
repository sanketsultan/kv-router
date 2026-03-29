# kv-router
Cache-aware load balancer for multi-replica LLM inference. Routes requests to the replica with the warmest KV cache, reducing TTFT by eliminating redundant prefill compute.
