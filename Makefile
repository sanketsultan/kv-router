.PHONY: install backends router bench-naive bench-router bench health

## Install Python dependencies
install:
	pip3 install -r router/requirements.txt

## Start 3 fake vLLM backends (run in separate terminals or use tmux)
backends:
	@echo "Starting 3 fake backends on :8001 :8002 :8003 ..."
	@PORT=8001 BACKEND_ID=backend-0 python3 simulator/fake_backend.py &
	@PORT=8002 BACKEND_ID=backend-1 python3 simulator/fake_backend.py &
	@PORT=8003 BACKEND_ID=backend-2 python3 simulator/fake_backend.py &
	@echo "Backends started (PIDs: $$!)"

## Start the KV-cache-aware router on :8000
router:
	BACKENDS="http://localhost:8001,http://localhost:8002,http://localhost:8003" \
	  uvicorn router.main:app --host 0.0.0.0 --port 8000 --reload

## Run the full demo: start everything + benchmark
demo: install
	@echo "Starting backends..."
	PORT=8001 BACKEND_ID=backend-0 python3 simulator/fake_backend.py &
	PORT=8002 BACKEND_ID=backend-1 python3 simulator/fake_backend.py &
	PORT=8003 BACKEND_ID=backend-2 python3 simulator/fake_backend.py &
	sleep 1
	@echo "Starting router..."
	BACKENDS="http://localhost:8001,http://localhost:8002,http://localhost:8003" \
	  uvicorn router.main:app --host 0.0.0.0 --port 8000 &
	sleep 2
	@echo "Running benchmark..."
	python3 simulator/load_gen.py --mode both --requests 60

## Benchmark: naive round-robin only
bench-naive:
	python3 simulator/load_gen.py --mode naive --requests 60

## Benchmark: cache-aware router only
bench-router:
	python3 simulator/load_gen.py --mode router --requests 60

## Benchmark: compare both side-by-side
bench:
	python3 simulator/load_gen.py --mode both --requests 60

## Check router health
health:
	curl -s http://localhost:8000/health | python3 -m json.tool

## Docker Compose (full stack)
up:
	docker compose up --build

down:
	docker compose down
