network := "gratheon_default"

ensure-network:
	@docker network inspect {{network}} >/dev/null 2>&1 || \
		docker network create \
			--label com.docker.compose.project=gratheon \
			--label com.docker.compose.network=default \
			{{network}} >/dev/null

start: ensure-network
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml up --build

start-d: ensure-network
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml up --build -d

start-prod:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.yml up --build

start-prod-d:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.yml up --build -d

stop:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml down

stop-prod:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.yml down

logs:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml logs -f models-queen-bee-detector

test:
	@if [ -x .venv/bin/pytest ]; then \
		PYTHONPATH=. .venv/bin/pytest; \
	else \
		PYTHONPATH=. python3 -m pytest; \
	fi
