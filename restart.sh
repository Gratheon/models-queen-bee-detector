#!/bin/bash
cd /www/models-queen-bee-detector/

COMPOSE_PROJECT_NAME=gratheon docker-compose down
COMPOSE_PROJECT_NAME=gratheon docker-compose up -d --build
