.PHONY: install run test lint docker-build

install:
	pip install -r api/requirements.txt
	pip install pytest httpx

run:
	cd api && uvicorn app:app --reload

test:
	python -m pytest api/tests

lint:
	ruff check api/

docker-build:
	docker build -t mushroom-api api/
