PYTHON = .venv/bin/python

install:
	$(PYTHON) -m pip install -r api/requirements.txt
	$(PYTHON) -m pip install pytest httpx ruff

run:
	cd api && ../$(PYTHON) -m uvicorn app:app --reload

test:
	$(PYTHON) -m pytest api/tests

lint:
	$(PYTHON) -m ruff check api/

docker-build:
	docker build -t mushroom-api api/
