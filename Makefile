.PHONY: install train test serve docker-build docker-run lint

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python -m src.train

test:
	pytest -v --cov=src --cov=api

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t credit-risk-api .

docker-run:
	docker run --rm -p 8000:8000 credit-risk-api

lint:
	ruff check src api tests
