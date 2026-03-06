.PHONY: install test lint clean run dashboard download-data docker docker-build docker-run docker-compose

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

run:
	python -m src.main

dashboard:
	streamlit run src/dashboard/app.py

download-data:
	python -c "from src.data.downloader import DatasetDownloader; DatasetDownloader().prepare_combined_dataset()"

docker: docker-build docker-run

docker-build:
	docker build -t multilingual-classifier:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data multilingual-classifier:latest

docker-compose:
	docker compose up --build
