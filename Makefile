PYTHON := python3
PIP := pip3
VENV := .venv
UVICORN := $(VENV)/bin/uvicorn

.PHONY: help venv install train serve format lint clean

help:
	@echo "Commands:"
	@echo "  make venv     - create virtual environment"
	@echo "  make install  - install dependencies"
	@echo "  make train    - train model + save artifacts"
	@echo "  make serve    - run FastAPI server"
	@echo "  make clean    - remove artifacts + caches"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

train:
	$(VENV)/bin/python -m ml.train

serve:
	$(UVICORN) app.main:app --reload

clean:
	rm -rf artifacts __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
