.PHONY: install install-dev lint test train-smoke clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,sweep]"

lint:
	ruff check synbadd/ tests/

lint-fix:
	ruff check --fix synbadd/ tests/

test:
	pytest tests/ -v

test-quick:
	pytest tests/ -v -x --timeout=30

train-smoke:
	synbadd train --config configs/base.yaml --fold 1 \
		--set training.n_epochs=2 --set system.debug=true --set wandb.enabled=false

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
