.PHONY: quality style

quality:
	python -m black --check --line-length 119 --target-version py38 .
	python -m isort --check-only .
	python -m flake8 --max-line-length 119 .

style:
	python -m black --line-length 119 --target-version py38 .
	python -m isort .