.PHONY: test lint check

# run all unit tests
test:
	python -m unittest discover -s src -p "*_test.py" -v

# run ruff linting
lint:
	ruff check --exclude "*.ipynb" .

# run linting with auto-fix
lint-fix:
	ruff check --fix --exclude "*.ipynb" .

# run all checks (lint + test)
check: lint test
