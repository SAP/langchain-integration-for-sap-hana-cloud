.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/


# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

coverage:
	poetry run coverage run --source=langchain_hana -m pytest $(TEST_FILE)
	poetry run coverage html
	@echo 'Coverage report generated in htmlcov/index.html. Open this file in your browser to view the coverage.'

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=langchain_hana
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=langchain_hana
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=langchain-hana --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_hana
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_hana -name '*.py')
	poetry run python ./scripts/check_imports.py $^


######################
# HELP
######################


help:
	@echo '----'
	@echo 'check_imports                          - check imports'
	@echo 'format                                 - run code formatters (e.g., Ruff)'
	@echo 'lint                                   - run linters'
	@echo 'test                                   - run unit tests'
	@echo 'tests                                  - alias for "test" target'
	@echo 'test TEST_FILE=<test_file>             - run all unittests in file'
	@echo 'test_watch                             - watch for file changes and re-run tests'
	@echo 'integration_test                       - run integration tests'
	@echo 'integration_tests                      - alias for "integration_test" target'
	@echo 'integration_test TEST_FILE=<test_file> - run integration tests in a file'
	@echo 'lint_diff                              - lint only files changed since the last commit'
	@echo 'format_diff                            - format only files changed since the last commit'
	@echo 'coverage                               - run tests with coverage measurement and generate an HTML report'
