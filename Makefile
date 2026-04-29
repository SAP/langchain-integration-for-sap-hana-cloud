.PHONY: all help
.PHONY: format format_tests format_diff format_examples
.PHONY: lint lint_tests lint_diff lint_examples
.PHONY: test tests integration_test integration_tests test_watch
.PHONY: check_imports

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

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=langchain_hana
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=langchain_hana
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d HEAD~1 | grep -E '\.py$$|\.ipynb$$')
lint_tests format_tests: PYTHON_FILES=tests
lint_tests format_tests: MYPY_CACHE=.mypy_cache_test
lint_examples format_examples: PYTHON_FILES=examples

lint lint_diff lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_examples:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff

format format_diff format_tests format_examples:
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
	@echo 'format                                 - format package code (langchain_hana)'
	@echo 'format_tests                           - format tests/'
	@echo 'format_diff                            - format only files changed since last commit'
	@echo 'format_examples                        - format examples/ (notebooks)'
	@echo 'lint                                   - lint package code (langchain_hana)'
	@echo 'lint_tests                             - lint tests/'
	@echo 'lint_diff                              - lint only files changed since last commit'
	@echo 'lint_examples                          - lint examples/ (notebooks)'
	@echo 'test                                   - run unit tests'
	@echo 'tests                                  - alias for "test" target'
	@echo 'test TEST_FILE=<test_file>             - run all unittests in file'
	@echo 'test_watch                             - watch for file changes and re-run tests'
	@echo 'integration_test                       - run integration tests'
	@echo 'integration_tests                      - alias for "integration_test" target'
	@echo 'integration_test TEST_FILE=<test_file> - run integration tests in a file'
