# Developer Setup Documentation

## Overview

This document provides guidelines for setting up the development environment for working on this package. Additionally, it includes links to the LangChain repository for usage instructions.

---

## Developer Setup

### Prerequisites

To work on the package as a developer, you will need the following tools:

1. **Poetry**
   - This project uses Poetry v1.7.1+ as a dependency manager, as recommended by LangChain for integration packages.
   - Refer to the [Poetry installation guide](https://python-poetry.org/docs/#installation) for instructions on how to install Poetry.

2. **Make** (optional but highly recommended)
   - Make allows you to use the Makefile for tasks such as formatting, linting, spell-checking, and running tests.

### Setup Instructions

To set up the development environment, follow these steps:

1. Install Poetry and Make.
2. Clone the repository.
3. Run the following command to install all necessary dependencies:

   ```bash
   poetry install --with lint,typing,test,test_integration
   ```

   This command installs dependencies for development, including those for linting, formatting, spell-checking, and testing.

4. Configure nbstripout to prevent notebook metadata noise in git diffs:

   ```bash
   nbstripout --install --attributes .gitattributes
   git config filter.nbstripout.clean "$(git config filter.nbstripout.clean | sed 's/"$//' ) --keep-output --extra-keys \"metadata.kernelspec metadata.language_info\""
   ```

   This ensures that environment-specific notebook metadata (like kernel names and Python versions) are automatically stripped when committing notebooks, while preserving cell outputs.

5. You are now ready to work on the package!

---

## Additional Tips

- **Creating Distribution Artifacts**
  - To create distribution artifacts such as a `.whl` file and a `.tar.gz` source file, run:

    ```bash
    poetry build
    ```

- **Updating Dependencies**
  - After adding new dependencies to `pyproject.toml`, update the lock file by running:

    ```bash
    poetry lock
    ```

- **Pushing to PyPI**
  - To upload the package to PyPI, follow these steps:
    1. Ensure you have an account on [PyPI](https://pypi.org/) and have the appropriate credentials.
    2. Build the distribution artifacts:

       ```bash
       poetry build
       ```

    3. Publish the package to PyPI:

       ```bash
       poetry publish --build
       ```

- **Changing the Version**
  - To update the package version, use Poetry's versioning command:
  
  ```bash
   poetry version <new_version>
   ```

    Replace `<new_version>` with the desired version (e.g., `1.0.1`). This will update the `pyproject.toml` file automatically.

---

## Usage Documentation

For usage instructions and examples, please refer to the [LangChain How-to guides](https://python.langchain.com/docs/how_to/).
