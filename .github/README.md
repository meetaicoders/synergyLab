# GitHub Actions for SynergyLab

This directory contains GitHub Actions workflows for testing and publishing the SynergyLab package.

## Available Workflows

1. **Test Python Package** (`python-test.yml`)

   - Runs on push to main/master and on pull requests
   - Tests on Python 3.10, 3.11, and 3.12
   - Runs linting and tests

2. **Publish Python Package** (`python-publish.yml`)
   - Runs when a new release is created or manually triggered
   - Tests the package before publishing
   - Builds and publishes to PyPI

## Setting Up PyPI Publishing

To publish to PyPI, you need to set up a PyPI API token:

1. Create an account on [PyPI](https://pypi.org/)
2. Generate an API token in your PyPI account settings
3. Add the token as a secret in your GitHub repository:
   - Go to your GitHub repository
   - Navigate to Settings > Secrets and variables > Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token
   - Click "Add secret"

## Publishing a New Release

To publish a new version:

1. Update the version in both `pyproject.toml` files (under `[project]` and `[tool.poetry]`)
2. Commit and push the changes
3. Create a new release on GitHub with the appropriate tag (e.g., v0.1.0)
4. The GitHub Action will automatically build and publish the package to PyPI

## Manual Publishing

If you need to manually trigger the publishing workflow:

1. Go to your GitHub repository
2. Navigate to Actions > Publish Python Package
3. Click "Run workflow"
4. Select the branch and click "Run workflow"
