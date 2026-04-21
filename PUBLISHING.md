# Publishing this repository

This folder is designed to be a standalone public repository.

## 1. Final pre-publish checks

- Review `README.md`
- Review `LICENSE`
- Review the package name in `pyproject.toml`
- Run the smoke test:

```bash
python -m pytest
```

## 2. Create the GitHub repository

Create an empty public repository on GitHub, for example:

- `rsfphate`
- or `rsfphate-survival`

Do not add a README or license from the GitHub UI, because they already exist here.

## 3. Publish with Git

From inside this folder:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:YOUR-ACCOUNT/YOUR-REPO.git
git push -u origin main
```

## 4. Alternative with GitHub CLI

If you use `gh`:

```bash
git init
git add .
git commit -m "Initial public release"
gh repo create YOUR-ACCOUNT/YOUR-REPO --public --source=. --remote=origin --push
```

## 5. Optional PyPI publication

If you later want to distribute the package on PyPI:

```bash
python -m pip install build twine
python -m build
python -m twine upload dist/*
```

Before that, make sure:

- the package name in `pyproject.toml` is available,
- the version is correct,
- and the README renders correctly on PyPI.

## 6. Recommended release checklist

- Tag the first release as `v0.1.0`
- Add a short GitHub repository description
- Add topics such as `survival-analysis`, `clustering`, `phate`, `random-survival-forest`
- Enable GitHub Actions later if you want CI, but it is not required for the initial release

