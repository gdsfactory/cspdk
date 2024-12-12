install:
	pip install -e .[dev,docs]

test:
	pytest -s tests/test_si220.py
	pytest -s tests/test_si500.py
	pytest -s tests/test_sin300.py

test-force:
	pytest -s tests/test_si220.py --force-regen
	pytest -s tests/test_si500.py --force-regen
	pytest -s tests/test_sin300.py --force-regen

test-fail-fast:
	pytest -s tests/test_si220.py -x
	pytest -s tests/test_si500.py -x
	pytest -s tests/test_sin300.py -x

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	pip install build
	python -m build

jupytext:
	jupytext docs/**/*.ipynb --to py

notebooks:
	jupytext docs/**/*.py --to ipynb

docs:
	python .github/write_cells_si220.py
	python .github/write_cells_si500.py
	python .github/write_cells_sin300.py
	jb build docs

.PHONY: drc doc docs
