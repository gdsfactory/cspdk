install:
	uv sync --extra docs --extra dev

test:
	uv run pytest -s tests/test_si220_cband.py
	# uv run pytest -s tests/test_si500.py
	# uv run pytest -s tests/test_sin300.py

test-force:
	uv run pytest -s tests/test_si220.py --force-regen
	uv run pytest -s tests/test_si500.py --force-regen
	uv run pytest -s tests/test_sin300.py --force-regen

test-fail-fast:
	uv run pytest -s tests/test_si220.py -x
	uv run pytest -s tests/test_si500.py -x
	uv run pytest -s tests/test_sin300.py -x

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
	uv run python .github/write_cells_si220_cband.py
	uv run python .github/write_cells_si500.py
	uv run python .github/write_cells_sin300.py
	uv run jb build docs

.PHONY: drc doc docs
