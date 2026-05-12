install:
	uv venv --python 3.12
	uv sync --extra docs --extra dev

dev: install
	curl -sf https://raw.githubusercontent.com/doplaydo/pdk-ci-workflow/main/templates/.pre-commit-config.yaml -o .pre-commit-config.yaml
	uv run pre-commit install

rm-samples:
	rm -rf cspdk/si220/oband/samples cspdk/si220/cband/samples cspdk/sin300/samples cspdk/si500/samples

test:
	uv run pytest -s tests/test_si220_cband.py
	uv run pytest -s tests/test_si220_oband.py
	uv run pytest -s tests/test_routing.py
	# uv run pytest -s tests/test_si500.py
	# uv run pytest -s tests/test_sin300.py

test-ports:
	uv run pytest -s tests/test_si220_cband.py::test_optical_port_positions tests/test_si220_oband.py::test_optical_port_positions tests/test_si500.py::test_optical_port_positions tests/test_sin300.py::test_optical_port_positions

test-force:
	uv run pytest -s tests/test_si220_cband.py --update-gds-refs --force-regen
	uv run pytest -s tests/test_si220_oband.py --update-gds-refs --force-regen
	# uv run pytest -s tests/test_si500.py --update-gds-refs --force-regen
	# uv run pytest -s tests/test_sin300.py --update-gds-refs --force-regen

test-fail-fast:
	uv run pytest -s tests/test_si220_cband.py -x
	uv run pytest -s tests/test_si500.py -x
	uv run pytest -s tests/test_sin300.py -x

update-pre:
	pre-commit autoupdate

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
