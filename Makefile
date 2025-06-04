.PHONY: install run test

install:
	pip install -r requirements.txt

run:
	python -m src.pipelines.main

test:
	pytest -vv
