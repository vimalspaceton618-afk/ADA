.PHONY: help test

help:
	@echo "ADA research scaffold"
	@echo "  make test  Run the configuration unit tests"

test:
	python -m pytest -q
