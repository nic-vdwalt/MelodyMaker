# Makefile

.PHONY: clean

clean:
	# remove Python bytecode files
	find . -type f -name '*.py[co]' -delete
	# remove all __pycache__ directories
	find . -type d -name '__pycache__' -exec rm -rf {} +
	# remove packaging/build artifacts
	rm -rf build/ dist/ *.egg-info
