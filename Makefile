# Makefile

PYTHON = python
TRAIN_SCRIPT = src/train.py
GENERATE_SCRIPT = src/generate.py

MELODY_MODEL = melody_model.h5
CHORD_MODEL = chord_model.h5
SEED = seed.pkl

.PHONY: clean train generate install

clean:
	# remove Python bytecode files and build artifacts
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info

train:
	$(PYTHON) $(TRAIN_SCRIPT)

generate:
	$(PYTHON) $(GENERATE_SCRIPT)

install:
	pip install -r requirements.txt
