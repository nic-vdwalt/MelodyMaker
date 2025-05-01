.PHONY: clean train generate

clean:
	# remove Python bytecode files
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
	rm -f melody_model.h5 chord_model.h5 scaler.pkl seed.pkl

train:
	python src/rnn.py --train

generate:
	python src/rnn.py
