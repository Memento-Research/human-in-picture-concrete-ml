.PHONY: deps

deps:
	python3.9 -m venv cnn_venv
	. cnn_venv/bin/activate && pip install -r requirements.txt