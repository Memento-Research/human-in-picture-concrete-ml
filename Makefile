.PHONY: deps

deps:
	python3 -m venv cnn_venv
	. cnn_venv/bin/activate && pip install -r requirements.txt
	mkdir -p results/times
	mkdir -p results/losses
	mkdir -p outputs

data:
	. cnn_venv/bin/activate && kaggle datasets download -d aliasgartaksali/human-and-non-human
	unzip human-and-non-human.zip
	rm human-and-non-human.zip
	mkdir -p data
	mv human-and-non-human data/human-and-non-human
