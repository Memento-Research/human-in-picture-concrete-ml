.PHONY: deps

deps:
	python3.9 -m venv cnn_venv
	. cnn_venv/bin/activate && pip install -r requirements.txt

data:
	kaggle datasets download -d aliasgartaksali/human-and-non-human
	unzip human-and-non-human.zip
	rm human-and-non-human.zip
	mkdir data
	mv human-and-non-human data/human-and-non-human