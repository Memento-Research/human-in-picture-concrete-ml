
deps:
	curl -sSL https://install.python-poetry.org | python3 -
	mkdir -p results/times
	mkdir -p results/losses
	mkdir -p outputs

data:
	poetry run kaggle datasets download -d aliasgartaksali/human-and-non-human
	unzip human-and-non-human.zip
	rm human-and-non-human.zip
	mkdir -p data
	mv human-and-non-human data/human-and-non-human

run:
	./scripts/run.sh

run_multiple:
	./scripts/run_multiple.sh

benchmark: run_multiple
	./scripts/data_processing.sh
