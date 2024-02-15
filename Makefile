
deps:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install --no-root
	mkdir -p results/times/image_size
	mkdir -p results/times/n_bits
	mkdir -p results/times/p_error
	mkdir -p results/losses/image_size
	mkdir -p results/losses/n_bits
	mkdir -p results/losses/p_error
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
	./scripts/run_multiple_image_size.sh
	./scripts/run_multiple_n_bits.sh
	./scripts/run_multiple_p_error.sh

data_processing:
	./scripts/data_processing.sh

benchmark: run_multiple data_processing

webapp:
	./scripts/webapp.sh

help:
	@echo "deps: install dependencies"
	@echo "data: download and unzip data"
	@echo "run: run the model"
	@echo "run_multiple: run the model with different parameters"
	@echo "data_processing: process the data"
	@echo "webapp: run the webapp"
	@echo "benchmark: run the model with different parameters and process the data"