SHELL := /bin/bash
ENV_NAME := evaluation-env

setup:
	python3 -m venv $(ENV_NAME)
	source ./$(ENV_NAME)/bin/activate \
		&& pip install -r requirements.txt

source:
	source ./$(ENV_NAME)/bin/activate

clean:
	rm -rf $(ENV_NAME)
