.ONESHELL:
.SHELLFLAGS: -ec -o pipefail
SHELL := /bin/bash
.PHONY: setup data clean # data and clean are not files but executables
include .env

setup:
# direnv
	sudo apt install direnv -y
	echo 'eval "$$(direnv hook bash)"' >> ~/.bashrc
	echo 'dotenv' > .envrc && direnv allow || true

# language settings
	sudo apt install locales -y
	echo "en_US.UTF-8 UTF-8" | sudo tee -a /etc/locale.gen
	sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# pyenv dependencies
	sudo apt install -y make build-essential libssl-dev zlib1g-dev \
		libbz2-dev libreadline-dev libsqlite3-dev wget curl \
		llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
		libffi-dev liblzma-dev python3-openssl

# pyenv installation
	curl -fsSL https://pyenv.run | bash
	echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bashrc
	echo '[[ -d $$PYENV_ROOT/bin ]] && export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bashrc
	echo 'eval "$$(pyenv init - bash)"' >> ~/.bashrc
	echo 'eval "$$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Set up environment variables for current shell
	export PYENV_ROOT="$$HOME/.pyenv"
	export PATH="$$PYENV_ROOT/bin:$$PATH"
	eval "$$($$PYENV_ROOT/bin/pyenv init -)"
	eval "$$($$PYENV_ROOT/bin/pyenv virtualenv-init -)"

# Python installation
	$$PYENV_ROOT/bin/pyenv install 3.10.6

# Project dependencies
	$$PYENV_ROOT/bin/pyenv virtualenv 3.10.6 leuk-detect
	$$PYENV_ROOT/bin/pyenv local leuk-detect
	$$PYENV_ROOT/versions/leuk-detect/bin/python -m pip install --upgrade pip
	$$PYENV_ROOT/versions/leuk-detect/bin/pip install -r requirements.txt

# Also download the data
	make data

# Download the data from Google Cloud Storage and unzip it
data:
	mkdir -p $(DATA_DIR)
	gsutil cp gs://$(BUCKET_NAME)/$(ZIP_FILE) .
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)
	rm -f $(ZIP_FILE)

# Split the data into train and test
	mv $(DATA_DIR)/AML-Cytomorphology_LMU $(DATA_DIR)/train
	mkdir -p $(DATA_DIR)/test
	mkdir -p $(DATA_DIR)/validation

# Initializing pyenv
	export PYENV_ROOT="$$HOME/.pyenv"
	export PATH="$$PYENV_ROOT/bin:$$PATH"
	eval "$$($$PYENV_ROOT/bin/pyenv init -)"
	eval "$$($$PYENV_ROOT/bin/pyenv virtualenv-init -)"

# Load .env file
	set -a; source .env; set +a
# running the split script
	$$PYENV_ROOT/versions/leuk-detect/bin/python ml_logic/train_val_test_split.py

# make models directory
	mkdir models

# Remove the data directory
clean:
	rm -rf $(DATA_DIR)

api:
	uvicorn api.simple_api:app --reload
