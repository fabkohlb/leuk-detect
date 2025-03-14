SHELL := /bin/bash
.ONESHELL:
.PHONY: setup data clean api # data and clean are not files but executables
include .env

setup:
# direnv
	sudo apt install direnv -y
	eval "$$(direnv hook bash)"
	source ~/.bashrc
	echo dotenv >> .envrc && direnv allow

# language settings
	sudo apt install locales -y
	echo "en_US.UTF-8 UTF-8" | sudo tee -a /etc/locale.gen
	sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# pyenv
	curl -fsSL https://pyenv.run | bash
	echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bashrc
	echo '[[ -d $$PYENV_ROOT/bin ]] && export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bashrc
	echo 'eval "$$(pyenv init - bash)"' >> ~/.bashrc
	echo 'eval "$$(pyenv virtualenv-init -)"' >> ~/.bashrc
	source ~/.bashrc

# python
	sudo apt install -y make build-essential libssl-dev zlib1g-dev
	sudo apt install -y libbz2-dev libreadline-dev libsqlite3-dev wget curl
	sudo apt install -y llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
	sudo apt install -y libffi-dev liblzma-dev python3-openssl

	source ~/.bashrc
	pyenv install 3.10.6

# project dependencies
	pyenv virtualenv 3.10.6 leuk-detect
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# Download the data from Google Cloud Storage and unzip it
data:
	mkdir -p $(DATA_DIR)
	gsutil cp $(BUCKET)/$(ZIP_FILE) .
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)
	rm -f $(ZIP_FILE)

# Split the data into train and test
	mv $(DATA_DIR)/AML-Cytomorphology_LMU $(DATA_DIR)/train
	mkdir -p $(DATA_DIR)/test
	mkdir -p $(DATA_DIR)/validation
	python ml_logic/train_val_test_split.py

# Remove the data directory
clean:
	rm -rf $(DATA_DIR)

api:
	uvicorn api.simple_api:app --reload
