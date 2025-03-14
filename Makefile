.PHONY: data clean setup # data and clean are not files but executables

setup:
	curl -fsSL https://pyenv.run | bash
	export PYENV_ROOT="$HOME/.pyenv"
	eval "$(pyenv virtualenv-init -)"
	[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init - bash)"

	sudo apt install -y make build-essential libssl-dev zlib1g-dev
	sudo apt install -y libbz2-dev libreadline-dev libsqlite3-dev wget curl
	sudo apt install -y llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
	sudo apt install -y libffi-dev liblzma-dev python3-openssl

	pyenv install 3.10.6
	pyenv virtualenv 3.10.6 leuk-detect

	pyenv activate leuk-detect
	pip install -r requirements.txt

	cp .env.sample .env
	make data

include .env

# Download the data from Google Cloud Storage and unzip it
data:
	make clean
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
