.PHONY: data clean setup # data and clean are not files but executables

include .env

# Download the data from Google Cloud Storage and unzip it
data:
	mkdir -p $(DATA_DIR)
	gsutil cp $(BUCKET)$(ZIP_FILE) .
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)
	rm -f $(ZIP_FILE)

# Split the data into train and test
	mv $(DATA_DIR)/AML-Cytomorphology_LMU $(DATA_DIR)/train
	mkdir -p $(DATA_DIR)/test
	python ml_logic/split_test_data.py

# Remove the data directory
clean:
	rm -rf $(DATA_DIR)


# Setting up a virtual machine
# sudo apt install git -y
# git clone https://github.com/fabkohlb/leuk-detect.git

setup:
	curl https://pyenv.run | bash
