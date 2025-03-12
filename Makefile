.PHONY: data clean # data and clean are not files but executables

include .env

# Download the data
data:
	mkdir -p $(DATA_DIR)
	gsutil cp $(BUCKET)$(ZIP_FILE) .
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)
	rm -f $(ZIP_FILE)

clean:
	rm -rf $(DATA_DIR)
