
DIRS := logs certs models data

# Check the code style using peop8 standard and flake8
check:
	@flake8 app/

# Build and train the model CLI
build: setup
	@python3 app/model_train_main.py

# Start the app frontend and backend
start: setup
	@python3 -m app.app_main

# Cleanup project path
clean:
	@rm -rf `find . -name __pycache__`

$(DIRS):
	mkdir -p $@

setup: $(DIRS)

.PHONY: build run check clean start prepare
#.DEFAULT_GOAL :=
