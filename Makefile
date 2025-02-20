
# Check the code style using peop8 standard and flake8
check:
	@flake8 app/

# Build and train the model CLI
build:
	@python3 app/model_train_main.py

# Run the model inference CLI
run:
	@python3 app/model_inference_main.py

# Start the app frontend and backend
start:
	@python3 -m app.app_main

# Cleanup project path
clean:
	@rm -rf `find . -name __pycache__`

.PHONY: build run check clean start
#.DEFAULT_GOAL :=
