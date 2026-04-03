TRAIN_ARGS ?=
TRAIN_CMD := uv run --no-sync --frozen --no-dev python scripts/train.py

.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync

.PHONY: check
check: ## Run ruff checks
	@echo "🚀 Running ruff checks"
	@uv run python -m ruff check

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: data-download
data-download: ## Download HAM10000 dataset from Harvard Dataverse
	@echo "📥 Downloading HAM10000 dataset..."
	@bash scripts/download_ham10000.sh

.PHONY: model-download
model-download: ## Download the default pretrained DINOv3 backbone
	@echo "📥 Downloading pretrained DINOv3 backbone..."
	@uv run python scripts/download_model.py

.PHONY: data-split
data-split: ## Rebuild processed HAM10000 splits locally when DVC data is unavailable
	@echo "✂️ Splitting HAM10000 dataset..."
	@uv run scripts/data_processing.py

.PHONY: mlflow-up
mlflow-up: ## Start or rebuild the Docker MLflow service
	@docker compose up -d --build mlflow

.PHONY: mlflow-stop
mlflow-stop: ## Stop the Docker MLflow service but keep its state
	@docker compose stop mlflow

.PHONY: train-docker
train-docker: mlflow-up ## Run Docker training; pass extra args with TRAIN_ARGS="--epochs 1"
	@docker compose --profile train run --build --rm train $(TRAIN_CMD) $(TRAIN_ARGS)

.PHONY: train-docker-smoke
train-docker-smoke: mlflow-up ## Run a one-epoch Docker smoke test on CPU
	@docker compose --profile train run --build --rm train $(TRAIN_CMD) --epochs 1 --max-train-batches 1 --max-val-batches 1 --device cpu

.PHONY: ui-up
ui-up: ## Start or rebuild the Docker UI stack
	@docker compose --profile ui up -d --build

.PHONY: ui-down
ui-down: ## Stop the Docker UI stack but keep MLflow running
	@docker compose stop api ui

.PHONY: docker-down
docker-down: ## Remove all Docker Compose containers, networks, and named volumes
	@docker compose --profile ui --profile train down -v --remove-orphans

.PHONY: help
help: ## Show available make targets
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
