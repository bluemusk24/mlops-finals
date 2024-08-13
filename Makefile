
test:
	pytest unit_test/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -t landmine:v1 .

integration_test: build
	IMAGE_NAME=landmine:v1 bash integration_test/./run.sh

publish: build integration_test
	IMAGE_NAME=landmine:v1 bash scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install
