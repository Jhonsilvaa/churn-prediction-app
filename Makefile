install:
	@poetry install

format:
	@isort .
	@blue .

lint:
	@blue . --check
	@isort . --check
	@pydocstyle .

sec:
	@pip-audit

run:
	@poetry run streamlit run app/app.py