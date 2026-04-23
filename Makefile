PYTHON ?= python3

.PHONY: dev-api dev-ui test eval

dev-api:
	$(PYTHON) -m uvicorn app.main:app --reload

dev-ui:
	$(PYTHON) -m streamlit run app/ui/streamlit_app.py

test:
	pytest

eval:
	$(PYTHON) scripts/run_eval.py
