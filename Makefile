PY ?= python
export PYTHONPATH := src

.PHONY: install data train test serve docker smoke

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m readmit.download

train: data
	$(PY) -m readmit.pipeline --data data/raw/diabetic_data.csv --out artifacts

smoke:
	$(PY) -m readmit.pipeline --synthetic --out artifacts_synthetic --n-boot 50

test:
	$(PY) -m pytest -q

serve:
	MODEL_DIR=artifacts/model uvicorn readmit.api:app --port 8000

docker:
	docker build -t readmit-api .
	docker run --rm -p 8000:8000 readmit-api
