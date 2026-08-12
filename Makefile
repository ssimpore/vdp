PYTHON ?= python
CONFIG ?= configs/default.yaml
ENGINE ?= baseline
VIN ?= VIN006
REFERENCE_TRIP ?= VIN006_20260213T070000Z_0001
PREFIX ?= 0.50
TOP_K ?= 5
DEPARTURE ?=
ORIGIN_LAT ?=
ORIGIN_LON ?=
PREFIX_FILE ?=
HISTORY_TRIP_IDS ?=
INFERENCE_DIR ?= artifacts/demo/inference
PORT ?= 8501
VERBOSE ?=
VERBOSE_FLAG = $(if $(filter 1 true yes on,$(VERBOSE)),--verbose,)

.DEFAULT_GOAL := help
.PHONY: help setup setup-app setup-all clean trips train evaluate predict list-vins \
	run-all app notebook test check package

help:
	@$(PYTHON) -c "print('Vehicle Destination Lab\n\nSetup\n  make setup       Install core dependencies\n  make setup-app   Install core + Streamlit\n  make setup-all   Install every optional dependency\n\nPipeline\n  make clean       Clean raw telemetry and write a preconstruction audit\n  make trips       Clean data and reconstruct trip artifacts\n  make train       Train ENGINE=baseline|keras\n  make evaluate    Evaluate saved predictions\n  make run-all     Run trips, training and evaluation\n  Add VERBOSE=1 to show progress for clean/trips/train/run-all\n\nInference\n  make list-vins   List VINs and reference-trip availability\n  make predict     VIN-driven prediction + JSON and HTML map\n  Override with VIN=... REFERENCE_TRIP=... PREFIX=.25 TOP_K=10\n  Optional: DEPARTURE=... ORIGIN_LAT=... ORIGIN_LON=...\n            PREFIX_FILE=... HISTORY_TRIP_IDS=id1,id2\n\nInterfaces & QA\n  make app         Launch Streamlit on PORT=8501\n  make notebook    Launch the end-to-end Jupyter notebook\n  make test        Run all tests\n  make check       Compile, test and smoke-test the CLI\n  make package     Build a clean ZIP under dist/\n')"

setup:
	$(PYTHON) -m pip install -e .

setup-app:
	$(PYTHON) -m pip install -e ".[app]"

setup-all:
	$(PYTHON) -m pip install -e ".[all]"

# In this project, `make clean` means data cleaning. It never deletes raw data.
clean:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" clean-data $(VERBOSE_FLAG)

trips:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" build-trips $(VERBOSE_FLAG)

train:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" train --engine "$(ENGINE)" $(VERBOSE_FLAG)

evaluate:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" evaluate --engine "$(ENGINE)" --split test

list-vins:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" list-vins

predict:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" predict-vin \
		--engine "$(ENGINE)" \
		--vin "$(VIN)" \
		--reference-trip-id "$(REFERENCE_TRIP)" \
		--prefix-fraction "$(PREFIX)" \
		--top-k "$(TOP_K)" \
		$(if $(DEPARTURE),--departure-time "$(DEPARTURE)",) \
		$(if $(ORIGIN_LAT),--origin-latitude "$(ORIGIN_LAT)",) \
		$(if $(ORIGIN_LON),--origin-longitude "$(ORIGIN_LON)",) \
		$(if $(PREFIX_FILE),--prefix-file "$(PREFIX_FILE)",) \
		$(if $(HISTORY_TRIP_IDS),--history-trip-ids "$(HISTORY_TRIP_IDS)",) \
		--output-json "$(INFERENCE_DIR)/prediction.json" \
		--map-json "$(INFERENCE_DIR)/map_payload.json" \
		--map-html "$(INFERENCE_DIR)/prediction_map.html"

run-all:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" run-all --engine "$(ENGINE)" $(VERBOSE_FLAG)

app:
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" serve --port "$(PORT)"

notebook:
	$(PYTHON) -m jupyter lab notebooks/end_to_end_destination_prediction.ipynb

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) -m compileall -q vehicle_destination tests scripts streamlit_app.py
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" list-vins >/dev/null
	$(PYTHON) -m vehicle_destination --config "$(CONFIG)" predict-vin \
		--engine baseline --vin "$(VIN)" --reference-trip-id "$(REFERENCE_TRIP)" \
		--prefix-fraction "$(PREFIX)" --top-k 3 >/dev/null

package:
	$(PYTHON) scripts/package_project.py --output dist/vehicle_destination_lab_v1_6_ready.zip
