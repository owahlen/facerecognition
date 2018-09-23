PYTHON := python
DATA_DIR := data
OUTPUT_DIR := output
LFW := lfw
LFW_DIR := $(DATA_DIR)/$(LFW)
LFW_SENTINEL := $(LFW_DIR)/.sentinel
LANDMARKS := shape_predictor_68_face_landmarks.dat
LANDMARKS_FILE := $(DATA_DIR)/$(LANDMARKS)
PREPROCESS_INPUT_DIR := $(LFW_DIR)
PREPROCESS_OUTPUT_DIR := $(OUTPUT_DIR)/intermediate
PREPROCESS_SENTINEL := $(PREPROCESS_OUTPUT_DIR)/.sentinel
TF_MODEL_DIR := model
TF_MODEL := $(TF_MODEL_DIR)/20170511-185253/20170511-185253.pb
CLASSIFIER := $(OUTPUT_DIR)/classifier.pkl
DOWNLOADS := $(LFW_SENTINEL) $(LANDMARKS_FILE) $(TF_MODEL)

.PHONY: all train preprocess download clean

all: train

train: $(CLASSIFIER)

$(CLASSIFIER): $(PREPROCESS_SENTINEL)
	$(PYTHON) train_classifier.py \
		--input-dir $(PREPROCESS_OUTPUT_DIR) \
		--model-path $(TF_MODEL) \
		--classifier-path $(CLASSIFIER) \
		--num-threads 16 \
		--num-epochs 25 \
		--min-num-images-per-class 10 \
		--is-train

preprocess: $(PREPROCESS_SENTINEL)

$(PREPROCESS_SENTINEL): $(DOWNLOADS)
	$(PYTHON) preprocess.py \
		--input-dir $(LFW_DIR) \
		--predictor $(LANDMARKS_FILE) \
		--output-dir $(PREPROCESS_OUTPUT_DIR) \
		&& touch $@

download: $(DOWNLOADS)

$(LFW_SENTINEL):
	curl -SL http://vis-www.cs.umass.edu/lfw/$(LFW).tgz | tar -xz -C $(DATA_DIR) && touch $@

$(LANDMARKS_FILE):
	curl -SL http://dlib.net/files/$(LANDMARKS).bz2 | bunzip2 >$(LANDMARKS_FILE)

$(TF_MODEL):
	$(PYTHON) download_and_extract_model.py --model-dir $(TF_MODEL_DIR)

clean:
	rm -rf $(DATA_DIR)/* $(OUTPUT_DIR)/* $(TF_MODEL_DIR)/*
