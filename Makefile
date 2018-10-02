PYTHON := python
export PYTHONPATH = src
DATASETS_DIR := datasets
OUTPUT_DIR := output
LFW_DIR := $(DATASETS_DIR)/lfw
LFW_RAW_DIR := $(LFW_DIR)/raw
LFW_RAW_SENTINEL := $(LFW_RAW_DIR)/.sentinel
LFW_ALIGN_OUTPUT_DIR := $(LFW_DIR)/lfw_mtcnnpy_160
LFW_ALIGN_SENTINEL := $(LFW_ALIGN_OUTPUT_DIR)/.sentinel
TF_MODEL_DIR := model
TF_MODEL_NAME := 20180402-114759
TF_MODEL := $(TF_MODEL_DIR)/$(TF_MODEL_NAME)/$(TF_MODEL_NAME).pb
CLASSIFIER := $(OUTPUT_DIR)/classifier.pkl
DOWNLOADS := $(LFW_RAW_SENTINEL) $(TF_MODEL)

.PHONY: all test train align download clean

all: test

test: $(CLASSIFIER)
	$(PYTHON) train_classifier.py \
		--input-dir $(LFW_ALIGN_OUTPUT_DIR) \
		--model-path $(TF_MODEL) \
		--classifier-path $(CLASSIFIER) \
		--num-threads 16 \
		--num-epochs 25 \
		--min-num-images-per-class 10 \

train: $(CLASSIFIER)

$(CLASSIFIER): $(LFW_ALIGN_SENTINEL)
	$(PYTHON) train_classifier.py \
		--input-dir $(LFW_ALIGN_OUTPUT_DIR) \
		--model-path $(TF_MODEL) \
		--classifier-path $(CLASSIFIER) \
		--num-threads 16 \
		--num-epochs 25 \
		--min-num-images-per-class 10 \
		--is-train

align: $(LFW_ALIGN_SENTINEL)

$(LFW_ALIGN_SENTINEL): $(DOWNLOADS)
	$(PYTHON) src/align/align_dataset_mtcnn.py \
		$(LFW_RAW_DIR) \
		$(LFW_ALIGN_OUTPUT_DIR) \
		--image_size 160 \
		--margin 32 \
		--random_order \
		--gpu_memory_fraction 0.25 && \
		touch $@

download: $(DOWNLOADS)

$(LFW_RAW_SENTINEL):
	mkdir -p $(LFW_RAW_DIR) && \
	curl -SL http://vis-www.cs.umass.edu/lfw/lfw.tgz | tar -xz -C $(LFW_RAW_DIR) --strip-components 1 && \
	touch $@

$(TF_MODEL):
	$(PYTHON) src/download_and_extract_model.py --model-dir $(TF_MODEL_DIR) --model-name $(TF_MODEL_NAME)

clean:
	rm -rf $(DATASETS_DIR)/* $(OUTPUT_DIR)/* $(TF_MODEL_DIR)/*
