PYTHON := python3
PYTHONPATH := src
DATASETS_DIR := datasets
OUTPUT_DIR := output
LFW_DIR := $(DATASETS_DIR)/lfw
LFW_RAW_DIR := $(LFW_DIR)/raw
LFW_RAW_SENTINEL := $(LFW_RAW_DIR)/.sentinel
LFW_ALIGN_OUTPUT_DIR := $(LFW_DIR)/lfw_mtcnnalign_160
LFW_ALIGN_SENTINEL := $(LFW_ALIGN_OUTPUT_DIR)/.sentinel
CASIA_DIR := $(DATASETS_DIR)/casia
CASIA_NAME := CASIA-WebFace
CASIA_RAW_DIR := $(CASIA_DIR)/$(CASIA_NAME)
CASIA_RAW_SENTINEL := $(CASIA_RAW_DIR)/.sentinel
CASIA_ALIGN_OUTPUT_DIR := $(CASIA_DIR)/casia_mtcnnpy_182
CASIA_ALIGN_SENTINEL := $(CASIA_ALIGN_OUTPUT_DIR)/.sentinel
TF_MODEL_DIR := model
TF_MODEL_NAME := 20180402-114759
TF_MODEL := $(TF_MODEL_DIR)/$(TF_MODEL_NAME)/$(TF_MODEL_NAME).pb
LOG_DIR := logs
DOWNLOADS := $(CASIA_RAW_SENTINEL) $(LFW_RAW_SENTINEL)

export PYTHONPATH

.PHONY: all test train align download clean

all: test

test:
	$(PYTHON) src/validate_on_lfw.py \
		$(LFW_ALIGN_OUTPUT_DIR) \
		$(TF_MODEL_DIR)/$(TF_MODEL_NAME) \
		--distance_metric 1 \
		--use_flipped_images \
		--subtract_mean \
		--use_fixed_image_standardization

train: $(LFW_ALIGN_SENTINEL) $(CASIA_ALIGN_SENTINEL)
	$(PYTHON) src/train_softmax.py \
		--logs_base_dir $(LOG_DIR) \
		--models_base_dir $(TF_MODEL_DIR) \
		--data_dir $(CASIA_ALIGN_OUTPUT_DIR) \
		--image_size 160 \
		--model_def models.inception_resnet_v1 \
		--lfw_dir $(LFW_ALIGN_OUTPUT_DIR) \
		--optimizer ADAM \
		--learning_rate -1 \
		--max_nrof_epochs 150 \
		--keep_probability 0.8 \
		--random_crop \
		--random_flip \
		--use_fixed_image_standardization \
		--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
		--weight_decay 5e-4 \
		--embedding_size 512 \
		--lfw_distance_metric 1 \
		--lfw_use_flipped_images \
		--lfw_subtract_mean \
		--validation_set_split_ratio 0.05 \
		--validate_every_n_epochs 5 \
		--prelogits_norm_loss_factor 5e-4

align: $(LFW_ALIGN_SENTINEL) $(CASIA_ALIGN_SENTINEL)

$(LFW_ALIGN_SENTINEL): $(LFW_RAW_SENTINEL)
	$(PYTHON) src/align/align_dataset_mtcnn.py \
		$(LFW_RAW_DIR) \
		$(LFW_ALIGN_OUTPUT_DIR) \
		--image_size 182 \
		--margin 44 \
		--random_order \
		--gpu_memory_fraction 0.25 && \
		touch $@

$(CASIA_ALIGN_SENTINEL): $(CASIA_RAW_SENTINEL)
	$(PYTHON) src/align/align_dataset_mtcnn.py \
		$(CASIA_RAW_DIR) \
		$(CASIA_ALIGN_OUTPUT_DIR) \
		--image_size 182 \
		--margin 44 \
		--random_order \
		--gpu_memory_fraction 0.25 && \
		touch $@

download: $(DOWNLOADS)

$(LFW_RAW_SENTINEL):
	mkdir -p $(LFW_RAW_DIR) && \
	curl -SL http://vis-www.cs.umass.edu/lfw/lfw.tgz | tar -xz -C $(LFW_RAW_DIR) --strip-components 1 && \
	touch $@

$(CASIA_RAW_SENTINEL):
	mkdir -p $(CASIA_DIR) && \
	$(PYTHON) src/download_and_extract_model.py --data-dir $(CASIA_DIR) --casia-name $(CASIA_NAME) && \
	touch $@


$(TF_MODEL):
	$(PYTHON) src/download_and_extract_model.py --model-dir $(TF_MODEL_DIR) --model-name $(TF_MODEL_NAME)

clean:
	rm -rf $(DATASETS_DIR)/* $(OUTPUT_DIR)/* $(TF_MODEL_DIR)/* $(LOGS)/*
