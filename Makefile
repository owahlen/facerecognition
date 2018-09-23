PYTHON := python
DATADIR := data
OUTPUTDIR := output
LFW := lfw
LANDMARKS := shape_predictor_68_face_landmarks.dat
LFWDIR := $(DATADIR)/$(LFW)
LFWSENTINEL := $(LFWDIR)/.sentinel
LANDMARKSFILE := $(DATADIR)/$(LANDMARKS)
TFMODELDIR := model
TFMODEL := $(TFMODELDIR)/20170511-185253/20170511-185253.pb

all: preprocess

train: preprocess
	$(PYTHON) train_classifier.py \
		--input-dir output/intermediate \
		--model-path model/20170511-185253/20170511-185253.pb \
		--classifier-path output/classifier.pkl \
		--num-threads 16 \
		--num-epochs 25 \
		--min-num-images-per-class 10 \
		--is-train

preprocess: download
	$(PYTHON) preprocess.py

download: $(LFWSENTINEL) $(LANDMARKSFILE) $(TFMODEL)

$(LFWSENTINEL):
	curl -SL http://vis-www.cs.umass.edu/lfw/$(LFW).tgz | tar -xz -C $(DATADIR) && touch $(LFWDIR)/.sentinel

$(LANDMARKSFILE):
	curl -SL http://dlib.net/files/$(LANDMARKS).bz2 | bunzip2 >$(LANDMARKSFILE)

$(TFMODEL):
	$(PYTHON) download_and_extract_model.py

clean:
	rm -rf $(DATADIR)/* $(OUTPUTDIR)/* $(TFMODELDIR)/*
