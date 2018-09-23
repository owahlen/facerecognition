DATADIR := data
OUTPUTDIR := output
LFW := lfw
LANDMARKS := shape_predictor_68_face_landmarks.dat
LFWDIR := $(DATADIR)/$(LFW)
LFWSENTINEL := $(LFWDIR)/.sentinel
LANDMARKSFILE := $(DATADIR)/$(LANDMARKS)

all: $(LFWSENTINEL) $(LANDMARKSFILE)

$(LFWSENTINEL): $(LFWDIR).tgz
	mkdir -p $(DATADIR) && tar -xvz -C $(DATADIR) -f $< && touch $(LFWDIR)/.sentinel

$(LFWDIR).tgz:
	curl -SL --create-dirs -o $@ http://vis-www.cs.umass.edu/lfw/lfw.tgz

$(LANDMARKSFILE): $(LANDMARKSFILE).bz2
	bzip2 -d $<

$(LANDMARKSFILE).bz2:
	curl -SL --create-dirs -o $@ http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

clean:
	rm -rf $(DATADIR)/* $(OUTPUTDIR)/*

