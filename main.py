from wrapper import \
    Accommodator as accomm, \
    AudioAnnotation as audioNotes, \
    Configs as cfg, \
    HmmLearner as hmm

"""
	Building models to detect linguistic accommodation.

	This script takes us from wav files segmented by word, 
	to 12 speaker models, then to 132 speaker-pair likelihood-ratio correlations. 

	(The data is an experiment with 12 speakers (6 pairs) 
	co-operating on 11 tasks while screened off from each other: no nonverbal cues.)

	First real Python project; even so, just HTK and SciPy calls.
"""

########################################################
#  Get hyperparameters and build filesystem.
########################################################

args = cfg.Configs.parse_args()
#  How many states will each HMM have?
#  Which acoustic features will you take from the speech signal?
#  How many training iterations?
states, vectorType, iters = cfg.Configs.validate(args)

conf = cfg.Configs( states, vectorType, iters )#  Build the complicated filepaths
conf.construct_directories()            #  Build the complicated filesystem


########################################################
#  Segment speech into words.
########################################################

annotator = audioNotes.AudioAnnotation(conf)
annotator.annotate_raw_speech()		#  Load conversation wavs and split into individual words:


########################################################
#  Vectorise and label data, then train HMMs
########################################################

learner = hmm.HmmLearner( conf )
learner.extract_acoustic_features()
learner.create_labels_from_vectors()
conf.create_map_file()              #  Create a "SCP" (script input) file: a wavFile -> vectorFile map

learner.initialise_all()			#  set means and variances to a useful prior.
learner.multiply_gaussians()        #  set the emission function of each state to a GMM
learner.train()
learner.create_lattices()
learner.evaluate_speaker_models()


##########################################################
#  Use HMMs to measure accommodation: 
#  i.e. correlation to interlocutor likelihood
##########################################################

detector = accomm.Accommodator(conf)
detector.compare_word_likelihoods() #  likelihood ratios for all words, given paired speaker models
detector.detect_accommodation()     #  calculates ratio correlations.

