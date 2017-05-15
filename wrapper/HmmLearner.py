"""  
    A Python wrapper for the Cambridge Hidden Markov Model ToolKit (HTK).
    (In particular, for using HTK for speaker recognisers from dyadic-
    conversation speech data.)
"""

import os
from wrapper import Configs as cfg


class HmmLearner :

    def _init__ ( self, config: cfg ) :
        self.cfg = config
        self.temp = "tmp.dat"
        self.training = "trainingset.txt"
        self.testFile = "testset.txt"

        self.htkCommands = {
            "featurise" : "HCopy",
            "label" : "HList",
            "initialise" : "HCompV",
            "edit" : "HHEd",
            "train": "HERest",
            "toSyntax" : "HParse",
            "runViterbi" : "HVite"
        }


    def call_htk(self, action: str, args: str) :
        cmd = self.htkCommands[action] + " "
        os.system(cmd + args)


    def extract_acoustic_features (self) :
        featureArgs = " ".join([
            "-l " + self.cfg.paths['vectorFilesDir'],   #  output directory.
            "-C " + self.cfg.paths['configurationPath'],
            "-S " + self.cfg.paths['scpPath'],          #  list of source & target files.
            "-T " + self.cfg.verbosityLevel             #  trace to a line by line report.
        ])

        self.call_htk('featurise', featureArgs)


    def create_labels_from_vectors(self) :
        # Process the vector files and writing the label files
        count = 0

        for vectorPath in self.__load_vectors() :
            vectorPath = vectorPath[:-1]
            outData = self.__read_vector(vectorPath)

            # Creating the label file: split on dots and then rebuild path after.
            label = self.__create_label_name(vectorPath)
            speaker = self.__get_speaker_from_path(label)

            self.__write_label(label, int(outData[2]), speaker)
            count = count + 1

        print("Program completed, written " + str(count) + " labels to " + self.cfg.paths['labelsDir'])


    def __write_label(self, path: str, timeInMs: int, speaker: str) :
        with open(path, 'w') as of:
            timeInNs = (timeInMs - 1) * 100000
            of.writelines(str(0) + " " + str(timeInNs) + " " + speaker)


    def __get_speaker_from_path(self, label: str) :
        speaker = label.split('_')[2]
        return speaker.split('-')[0]


    def __load_vectors(self) :
        vectorsPath = self.cfg.paths['vectorFileList']
        with open(vectorsPath, 'r') as f:
            return f.readlines()


    # HList is a vector reader, to extract label info.
    def __read_vector(self, vec):
        self.call_htk('featurise', "-h " + vec + " > " + self.temp )

        with open(self.temp, 'r') as f :
            for i in range(3) :
                next(f)  #  Skip headers
            outData = f.readline()

        return outData.split()


    def __create_label_name (self, vecPath: str):
        label = vecPath[:-3] + ".lab"
        return label.replace(self.cfg.paths['vectorFilesDir'], self.cfg.paths['labelsDir'], 1)


    def initialise_all( self ) :
        for speaker in self.cfg.speakers :
            self.initialise_hmm( speaker )


    """
      `initialise_hmm` takes one model prototype and yields a 'flat-start' model.

      A 'flat' initialisation (each state is set to the same initial value), pre-training.
    """
    def initialise_hmm( self, speaker ) :
        baseModels = self.cfg.paths['rootModelsPath']
        trainingPath = self.cfg.paths['trainingDataDir'] + speaker + "trainingset.txt"

        initArgs = " ".join([
            "-S " + trainingPath,				# Training files
            "-M " + self.cfg.initializedModelsPath,	# Output directory
            "-L " + self.cfg.labelFilesPath, 	# Label files directory
            "-m" ,								# Flag: compute means AND covariances
            "-l " + speaker, 					# The segment label the model will account for
            "-v " + self.cfg.varianceFloor, 	# Set the variance floor
            "-o " + speaker,					# Set the name of the output model
            "-T " + self.cfg.verbosityLevel,
            baseModels + speaker 		        # Name of model structure file
        ])

        self.call_htk('initialise', initArgs)
        print ( "Completed initialising " + speaker )


    """ 
        Adds a number of Gaussians to each state of the initialized HMM.
        Forms the emission functions.
    """
    def multiply_gaussians(self) :
        modelCommands = self.__get_mixture_commands

        mixtureArgs = " ".join([
            " -M " + self.cfg.paths['mixtureModelsPath'],   #  Output dir
            " -T " + self.cfg.verbosityLevel,               #  Set verbosity
            modelCommands                                   #  One arg per initialised model
        ])
        self.call_htk('edit', mixtureArgs)
        print("Completed mixing gaussians.")


    def __get_mixture_commands(self) :
        with open( self.cfg.paths['interlocutors'], 'r' ) as f :
            models = f.readlines()

        models = self.cfg.ensure_newlines(models)
        inits = self.cfg.paths['initializedModelsPath']
        trimmed = [ model[:-1] for model in models ]
        return " ".join( [ "-H " + inits + model + " " for model in trimmed ] )


    """
       Trains a 'recogniser', that is, a speaker model.
    """
    def train(self) :
        for speaker in self.cfg.speakers :
            speakerTrainingPath = self.cfg.paths['trainingDataDir'] + speaker + self.training

            #  Train model until parameters converge
            #  (embedded Baum-Welch re-estimation)
            for k in range(self.cfg.iterations) :
                self.__baum_welch_reestimation(str(k), speakerTrainingPath)


    def __baum_welch_reestimation(self, iter, trainingPath) :

        trainArgs = " ".join([
            "-S " + trainingPath,
            "-M " + self.cfg.paths['trainedModels'] + "iter" + iter,  # Output directory
            "-L " + self.cfg.paths['labelsDir'],
            "-d " + self.cfg.paths['mixtureModels'],         # Sets mixture models as the HMM emission
            "-v " + self.cfg.varianceFloor,
            "-T " + self.cfg.verbosityLevel,
            " " + self.cfg.paths['interlocutors']           # List of models
        ])

        self.call_htk('train', trainArgs)


    #  Get lattice
    def create_lattices(self) :

        for speaker in self.cfg.speakers :
            grammarPath = self.cfg.paths['grammarPath'] + speaker + "Grammar.txt"
            syntaxPath = self.cfg.paths['syntaxPath'] + speaker + "Syntax.txt"

            # Parses the Backus-Naur formula ('grammar') into a word network ('syntax')
            grammarArgs = " ".join([
                grammarPath,
                syntaxPath,
                " -T " + self.cfg.verbosityLevel
            ])
            self.call_htk('toSyntax', grammarArgs)
            print("Made lattices.")


    """
       Tests HMMs: 'recognise' utterances using the speaker ('self') or interlocutor ('target').
       Yields recognition files (conditional probability of word given model):
       transcriptions :  [start] [end] [speaker[state]] [log likelihood] [pronunciation]
    """
    def evaluate_speaker_models(self) :
        recognitionsPath = self.cfg.paths['recognitionsPath']

        #  HVite recognition for each word of our twelve speakers,
        #  tested against their model and that of their interlocutor:
        for speaker in self.cfg.speakers :
            self.__eval_speaker(speaker, recognitionsPath)
            print ( "Completed recognitions for " + speaker )


    def __eval_speaker(self, speaker, recPath) :
        pairPath = self.cfg.paths['interlocutors'] + speaker + ".txt"
        interlocutor = self.__get_interlocutor(pairPath)

        testPath = self.cfg.paths['testDir']
        selfData = testPath + speaker + self.testFile       #  Own test set.
        testData = testPath + interlocutor + self.testFile  # Then use test data from paired speaker.

        # Evaluates the model and outputs log probabilities, log p(word | model).
        print("Testing " + speaker + " against " + interlocutor + " data")
        testArgs = self.__get_test_dict(speaker, pairPath, testData, recPath + "_self")
        self.call_htk('runViterbi', testArgs)


        print("Testing " + speaker + " against own data")
        testArgs = self.__get_test_dict(speaker, pairPath, selfData, recPath + "_target")
        self.call_htk('runViterbi', testArgs)


    def __get_interlocutor(self, pairPath) :
        with open(pairPath, 'r') as f:
            pairOfSpeakers = f.readlines()

        return pairOfSpeakers[1]


    def __get_test_dict(self, speaker, pairPath, data, output)  -> str :
        syntaxPath = self.cfg.paths['syntaxPath'] + speaker + "Syntax.txt"
        dictionaryPath = self.cfg.paths['dictionaryPath'] + speaker + "Dictionary.txt"

        return " ".join([
            "-w " + syntaxPath,         #  Setting the syntax
            "-S " + data,               #  List of test files to use
            "-l " + output,             #  Set the output directory of recognitions
            "-H " + self.cfg.paths['finalModels'],  #  Trained model to test against
            "-f ",                      #  Track the full state alignment
            " " + dictionaryPath,       #  The speaker model and a monophone representing it.
            " " + pairPath,             #  File containing codenames of the pair of speakers
        ])
