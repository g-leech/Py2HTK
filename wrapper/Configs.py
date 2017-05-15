import os
import subprocess
import sys

class Configs:

    def __init__(self, states, vectorType, iters):
        self.states = states
        self.vectorType = vectorType
        self.iterations = iters

        self.statePath = states + "-State/"

        #  Hyperparameters we're not very interested in
        self.varianceFloor = "0.000001"
        self.verbosityLevel = "00001"
        self.framesPerSecond = 0.00001

        #  Types of acoustic vector to accept:
        self.acousticVectors = ["MFCC", "LPC", "PLP", "VQ"]

        #  HTK requires a plaintext config file too :
        self.configFile = "configparameters.cfg"

        #  List of codenames for our twelve speakers.
        self.speakers = ["ARA14", "GJN14", "HLH30", "JSE11",
                         "JTN20", "JYN22", "KBN30", "SCA01",
                         "SHA13", "SKN03", "TMY30", "ZSE07"]

        #  List of codes representing the twelve Diapix tasks each pair of speakers completed together.
        self.tasks = ["raB_1_", "raB_2_", "raB_3_", "raB_4_",
                      "raF_1_", "raF_2_", "raF_3_", "raF_4_",
                      "raS_1_", "raS_2_", "raS_3_", "raS_4_"]

        #  Which of the tasks to use for training the models
        self.trainingTask = "raB_1"

        self.annotationSchema = {
            "transcription": 2,
            "speaker": 3,
            "task": 11,
            "startTime": 12,
            "endTime": 13
        }

        # Shared paths
        self.root = "/home/gleech/Documents/Output/Code/HTK_for_Python/safe/" #"~/Thesis/Files/"
        self.paths = self.__build_paths(vectorType)
        self.wordList = "wavFiles.txt"
        self.scpPath = "scpFile.txt"


    #  Dict of shared absolute paths
    def __build_paths( self, vec ) :
        paths = {}
        paths['configDir'] = self.__get_path("Lists/Config/")  # feature extraction settings files.
        paths['configurationPath'] = paths['configDir'] + vec + self.configFile
        paths['interlocutors'] = self.__get_path("Lists/SpeakerPairs/")
        # File containing chosen speaker and their paired speaker.

        ##  Paths to input data
        paths['audioDir'] = self.__get_path("0.wavReal/")  # source waveforms.
        paths['labbCatDir'] = self.__get_path("1.LaBB-CAT/")  # conversation annotations
        paths['wavWordsDir'] = self.__get_path("2.WavWords/")  # single word waveforms tagged by time
        paths['vectorFilesDir'] = self.__get_path("3.Vectors/" + vec + "s/")  # source vector files.
        paths['labelsDir'] = self.__get_path("4.Labels/speakerLabels/")  # label files.
        paths['trainingDataDir'] = self.__get_path("Lists/TrainingDataLists/" + vec + "/")
        paths['vectorListDir'] = self.__get_path("Lists/VectorLists/")
        # Path of full list of source vector files.
        paths['vectorFilelist'] = paths['vectorListDir'] + "All_" + vec + "s.txt"
        paths['scpPath'] = self.__get_path("Lists/VectorLists/" + vec + "scpList.txt")
        paths['testDir'] = self.__get_path("Lists/TestDataLists/" + vec + "/")
        paths['taskDir'] = paths['testDir'] + "/TestTasks/"

        ## Paths to the Backus-Naur specs
        paths['syntaxPath'] = self.__get_path("Lists/Syntax/")
        paths['grammarPath'] = self.__get_path("Lists/Grammars/")
        paths['dictionaryPath'] = self.__get_path("Lists/Dictionaries/" + self.statePath)

        ##  Paths to the models at various stages
        currentModel = self.statePath + vec + "/"
        modelDir = self.__get_path("5.Models/")

        paths['rootModels'] = modelDir + "baseModels/" + currentModel
        paths['initializedModels'] = modelDir + "initializedModels/" + currentModel
        paths['mixtureModels'] = modelDir + "mixtureModels/" + currentModel
        paths['trainedModels'] = modelDir + "trainedModels/" + currentModel
        # models after iterations and mixing:
        paths['finalModels'] = paths['trainedModels'] + "/iter10/newMacros"


        ##  Paths to model outputs and their correlations
        paths['recognitionListPath'] = self.__get_path("Lists/RecognitionLists/" + currentModel)
        paths['recognitionsPath'] = self.__get_path("6.Recognitions/" + currentModel)  # Output directory.
        paths['accommPath'] = self.__get_path("7.Outputs/Accommodations/" + currentModel)
        paths['correlationsPath'] = self.__get_path ("Correlations/" + currentModel + "rawOutputTest.csv")

        return paths


    def __get_path(self, addend) :
        if self.root[-1] != '/' :
            addend = '/' + addend
        return self.root + addend


    def construct_directories ( self ) :
        for name in self.paths:
            self.__check_and_mkdir(self.paths[name])


    def __check_and_mkdir(self, path) :
        d = os.path.dirname(path)
        if not os.path.exists(d):
            os.makedirs(d)


    @classmethod
    def parse_args(self)  -> dict:
        cleanArgs = {}
        for arg in sys.argv:
            if '=' in arg:
                param = arg.split('=')
                cleanArgs[param[0]] = param[1]
        return cleanArgs


    @classmethod
    def validate(self, args) :
        knownVectors = ["MFCC", "LPC", "PLP", "VQ"]
        states = args['statesPerHmm']
        vec = args['vectorType'].upper().replace(" ", "")

        if 'iterations' in args:
            iterations = args['iterations']

        if not states :
            sys.exit("Please set a states hyperparameter | int > 1.")

        if not vec or vec not in knownVectors :
            sys.exit("vectorType not recognised. Try one of " + ", ".join(knownVectors))

        if not isinstance(iterations, int):
            print("Couldn't find iteration setting. Using 10")
            iterations = 10

        return states, vec, iterations


    def create_map_file(self)  -> None:
        listWords = self.paths['wavWordsDir']
        subprocess.run(["ls", listWords, ">", self.wordList], stdout=subprocess.PIPE)

        words = self.paths['wordFilelist']
        vectors = self.paths['vectorFilelist']
        subprocess.run([ "cat", words, vectors, ">", self.scpPath ])


    # Note: all lines in the model file must have a newline appended, except the last (!)
    def ensure_newlines(self, filepath)  -> list :
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for i, l in enumerate(lines) :
            if l[:-1] != '\n' :
                lines[i] = l + '\n'

        lines[-1] = lines[-1].replace('\n', '')

        return lines


