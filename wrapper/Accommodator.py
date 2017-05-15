import sys
import csv
import scipy.stats as stats
from wrapper import Configs as cfg


class Accommodator:

    def __init__(self, config: cfg) :
        self.cfg = config
        self.fileSuffix = "accomm.csv"
        self.accommSchema = [ "Speaker-task",
                              "Correlation coefficient",
                              "Absolute rho",
                              "P-value",
                              "Max time" ]


    def compare_word_likelihoods( self ):
        for speaker in self.cfg.speakers :
            self.__get_speaker_likelihoods(speaker)


    """
        Calculates  log p(x_A|A) / p(x_A|B)  for each word.
        p(x_A|B) is the probability of an utterance by speaker A being produced by speaker B. 
        p(x_A|A) is the probability of x_A being produced by speaker A

        Outputs one .csv file per speaker, listing, for each rec file: 
            1) file path, 
            2) the time of utterance, 
            3) the log likelihood ratio, 
            4) the word, 
            5) word length in chars, 
            6) frame number.
    """
    def __get_speaker_likelihoods(self, speaker) :
        recListPath = self.cfg.paths['recognitionListPath'] + speaker + "recs.txt"
        outputPath = self.cfg.paths['accommDirectory'] + speaker + "_ratios.csv"
        old_sysout = sys.stdout     # Redirect stdout

        # Read the rec files
        with open(recListPath,'r') as f :
            targets = f.readlines()

        count = self.__compute_likelihoods(targets, outputPath)
        sys.stdout = old_sysout  	#  Return stdout to display
        print( "Completed computing", count, "likelihoods for", speaker )


    def __compute_likelihoods(self, targets, outputPath ) :
        count = 0

        with open(outputPath, 'w') as of:
            sys.stdout = of  # Redirect print to outFile.

            # For each recognition file in the 'target' directory:
            for targetPath in targets:
                self.__calculate_ratio(targetPath)
                count = count + 1

        return count


    def __calculate_ratio(self, targetPath) :
        # Getting the name of the rec files
        targetPath = targetPath[0:len(targetPath) - 1]  # Path of current target file.
        word, time = self.__get_metadata(targetPath)
        targetLikelihood, frameNumber = self.__get_likelihood(targetPath)

        # Create a path to the equivalent 'self' recognition file.
        selfPath = targetPath.replace("target", "self", 1)
        selfLikelihood = self.__get_likelihood(selfPath)
        ratio = selfLikelihood - targetLikelihood       # Subtract the logs for our A/B ratio.

        filepath = targetPath.split('/')
        recName  = filepath[len(filepath) - 1]
        # Output to CSV: [Rec path, log-likelihood ratio, word spoken, word length, frame extracted at].
        print(recName + "," + time + "," + str(ratio) + "," + word + "," + len(word) + "," + str(frameNumber))


    # Extract the time from the file name
    def __get_metadata(self, path: str) -> (str, float) :
        # Produces an Array of: [0] path + task_speaker; [1] word spoken; [2] the time
        itemList = path.split('-')
        # Array of: [0] secs time, [1] millisecs time, and [2] ".rec"
        timeList = itemList[len(itemList) - 1].split('.')
        timeInSecs = float(timeList[0]) + float(timeList[1]) / 1000
        word = itemList[1]

        return word, timeInSecs


    def __get_likelihood(self, path: str)  -> (float, float) :
        startTime = 1
        likelihoodIndex = 3

        with open(path, 'r') as f:
            recognition = f.readline()

        likelihood = float(recognition.split(',')[likelihoodIndex])
        # Find the frame it was spoken at:
        frameNumber = float(recognition.split()[startTime]) * self.cfg.framesPerSecond

        return likelihood, frameNumber


    """
        Derives a correlation from the point likelihood ratios over time.
        
        Input: log-likelihood ratio for each of a speaker's utterances.
        Output: csv summarising the correlation between log-likelihood and time, 
            for each analysis unit (that, for speaker i on task j).

    """
    def detect_accommodation(self) :
        # Redirect standard output stream to a csv:
        old_sysout = sys.stdout
        outPath = self.cfg.paths['correlationPath']
        with open(outPath, 'w') as of :
            sys.stdout = of

            #  Column headers:
            print( ",".join(self.accommSchema))

            #  Find correlations for each task for our twelve models:
            for speaker in self.cfg.speakers:
                self.__get_speaker_accommodation(speaker)

        sys.stdout = old_sysout									    # Return output stream to console.
        print( "Correlations computed, at " + outPath )


    def __get_speaker_accommodation(self, speaker: str) :
        accommodationPath = self.cfg.paths['accommPath'] + speaker + self.fileSuffix

        for task in self.__get_tasks(speaker):
            with open(accommodationPath, 'r') as f :
                reader = csv.reader(f)

                times, distances = self.__extract_task_ratios(reader, task)
                correlation = stats.spearman(times, distances)
                spearCoefficient = correlation[0]
                pValue = correlation[1]
                taskLength = max(times)  # Approximate the task end by the start of the last word.

                row = [speaker + " " + task, spearCoefficient, abs(spearCoefficient), pValue, taskLength ]
                print( ",".join(row) )


    def __get_tasks(self, speaker) :
        taskFile = self.cfg.paths['taskPath'] + speaker + "tasks.txt"
        with open(taskFile, 'r') as f:
            tasks = f.readlines()            # Read in the 11 test tasks.

        # Truncate trailing newlines (e.g. "raB_1\n")
        return [ task[:-1] if (task[-1] == '\n') else task for task in tasks ]


    def __extract_task_ratios(self, reader, currentTask) :
        times = []
        distances = []

        for record in reader:
            taskField = record[0]
            if taskField.find(currentTask) >= 0:            # If current record is of the current task:
                time = record[1]
                correlation = record[2]
                times.append(float(time))                   # Add time to array
                distances.append(0.0 - float(correlation))  # Add ratio to array

        return times, distances