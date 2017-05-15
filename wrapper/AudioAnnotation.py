import csv
import wave

from wrapper import Configs


#  Splits speech files into words given metadata, e.g. LaBB-Cat annotations.
class AudioAnnotation:

    def __init__(self, config: Configs):
        self.noteFmt = ".csv"           # Format of annotation file
        self.audioFmt = ".wav"          # Format of input speech file
        self.audioOutFmt = self.audioFmt # Format of output speech file
        self.timePrecision = 3          # Decimal places that audio can be cut to
        self.cfg = config
        self.schema = self.cfg.annotationSchema


    #  Entry point: breaks all .wav files into words, annotates filename, writes out.
    def annotate_raw_speech(self)  -> None:
        for speaker in self.cfg.speakers:
            self.__chunk_speaker_audio(speaker)


    """
        Cuts conversation .wavs into individual word .wavs 
        using LaBB-CAT annotation of timings and meaning.

        Inputs:
            1) Speaker whose files are processed
            2) Labb-cat data as csv.
            3) Raw .wav of the conversation. (Contains one speaker's half of one task.)

        Outputs:
            1) an individual .wav audio file for every word.
    """
    def __chunk_speaker_audio(self, speaker: str)  -> None :

        for task in self.cfg.tasks:
            speakerTask = task + speaker
            labbCatData = self.__get_labbCat(speakerTask)

            # Open .wav and construct metadata dict:
            rawAudio = self.__get_audio(speakerTask)
            acoustics = self.__get_wav_stats(rawAudio)

            # Loop over the wav file word by word, producing a series of annotated wav files:
            self.__process_annotations(speaker, labbCatData, acoustics)
            rawAudio.close()


    def __get_labbCat(self, speakerTask: str)  -> list :
        taskLabbCatPath = self.cfg.paths['labbCatDir'] + speakerTask + self.noteFmt

        with open(taskLabbCatPath, 'r') as f:
            data = csv.reader(f)
        data.next()  # Skip header

        return data


    def __get_audio(self, speakerTask: str)  -> wave.Wave_read :
        audioPath = self.cfg.paths['audioDir'] + speakerTask + self.audioFmt
        return wave.open(audioPath, 'r')


    def __get_wav_stats(self, audio: wave.Wave_read) :
        return {
            "waveform": audio,
            "frameRate": audio.getframerate(),
            "nChannels": audio.getnchannels(),
            "sampWidth": audio.getsampwidth()
        }


    def __process_annotations(self, speaker: str, labbCatData: list, acoustics: dict) -> None:
        for record in labbCatData:
            s = self.schema
            transcription = record[s['transcription']].split('.')[0]
            speakerOfWord = record[s['speaker']]
            task = self.__clean_task(record[s['task']])
            wordStart = record[s['startTime']]
            wordEnd = record[s['endTime']]

            if wordStart != "" and wordEnd != "" and speakerOfWord == speaker:
                acoustics['waveform'] = self.__extract_word(acoustics, float(wordStart), float(wordEnd))
                wordStart = self.__regularise_start_time(wordStart)
                outPath = self.__get_word_outPath(transcription, task, wordStart)
                self.__write_word_wav(outPath, acoustics)


    def __clean_task(self, task: str)  -> str :
        return task.replace("'", "").replace("\\", "", 3)


    #  Returns the audio corresponding to one word
    def __extract_word(self, acoustics: list, start: float, end: float) -> bytes :
        audio = acoustics['waveform']
        frameRate = acoustics['frameRate']

        firstFrame = int(start * frameRate)
        audio.setpos(firstFrame)
        framesPerWord = int((end - start) * frameRate)

        return audio.readframes(framesPerWord)


    # Ensure timePrecision decimal digits in the start time.
    def __regularise_start_time(self, time: str)  -> str:
        if '.' in time :
            endAt = self.timePrecision + 1
            limit = time.index('.') + endAt
            if len(time) >= limit:
                time = time[0:limit]
        else :
            zeroes = ("0" * self.timePrecision)
            time = time + "." + zeroes

        return time


    def __get_word_outPath (self, transcription, task, start)  -> str :
        return self.cfg.paths['wavWordsDir'] + transcription + '-' + task + '-' + start + self.audioOutFmt


    def __write_word_wav (self, path, acoustics)  -> None :
        with wave.open(path, 'w') as f:
            f.setnchannels(acoustics['nChannels'])
            f.setsampwidth(acoustics['sampWidth'])
            f.setframerate(acoustics['frameRate'])
            f.writeframes(acoustics['waveform'])