#  A Python wrapper for the Hidden Markov Model ToolKit

[HTK](http://htk.eng.cam.ac.uk/) is a venerable open-source modelling tool, which helped generations of linguists make state-of-the-art models of speech.
Once upon a time, anyway; you have no reason not to use `NLTK` or [hmmlearn](https://github.com/hmmlearn/hmmlearn) these days.

If, like me, you're forced to use it by some artificial constraint, you'll find that it is batch-only, requires hundreds of intermediate files for most processes, often takes 10 ordered command arguments, has appalling C99 error messages, crashes if it finds or does not find newlines in specific places, and extremely dense docs. This wrapper makes using it a bit less painful.

The wrapper doesn't really reflect HTK's generality: it builds speaker models from wavs. My usecase took raw speech files from a pair of interlocutors, Labb-Cat annotations for their conversation, built models for each speaker, and then reported their overall 'accommodation' to their interlocutor over time.

---

###  Usage

1. Install HTK and Python3.
2. Get speech data, annotate it.
2. Point `Configs.root` at your files.
3. Run like so :  `python main.py statesPerHmm=3 vectorType=LPC iterations=10`