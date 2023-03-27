import fasttext.util
import numpy as np
import os, sys

class suppress_output:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr

fasttext.util.download_model('en', if_exists='ignore')

class vectorize:
    def __init__(self, phrase_list, ext_model = True, model = None):
        self.ext_model = ext_model
        self.d2vmodel = model
        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            self.phrases = phrase_list
        
    def array(self):
        if not self.ext_model:
            fasttext.util.download_model('en', if_exists='ignore')
            self.d2vmodel = fasttext.load_model('cc.en.300.bin')         
        vec = []
        for phrase in self.phrases:
            for word, order in zip(phrase.split(' '), range(len(phrase.split(' ')))):
                if not order:
                    vec.append(self.d2vmodel.get_word_vector(word))
                else:
                    vec[-1] = np.vstack((vec[-1], self.d2vmodel.get_word_vector(word)))
                if len(phrase.split(' ')) == order + 1:
                    vec[-1] = np.expand_dims(vec[-1], axis=0)

        plarray = np.concatenate(vec, axis=0)
        
        return plarray