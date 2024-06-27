import numpy as np


class Translator():

    def __init__(self, model):
        self.model = model

    def translate(self, explanations1, explanations2):
        return 