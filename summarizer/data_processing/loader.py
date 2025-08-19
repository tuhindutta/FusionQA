import json
import os
from data_processing.utils import Pdf

class Loader(Pdf):
    def __init__(self, path:str):
        super().__init__()
        self.doc = tuple(enumerate(self.read(path).split('\n')))
        self.filename = os.path.basename(path).split('.')[0]

    def start_from_line(self, idx:int=0):
        doc = self.doc
        doc = '\n'.join([i[1] for i in doc][idx:])
        return doc
    

def json_loader(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def json_dumper(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

class PdfFiles:
    pdfs = [f"./documents/{i}" for i in os.listdir("./documents")]
    