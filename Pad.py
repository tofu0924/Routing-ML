from abc import *
class move(metaclass=ABCMeta):
    @abstractproperty
    pad = 'pad'
    @abstractmethod
    def move(self):
        pass

class moveInAnArc(move):
    def __init__(self, pad):
        self.pad = pad
    def move(self, theta):
        curline = self.pad.getCurLine()
        p1 = curline.p1
        p2 = curline.p2
        




class Pad():
    def __init__(self, startline, endline):
        self.curLine = startline
        self.endLine = endline2

    def getCurLine(self):
        return self.curLine
