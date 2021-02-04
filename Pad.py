import math
from collections import namedtuple
_Point = namedtuple("Point", ["x","y"])

class Point(_Point):
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    def dist2(self, other):
        if(type(other) != Point):
            raise ValueError("Param1 must be Point")
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx**2 + dy**2)

class Line:
    def __init__(self, p1, p2):
        self.__p1 = p1
        self.__p2 = p2
    @property
    def p1(self):
        return self.__p1
    @p1.setter
    def p1(self, point):
        if(type(point) != Point):
            raise ValueError("p1 must be Point type")
        self.__p1 = point
    @property
    def p2(self):
        return self.__p2
    @p2.setter
    def p2(self, point):
        if(type(point) != Point):
            raise ValueError("p2 must be Point type")
        self.__p2 = point

    def rotate(self,degree):
        #p2 is pivot
        p1Translation = self.p1 - self.p2
        radian = degree*(math.pi/180.)
        c, s = math.cos(radian), math.sin(radian)
        p1Rotation = Point(c*p1Translation.x - s*p1Translation.y, s*p1Translation.x + c*p1Translation.y)
        self.p1 = self.p2
        self.p2 = p1Rotation + self.p1

    def strecth(self, length):
        #p2 will is streched
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        if (dx == 0):
            self.p2 = Point(self.p2.x, self.p2.y + length)
        elif(dy==0):
            self.p2 = Point(self.p2.x + length, self.p2.y)
        else:
            diagonal = math.sqrt(dx**2 + dy**2)
            self.p2 = Point(self.p2.x + length * dx / diagonal, self.p2.y + length * dy / diagonal)

class Pad():
    def __init__(self, startline, endline):
        self.curLine = startline
        self.endLine = endline

    def move(self, theta, deltaR):
        if(theta != 0.):
            self.curLine.rotate(theta)
        if(deltaR != 0.):
            self.curLine.stretch(deltaR)
        print(self.curLine.p1)
        print(self.curLine.p2)

    def step(self, theta, deltaR):
        self.move(theta, deltaR)
        diff = self.getDiff()
        print(diff)
        return self.curLine, self.endLine, diff

    def getDiff(self):
        p1ToP1 = self.curLine.p1.dist2(self.endLine.p1)
        p2ToP2 = self.curLine.p2.dist2(self.endLine.p2)
        diff1 = p1ToP1 + p2ToP2

        p1ToP2 = self.curLine.p1.dist2(self.endLine.p2)
        p2ToP1 = self.curLine.p2.dist2(self.endLine.p1)
        diff2 = p1ToP2 + p2ToP1
        return diff1 if diff1 <= diff2 else diff2

if __name__ == "__main__":
    p1 =Point(1.0, 2.0)
    p2 =Point(2.0, 1.0)
    l1 = Line(p1, p2)

    p1 = Point(11.0, 12.0)
    p2 = Point(12.0, 11.0)
    l2 = Line(p1, p2)

    myPad=Pad(l1, l2)
    myPad.step(-35.,0.)
    myPad.step(35.,0.)
    myPad.step(-45.,-1.)
