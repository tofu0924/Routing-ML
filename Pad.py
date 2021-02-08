import math 
from collections import namedtuple 
import numpy as np 
_Point = namedtuple("Point", ["x", "y"]) 
import cv2 
import random 
import copy 
from abc import ABC, abstractmethod

class Point(_Point): 
    def __add__(self, other): 
        return Point(self.x + other.x, self.y + other.y) 
    def __sub__(self, other): 
        return Point(self.x - other.x, self.y - other.y) 
    def dist2(self, other): 
        if(type(other) != Point): 
            raise ValueError("Param1 must be Point") 
        dx = self.x-other.x 
        dy = self.y-other.y 
        return math.sqrt(dx**2 + dy**2) 
    def __str__(self): 
        return "x:%.3f, y:%.3f" % (self.x, self.y)

class Line():
    def __init__(self, p1, p2): 
        self._p1 = p1 
        self._p2 = p2
    @property
    def p1(self): 
        return self._p1 
    @p1.setter 
    def p1(self, point): 
        if(type(point) != Point): 
            raise ValueError("p1 must be Point type.") 
        self._p1 = point 
    @property 
    def p2(self): 
        return self._p2 
    @p2.setter 
    def p2(self, point): 
        if(type(point) != Point): 
            raise ValueError("p2 must be Point type.")
        self._p2 = point

    def rotate(self, degree):
        #p2 is pivot, p1 is moving
        p1Translation = self.p1 -self.p2
        radian = degree*(math.pi/180)
        c, s = math.cos(radian), math.sin(radian)
        p1Rotation = Point(c*p1Translation.x - s*p1Translation.y, s*p1Translation.x + c*p1Translation.y)
        if(p1Rotation.x < p1Translation.x ):
            c, s = math.cos(-radian), math.sin(-radian)
            p1Rotation = Point(c*p1Translation.x - s*p1Translation.y, s*p1Translation.x + c*p1Translation.y)
        self.p1 = self.p2
        self.p2 = p1Rotation + self.p1

    def stretch(self, length):
        #p2 will is stretched
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        if (dx == 0):
            self.p2 = Point(self.p2.x, self.p2.y + length)
        elif (dy == 0):
            self.p2 = Point(self.p2.x + length, self.p2.y)
        else:
            diagonal = math.sqrt(dx**2 + dy**2)
            self.p2 = Point(self.p2.x + length * dx / diagonal,
                self.p2.y + length * dy / diagonal)
            
    def getLength(self):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        return math.sqrt(dx**2 + dy**2)
        
    def __str__(self):
        return "(P1x:%.3f, P1y:%.3f) to (P2x:%.3f, P2y:%.3f)" % (self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return Line(copy.deepcopy(self._p1), copy.deepcopy(self._p2))
    
class Render(): 
    def __init__(self, startline, endline, shape): 
        self.img = np.zeros(shape,dtype=np.uint8) 
        cv2.line(self.img, (int(startline.p1.x), int(startline.p1.y)), (int(startline.p2.x), int(startline.p2.y)), color = (255,255,255), thickness = 1) 
        cv2.line(self.img, (int(endline.p1.x), int(endline.p1.y)), (int(endline.p2.x), int(endline.p2.y)), color = (255,255,255), thickness = 1)

    def drawTriangle(self, p1, p2, p3, color =None):
        if(color is None):
            color = (random.randrange(0,256), random.randrange(0,256), random.randrange(0,256))
        triangle_cnt = np.array([(p1.x, p1.y),(p2.x, p2.y),(p3.x, p3.y)])        
        cv2.drawContours(self.img, [triangle_cnt], 0, color, -1)

    def draw(self, startline, endline):
        p1 = Point(int(startline.p1.x), int(startline.p1.y))
        p2 = Point(int(startline.p2.x), int(startline.p2.y))
        p3 = Point(int(endline.p2.x), int(endline.p2.y))
        self.drawTriangle(p1,p2,p3)

    def getImage(self):
        return self.img

    def imgshow(self):
        cv2.imshow('%4s' % random.randint(0,9999), self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

class RewardStrategy(ABC): 
    @abstractmethod 
    def getReward(self, pad) -> float: 
        pass 
    @abstractmethod 
    def getMinReward(self, pad) -> float: 
        pass

class DefaultRewardStrategy(RewardStrategy): 
    def getReward(self, pad) -> float: 
        time_reward = 300. - float(pad.time) 
        diff = pad.getDiff() 
        if(diff == 0.): 
            diff_reward = 10000. 
        else: 
            diff_reward = 1000. - float(diff) 
        lenthDiff = abs(pad.curLine.getLength() - pad.endLine.getLength())
        if(time_reward < 0): 
            return -9999. 
        return time_reward + diff_reward - 100*lenthDiff
        
    def getMinReward(self, pad) -> float: 
        return -9999.

class Pad():
    @property 
    def rewardStrategy(self) -> RewardStrategy: 
        return self._rewardStrategy 
    @rewardStrategy.setter 
    def rewardStrategy(self, rewardStrategy: RewardStrategy): 
        self._rewardStrategy = rewardStrategy

    @property
    def shape(self):
        return self._renderShape 
    @shape.setter
    def shape(self, shape):
        self._renderShape = shape

    def __init__(self, startline, endline, rewardStrategy, useRender=False, renderSize=(100,100,3)):
        self.init_val = {"l1":startline,"l2":endline, "reward":rewardStrategy, "useRender":useRender}
        self._renderShape = renderSize
        self.curLine = copy.deepcopy(startline)
        self.endLine = copy.deepcopy(endline)
        self.time = 0
        self.useRender = useRender
        self._rewardStrategy = rewardStrategy
        if(useRender):
            self.rendObj = Render(startline,endline,(100,100,3))
        else:
            self.rendObj = []

    def move(self, theta, deltaR) -> bool:
        tempLine = copy.deepcopy(self.curLine)

        if (theta != 0.):
            self.curLine.rotate(theta)
        if( deltaR != 0.):
            self.curLine.stretch(deltaR)

        if( self.curLine.p1.x < 0 or self.curLine.p1.x >= self.shape[1] ):
            return False
        elif( self.curLine.p1.y < 0 or self.curLine.p1.y >= self.shape[0] ):
            return False
        elif( self.curLine.p2.x < 0 or self.curLine.p2.x >= self.shape[1] ):
            return False
        elif( self.curLine.p2.y < 0 or self.curLine.p2.y >= self.shape[0] ):
            return False

        if( self.useRender ):
            self.rendObj.draw(tempLine, self.curLine)
        return True

    def getObs(self, diff):
        return np.asarray( 
            [self.curLine.p1.x,self.curLine.p1.y,self.curLine.p2.x,self.curLine.p2.y,
            self.endLine.p1.x,self.endLine.p1.y,self.endLine.p2.x,self.endLine.p2.y, diff], 
            dtype=np.float32)

    def getReward(self) -> float:
        return self._rewardStrategy.getReward(self)

    def getMinReward(self) -> float:
        return self._rewardStrategy.getMinReward(self)

    def step(self, theta, deltaR):
        # if (deltaR <= 0 ):
            # raise ValueError("radius cannot be negative nor zero!")
        self.time = self.time + 1
        isMovingInBoundry = self.move(theta, deltaR)
        diff = self.getDiff()
        obs = self.getObs(diff)
        reward = self.getReward() if isMovingInBoundry else self.getMinReward()
        done = True if (diff == 0. or reward < 0) else False
        return obs, reward, done, {}

    def getDiff(self):
        p1ToP1 = self.curLine.p1.dist2(self.endLine.p1)
        p2ToP2 = self.curLine.p2.dist2(self.endLine.p2)
        diff1 = p1ToP1 + p2ToP2

        p1ToP2 = self.curLine.p1.dist2(self.endLine.p2)
        p2ToP1 = self.curLine.p2.dist2(self.endLine.p1)
        diff2 = p1ToP2 + p2ToP1
        return diff1 if diff1 <= diff2 else diff2

    def reset(self):
        self.curLine = copy.deepcopy(self.init_val["l1"])
        self.endLine = copy.deepcopy(self.init_val["l2"])
        self.time = 0
        diff = self.getDiff()
        obs = self.getObs(diff)
        if (self.init_val["useRender"]):
            self.rendObj = Render(self.init_val["l1"], self.init_val["l2"], self.shape)
        return obs

    def render(self):
        if(self.useRender):
            return self.rendObj
        else:
            raise Exception("No Reder Objects in this Pad Object.")



if __name__ =="__main__": 
    l1 = Line(Point(1.,2.), Point(2.,10.)) 
    l2 = Line(Point(90.,71.), Point(91.,80.))
    myPad=Pad(l1, l2, rewardStrategy = DefaultRewardStrategy(),  useRender=True)
    myPad.step(-35, 0)
    myPad.render().getImage()
    myPad.render().imgshow()
    myPad.reset()
    myPad.move(-35, 0)
    myPad.reset()
    myPad.move(-35, 0)
    myPad.reset()
    print('------------\n')
    print(myPad.step(-55.,0.))
    print('------------\n')
    print(myPad.step(55.,0.))
    print('------------\n')
    print(myPad.step(-55.,0.))
    print('------------\n')
    print(myPad.step(45,0.))
    print('------------\n')
    print(myPad.step(-35,0.))
    print('------------\n')

    print(l1,l2)