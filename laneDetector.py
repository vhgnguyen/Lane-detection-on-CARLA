#!/usr/bin/python

import numpy as np 
import cv2
import math
from imageFilters import ImageFilter, PerspectiveTransform
from line import Line
import config as cf

class LaneDetector():

    def __init__(self, nHistogram=cf.N_HISTOGRAM):

        self.projMgr = PerspectiveTransform()
        self.M, self.M_inv, self.img_shape = self.projMgr.get()

        self.curImg = None
        self.outImg = None
        self.outMask = np.zeros_like((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8)

        self.y, self.x = self.img_shape
        self.midPoint = self.x // 2

        self.left = 0
        self.right = 1

        self.lineBase = None
        self.leftLine = Line(self.left, self.img_shape)
        self.rightLine = Line(self.right, self.img_shape)
        self.mainLane = False

        self.nHistogram = nHistogram
        self.histogram = None
        self.hx = self.x // self.nHistogram

    def computeHistogram(self, curImg):
        h = np.sum(curImg[self.y//2:, :], axis=0)
        self.histogram = np.zeros_like(h)
        for i in range(self.nHistogram):
            left = np.int(self.hx * i)
            right = np.int(min(self.hx * (i+1) - 1, self.x -1))
            x_max = np.argmax(h[left:right]) + left
            if x_max > self.x:
                return
            self.histogram[x_max] = np.sum(h[left:right])
    
    # function to find left and right lane base position from histogram position
    def findMainLane(self, nx, ny):
        HLeft = np.array([self.midPoint-self.hx-1, self.midPoint-1])
        HRight = np.array([self.midPoint, self.midPoint+self.hx])
        leftBase = np.argmax(self.histogram[HLeft[0]:HLeft[1]]) + HLeft[0]
        rightBase = np.argmax(self.histogram[HRight[0]:HRight[1]]) + HRight[0]
        # find left lane lines
        self.leftLine.setBase(leftBase)
        self.outMask = self.leftLine.findLinesPoints(leftBase, nx, ny, self.outMask)
        self.leftLine.fitPolyPrior()
        # find right lane lines
        self.rightLine.setBase(rightBase)
        self.outMask = self.rightLine.findLinesPoints(rightBase, nx, ny, self.outMask)
        self.rightLine.fitPolyPrior()
    
    def findAdjacentLane(self, nx, ny):
        if not self.mainLane:
            return
        

    # function to find lane
    def findLane(self):
        if self.curImg is None:
            return
        filterImg = ImageFilter.applyFilter(self.curImg)
        lineMask = self.projMgr.warpToBirdEyeView(filterImg)

        self.outMask = np.dstack((lineMask, lineMask, lineMask)) * 255
        outWarped = np.zeros_like(self.curImg)
        outUnwarped = np.zeros_like(self.curImg)

        nonzero = lineMask.nonzero()
        nx = np.array(nonzero[1])
        ny = np.array(nonzero[0])

        if not self.leftLine.isDetected or not self.rightLine.isDetected:
        # if True:
            self.computeHistogram(lineMask)
            self.findMainLane(nx, ny)
            # add main lane
            if self.leftLine.confidence >= 0.5 and self.rightLine.confidence >= 0.5:
                self.mainLane = True
            else:
                self.mainLane = False
        else:
            self.leftLine.findLinesPointsAroundPoly(nx, ny)
            self.leftLine.fitPolySecond()
            self.rightLine.findLinesPointsAroundPoly(nx, ny)
            self.rightLine.fitPolySecond()
            if self.leftLine.isDetected and self.rightLine.isDetected:
                self.mainLane = True
            else:
                self.mainLane = False

        self.findAdjacentLane(nx, ny)

        # Draw the lane onto the warped blank image
        # if self.mainLane:
        leftMask = np.array([np.transpose(np.vstack([self.leftLine.linspace_x, self.leftLine.linspace_y]))])
        rightMask = np.array([np.flipud(np.transpose(np.vstack([self.rightLine.linspace_x, self.rightLine.linspace_y])))])
        mask = np.hstack((leftMask, rightMask))
        cv2.fillPoly(outWarped, np.int_([mask]), (0,255, 0))
        # Unwarp the image
        outUnwarped = self.projMgr.warpToCameraView(outWarped)
        self.outMask = cv2.addWeighted(self.outMask, 1, outWarped, 0.5, 0)
        # self.draw()

        return outUnwarped

    def draw(self):
        # curvature
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(self.outMask, 'Left: RoC: %fm'%(self.leftLine.measureCurvature()), (30,30), font, 1, color=(255,255,0), thickness=1)
        cv2.putText(self.outMask, 'Right: RoC: %fm'%(self.rightLine.measureCurvature()), (30,60), font, 1, color=(255,255,0), thickness=1)

    def run(self, curImg):
        self.curImg = curImg

        # run the pipeline
        outUnwarped = self.findLane()

        self.outImg = cv2.addWeighted(self.curImg, 1, outUnwarped, 0.3, 0)
        self.outImg = np.hstack((self.outImg, self.outMask))
        cv2.imshow("Img:", self.outImg)
        cv2.waitKey(30)





            


    
        









