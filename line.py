#!/usr/bin/python

import numpy as np 
import math
import cv2
import config as cf

class Line():

    def __init__(self, side, img_shape,
        polyDelta=cf.POLYFIT_MARGIN,
        nWindow=cf.N_WINDOWS, windowMargin=cf.WINDOW_MARGIN, reThresh=cf.RECENTER_WINDOW_THRESH):
        # side identifier
        self.side = side

        # shape of image
        self.img_shape, self.x, self.y = img_shape, img_shape[1], img_shape[0]

        # sliding window
        self.nWindow = nWindow
        self.w_height = self.y // self.nWindow
        self.w_width = cf.WINDOW_MARGIN
        self.reBaseThresh = reThresh

        # detection status in last iteration
        self.isDetected = False
        
        # confidence of line
        self.maxConfidence = 0.0
        self.confidence = 0.0

        # current pixel base postion from histogram
        self.pixelBase = None

        # distance of vehicle from center line
        self.lineBase = None
        
        # curvature
        self.curvature = 0.0

        # detected pixel indices
        self.x_inds = None
        self.y_inds = None
        
        # mask width for polyfit
        self.polyDelta = polyDelta

        # difference between last and new fit coefficients
        self.diffFit = np.array([0,0,0], dtype='float')
        # poly coefficients of most recent fit
        self.currentFit = None
        # poly coefficients of best fit
        self.bestFit = None
        # radius of curvature
        self.roc = None

        # mask for lane
        self.lanemask = np.zeros(self.img_shape, dtype=np.uint8)
        # drawing x y
        self.linspace_y = np.linspace(0, self.y-1, self.y)
        self.linspace_x = np.zeros_like(self.linspace_y)

        # temporary variables
        # x of current fitted line
        self.currentX = None
        # poly line for drawing
        self.linePoly = None
        
    def setBase(self, base):
        self.pixelBase = base
    
    def findLinesPoints(self, base, nonzero_x, nonzero_y, outMask, nWindow=cf.N_WINDOWS):
        lane_inds = []
        cur_base = base

        for window in range(nWindow):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.y - (window+1)*self.w_height
            win_y_high = self.y - window*self.w_height
            win_x_low = cur_base - self.w_width
            win_x_high = cur_base + self.w_width

            good_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)).nonzero()[0]

            lane_inds.append(good_inds)

            cv2.rectangle(outMask, (win_x_low, win_y_low), (win_x_high, win_y_high), (0,255,0), 2)

            # If found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.reBaseThresh:
                cur_base = np.int(np.mean(nonzero_x[good_inds]))
                
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
            
        xval = nonzero_x[lane_inds]
        yval = nonzero_y[lane_inds]
        self.x_inds = np.array(xval)
        self.y_inds = np.array(yval)
        
        return outMask

    def findLinesPointsAroundPoly(self, nonzero_x, nonzero_y, margin=cf.POLY_SEARCH_MARGIN):
        # margin = nonzero_y / self.y * margin
        lane_inds = ((nonzero_x > (self.bestFit[0]*(nonzero_y**2) + self.bestFit[1]*nonzero_y + 
                    self.bestFit[2] - margin)) & (nonzero_x < (self.bestFit[0]*(nonzero_y**2) + 
                    self.bestFit[1]*nonzero_y + self.bestFit[2] + margin)))
        xval = nonzero_x[lane_inds]
        yval = nonzero_y[lane_inds]
        self.x_inds = np.array(xval)
        self.y_inds = np.array(yval)

    # use existing lane on the right to create adjacent lane lines
    def polyFitLeft(self, curImg, rightLane):
        diff = np.polysub(rightLane.lines[rightLane.left].currentFit,
                          rightLane.lines[rightLane.right].currentFit)
        self.currentFit = np.polyadd(rightlane.lines[rightLane.left].currentFit, diff)
        poly = np.poly1d(self.currentFit)
        self.y_inds = rightLane.lines[rightLane.left].y_inds
        self.currentX = poly(self.y_inds)
        self.x_inds = self.currentX

        if len(self.y_inds) > cf.LINE_CREATE_THRESH:
            self.maxConfidence = len(self.y_inds) * 2
            self.confidence = 0.5
            self.isDetected = True
            # create mask
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.y_inds)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.y_inds)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)
            self.lanemask = np.zeros_like(self.lanemask)
            cv2.fillConvexPoly(self.lanemask, self.linePoly, 64)
            # add bottom point
            self.y_inds = np.append(self.y_inds, cf.IMG_HEIGHT - 1)
            self.x_inds = poly(self.y_inds)
            self.XYPolyLine = np.column_stack((self.x_inds, self.y_inds)).astype(np.int32)

            self.bestFit = self.currentFit
            x = poly([cf.IMG_HEIGHT - 1])
            self.pixelBase = x[0]

    # use existing lane on the left to create adjacent lane lines
    def polyFitRight(self, curImg, leftLane):
        diff = np.polysub(leftLane.lines[leftLane.left].currentFit,
                          leftLane.lines[leftLane.right].currentFit)
        self.currentFit = np.polyadd(leftLane.lines[right].currentFit, diff)
        poly = np.poly1d(self.currentFit)
        self.y_inds = leftLane.lines[leftLane.left].y_inds
        self.currentX = poly(self.y_inds)
        self.x_inds = self.currentX

        if len(self.y_inds) > cf.LINE_CREATE_THRESH:
            self.maxConfidence = len(self.y_inds) * 2
            self.confidence = 0.5
            self.isDetected = True
            # create mask
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.y_inds)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.y_inds)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)
            self.lanemask = np.zeros_like(self.lanemask)
            cv2.fillConvexPoly(self.lanemask, self.linePoly, 64)
            # add bottom point
            self.y_inds = np.append(self.y_inds, cf.IMG_HEIGHT - 1)
            self.x_inds = poly(self.y_inds)
            self.XYPolyLine = np.column_stack((self.x_inds, self.y_inds)).astype(np.int32)

            self.bestFit = self.currentFit
            x = poly([cf.IMG_HEIGHT - 1])
            self.pixelBase = x[0]

    # use existing lane one the right to update adjacent lane lines
    def updatePolyFitLeft(self, curImg, rightLane):
        diff = np.polysub(rightLane.lines[rightLane.left].currentFit,
                          rightLane.lines[rightLane.right].currentFit)
        self.currentFit = np.polyadd(
            rightLane.lines[rightLane.left].currentFit, diff)
        poly = np.poly1d(self.currentFit)
        self.y_inds = rightLane.lines[rightLane.left].y_inds
        self.currentX = poly(self.y_inds)
        self.x_inds = self.currentX

        if len(self.y_inds) > cf.LINE_UPDATE_THRESH:
            self.confidence = len(self.y_inds) / self.maxConfidence
            if self.confidence >= 0.5:
                self.isDetected = True
                if self.confidence > 1:
                    self.confidence = 1  
            else:
                self.isDetected = False
            # create line poly

            # create mask
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.y_inds)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.y_inds)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)
            self.lanemask = np.zeros_like(self.lanemask)
            cv2.fillConvexPoly(self.lanemask, self.linePoly, 64)
            # add bottom point
            y_inds = np.append(self.y_inds, cf.IMG_HEIGHT - 1)
            x_inds = poly(y_inds)
            self.XYPolyLine = np.column_stack((x_inds, y_inds)).astype(np.int32)

            self.bestFit = self.currentFit
    
    # use existing lane on the left to update adjacent lane lines
    def updatePolyFitRight(self, curImg, leftLane):
        diff = np.polysub(leftLane.lines[leftLane.left].currentFit,
                          leftLane.lines[leftLane.right].currentFit)
        self.currentFit = np.polyadd(
            leftLane.lines[leftLane.right].currentFit, diff)
        poly = np.poly1d(self.currentFit)
        self.y_inds = leftLane.lines[leftLane.right].y_inds
        self.currentX = poly(self.y_inds)
        self.x_inds = self.currentX

        if len(self.y_inds) > cf.LINE_UPDATE_THRESH:
            self.confidence = len(self.y_inds) / self.maxConfidence
            if self.confidence >= 0.5:
                self.isDetected = True
                if self.confidence > 1:
                    self.confidence = 1  
            else:
                self.isDetected = False

            # create mask
            xy1 = np.column_stack(
                (self.currentX + self.maskDelta, self.y_inds)).astype(np.int32)
            xy2 = np.column_stack(
                (self.currentX - self.maskDelta, self.y_inds)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)
            self.lanemask = np.zeros_like(self.lanemask)
            cv2.fillConvexPoly(self.lanemask, self.linePoly, 64)
            # add bottom point
            y_inds = np.append(self.y_inds, cf.IMG_HEIGHT - 1)
            x_inds = poly(y_inds)
            self.XYPolyLine = np.column_stack((x_inds, y_inds)).astype(np.int32)

            self.bestFit = self.currentFit
    
    # def drawPolyLine(self, curImg, size=5):
    #     if self.side == 1:
    #         color = (255,0,0)
    #     else:
    #         color = (0,0,255)
    #     cv2.polylines(curImg, [self.XYPolyLine], 0, color, size)
    
    def fitPolyPrior(self, deg=2):
        if len(self.y_inds) > cf.LINE_CREATE_THRESH:
            self.confidence = 0.5
            self.maxConfidence = len(self.y_inds) * 2
            self.isDetected = True

            self.currentFit = np.polyfit(self.y_inds, self.x_inds, deg)
            poly = np.poly1d(self.currentFit)
            self.y_inds = self.y_inds[::-1]
            self.currentX = poly(self.y_inds)

            # # create mask
            # xy1 = np.column_stack(
            #     (self.currentX + self.polyDelta, self.y_inds)).astype(np.int32)
            # xy2 = np.column_stack(
            #     (self.currentX - self.polyDelta, self.y_inds)).astype(np.int32)
            # self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)
            # self.lanemask = np.zeros_like(self.lanemask)
            # cv2.fillConvexPoly(self.lanemask, self.linePoly, 64)
            # # add bottom point
            # x_inds = poly(self.y_inds)
            # y_inds = np.append(self.y_inds, cf.IMG_HEIGHT - 1)
            # x_inds = np.append(self.x_inds, self.pixelBase)
            # self.XYPolyLine = np.column_stack((x_inds, y_inds)).astype(np.int32)

            self.bestFit = self.currentFit
            self.linspace_x = self.bestFit[0]*self.linspace_y**2 + self.bestFit[1]*self.linspace_y + self.bestFit[2]
        else:
            self.confidence = 0.0
            self.isDetected = False
    
    def fitPolySecond(self, deg=2):
        if len(self.y_inds) > cf.LINE_UPDATE_THRESH:
            self.currentFit = np.polyfit(self.y_inds, self.x_inds, deg)
            self.diffFit = self.currentFit - self.bestFit
            if abs(sum(self.diffFit)) < 20:
                poly = np.poly1d(self.currentFit)
                x = poly([cf.IMG_HEIGHT - 1])
                self.y_inds = np.append(self.y_inds, cf.IMG_HEIGHT - 1)
                self.x_inds = np.append(self.x_inds, x[0])
                if abs(self.pixelBase - x[0] > 50):
                    self.confidence = 0.0
                    self.isDetected = False
                    return

                self.pixelBase =  x[0]
                self.currentX = poly(self.y_inds)

                # self.XYPolyLine = np.column_stack(
                #     (self.currentX, self.y_inds)).astype(np.int32)
                # xy1 = np.column_stack(
                #     (self.currentX + self.maskDelta, self.y_inds)).astype(np.int32)
                # xy2 = np.column_stack(
                #     (self.currentX - self.maskDelta, self.y_inds)).astype(np.int32)
                # self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)
                # self.lanemask = np.zeros_like(self.lanemask)
                # cv2.fillConvexPoly(self.lanemask, self.linePoly, 64)

                self.bestFit = (self.currentFit + self.bestFit) / 2
                self.linspace_x = self.bestFit[0]*self.linspace_y**2 + self.bestFit[1]*self.linspace_y + self.bestFit[2]

                self.confidence = len(self.y_inds) / self.maxConfidence
                if self.confidence >= 0.5:
                    self.isDetected = True
                    if self.confidence > 1:
                        self.confidence = 1  
                else:
                    self.confidence = 0.0
                    self.isDetected = False
            else:
                self.confidence = 0.0
                self.isDetected = False
        else:
                self.confidence = 0.0
                self.isDetected = False
    
    def measureCurvature(self):
        y = int(self.y * 3 / 4)
        if self.isDetected:
            fitCurvature = np.polyfit(self.y_inds * cf.YM_PER_PIX, self.currentX * cf.XM_PER_PIX, 2)
            self.curvature = ((1 + (2*fitCurvature[0]*y*cf.YM_PER_PIX + fitCurvature[1])**2)**1.5) / np.absolute(2*fitCurvature[0])
        else:
            self.curvature = 0
        return self.curvature
