import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time
import tensorflow as tf
cv2.setUseOptimized(True)                                                                                  # Setting this flag to be true allows OpenCv to multiproccess. ~2x preformace
class ImageProcesses():
    def __init__(self):
        self.model = tf.keras.models.load_model('model.h5')
        self.SIFT = cv2.xfeatures2d.SIFT_create()
        self.BRUTEFORCEmatching = cv2.BFMatcher()
        self.closingkernel = np.ones((5,5),np.uint8)                                                       # defines morphological closing kernal
        self.smoothingkernel = np.ones((5,5),np.float32)/25                                                # defines 2D smoothing kernal
        self.corner = cv2.imread("board.jpg", 0)
        self.kp, self.des = self.SIFT.detectAndCompute(self.corner,None)

    def FindCroppingMask(self, img):                                                                       # img should be a high contrast image
        '''Creates mask for board'''                                                                       # working
        self.denoised = cv2.GaussianBlur(img,(5,5),-1)                                                     # removed image noise aka. smoothing
        self.edges = cv2.Canny(self.denoised, 40, 150, apertureSize = 3)                                   # Performs canny edge detection
        self.closededges= cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, self.closingkernel)                # closes holes in detected edges
        _,self.contours,_ = cv2.findContours(self.closededges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # finds contours in edges
        self.index = 0                                                                                     # finds contour with greatst hull area
        self.current_max = 0
        for x in range(len(self.contours)):
            self.hull = cv2.convexHull(self.contours[x])
            self.hull_area = cv2.contourArea(self.hull)
            if self.hull_area > self.current_max:
                self.current_max = self.hull_area
                self.index = x
        self.contourpoints = self.contours[self.index].reshape(self.contours[self.index].shape[0],2)       # reshapes contour array for easier handling
        self.height, self.width = img.shape
        self.maskboundary = np.zeros((self.height, self.width))
        cv2.drawContours(self.maskboundary, self.contours[self.index], -1, 255, -1)                        # draws selected boundary contour on to blank mask
        self.closedboundary= cv2.morphologyEx(self.maskboundary, cv2.MORPH_CLOSE, self.closingkernel)      # closes holes in drawn contour
        _, self.binarized = cv2.threshold(self.closedboundary,127,255,cv2.THRESH_BINARY)                   # binarizes
        self.height, self.width = self.closedboundary.shape
        self.mask = np.zeros((self.height+2,self.width+2)).astype(np.uint8)                                # makes mask for floodFilling
        self.maskedimage = cv2.floodFill(self.binarized.astype(np.uint8), self.mask, (0,0), newVal=255)[1] # masks ROI
        self.maskedimage = 255-self.maskedimage                                                            # Inverts image
        self.maskedimage = cv2.morphologyEx(self.maskedimage, cv2.MORPH_CLOSE, self.closingkernel)         # tidys up slight imperfections in the mask
        #cv2.imshow("board mask",self.maskedimage) #Remove
        #cv2.waitKey(4000)
        return self.maskedimage, self.edges

    def classify(self, board_segments):
        for self.segment in board_segments:
            self.segment = self.segment.astype(np.float32) # Using float16 encoded data may lead to significant performace increases
            self.segment = cv2.resize(self.segment,(200,200)) #the model is trained on 100x100 images so cannot take images of higher resolution
            self.segment  = np.reshape(self.segment,(1,200,200,1))/255.0 #tf is stupid so you have to reshape the array so i advise you dont touch this
            self.prediction = self.model.predict(self.segment)#runs the images though the CNN and returnes a probability distrobution
            self.num = np.argmax(self.prediction)#returns the index of the greatest probability
            self.board.append(chr(65+self.num))
        return self.board

    def ApplyMask(self, mask, array):                                                                      # array should contain grayscale, resized(full colour), and canny edges
        '''Crops and masks array of images''' #working
        self.maskedpoints = np.where(mask == 255)                                                          # find coordinates of positive mask
        self.background   = np.where(mask == 0)                                                            # finds coordinates of negative mask
        self.max_x = np.amax(self.maskedpoints[1])                                                         # find the maximum x value
        self.max_y = np.amax(self.maskedpoints[0])                                                         # find the maximum y value
        self.min_x = np.amin(self.maskedpoints[1])                                                         # find the minimum x value
        self.min_y = np.amin(self.maskedpoints[0])                                                         # find the minimum x value
        self.temp = []
        for img in array:                                                                                  # parses through array of images
            img[self.background[0], self.background[1]] = 0                                                # masks backgroud
            img = img[self.min_y:self.max_y,self.min_x:self.max_x]                                         # crops image
            self.temp.append(img)
        array = self.temp
        return array

    def Straighten(self, grayscale, fullcolour):
        '''Straightens board image based on grid in board'''                                               # working, unreliably
        self.img = cv2.GaussianBlur(grayscale,(5,5),-1)
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.img = 255 - self.img
        #self.edges = cv2.Canny(fullcolour, 40, 150, apertureSize = 3)                                      # performs canny edge detection
        #self.edges= cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, self.closingkernel)                      # closes gaps in canny edges
        self.lines = cv2.HoughLinesP(self.img, 1, np.pi/180, 30,minLineLength = 40, maxLineGap=5)       # finds board grids
        self.angles = []
        for line in self.lines:
           self.x1, self.y1, self.x2, self.y2 = line[0]
           self.angles.append((((np.arctan2([self.y1-self.y2], [self.x1-self.x2])/np.pi)*180)+180).astype(np.int))
        self.unique, self.counts = np.unique(self.angles, return_counts=True)
        self.unique = self.unique%90
        self.new_counts = []
        for x in range(max(self.unique)):
            self.new_counts.append(np.sum(self.counts[np.where(self.unique == x)]))
        self.counts = self.new_counts
        self.amounttorotate = 90-np.argmax(self.counts) # figure out what should go here
        return self.amounttorotate

    def FindGrid(self, grayscale, mask):
        '''Finds and crops straightend image in to individual grid squares'''
        self.pointarray = []
        self.height, self.width = mask.shape
        self.points = np.array(list(np.where(mask==255)))
        self.distance = np.hypot(self.points[0], self.points[1]).astype(np.int)
        print(self.points)
        self.pointarray.append(self.points[:,np.argmin(self.distance)][::-1])
        self.pointarray.append(self.points[:,np.argmax(self.distance)][::-1])

        self.points = np.array(list(np.where(np.rot90(mask,k=1)==255)))
        self.distance = np.hypot(self.points[0], self.points[1]).astype(np.int)
        self.pointarray.append([self.width,0]-self.points[:,np.argmin(self.distance)][::-1])
        self.pointarray.append([self.width,0]-self.points[:,np.argmax(self.distance)][::-1])
        self.pointarray = np.absolute(self.pointarray)
        self.dest = np.array([[0,0],[400,0],[0,400],[400,400]],np.float32)
        self.src = np.array(sorted(self.pointarray, key = lambda x: np.prod(x)),np.float32)
        self.mapping = cv2.getPerspectiveTransform(self.src, self.dest)
        self.warpedimage = cv2.warpPerspective(grayscale, self.mapping, (400, 400))
        self.cubes = np.array(list(map(lambda x:list(map(lambda y:self.warpedimage[y-100:y,x-100:x] ,range(100,500,100) )) ,range(100,500,100))))
        for row in self.cubes:
            for cube in row:
                cv2.imshow("", cube)
                cv2.waitKey(4000)
        print(self.classify(np.reshape(self.cubes,(16,100,100))))

#Note: make function to auto-tune hyper parameters
class Main():
    def __init__(self):
        self.imageprocesses = ImageProcesses()
        self.mainloop()

    def changesize(self, img):
        self.width, self.height = img.shape[:-1]
        self.min = min([self.width, self.height])/500
        self.resized = cv2.resize(img, (int(self.height/self.min), int(self.width/self.min)))
        return self.resized

    def mainloop(self):
        self.boardimage = cv2.imread("blankboard.jpg")                                                     # read in board image
        self.resized = self.changesize(self.boardimage)                                                    # adjust the size of image for further processing
        self.grayscale = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)                                    # get grayscale version of image

        self.bluechannel, self.greenchannel, self.redchannel = cv2.split(self.resized)                                                  # get indivudual colour channels of input image
        self.mask, self.edges = self.imageprocesses.FindCroppingMask(self.bluechannel)
        self.maskedimages = self.imageprocesses.ApplyMask(self.mask, [self.resized, self.edges, self.grayscale, self.mask])

        self.amounttorotate = self.imageprocesses.Straighten(self.grayscale, self.resized)

        self.gridpoints = self.imageprocesses.FindGrid(self.maskedimages[2], self.maskedimages[3])
        cv2.imshow("",self.maskedimages[2])
        cv2.waitKey(4000)



if __name__ == "__main__":
    Main()
