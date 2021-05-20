from fontTools.ttLib import TTFont
from multiprocessing import Pool   #just to speed things up
import tensorflow as tf            #text classification
from tqdm import tqdm              #fancy loading bars
import numpy as np                 # speed again
import imutils                     #general image operations
import cv2                         #Image processing
cv2.setUseOptimized(True)          #Setting this flag to be true allows OpenCv to multiproccess and use the AVX/2 and SSE2 inctruction sets. ~2x preformace
'''
IMPORTANT NOTE TO FUTURE ROLLO!!!
RMEMBER THAT THIS PROGRAM USES 27 LETTERS IN ITS APLHABET WITH THE 27TH BEING @ -> Qu and all q's have been removed bringing it back down to 26
'''
class Client():
    def __init__(self):
        self.port=49152                                             #Camera server port, DO NOT CHANGE
        self.host="192.168.1.74"                                    #Camera Server IP, change as needed
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #Creates Python socket object
        self.s.connect((self.host,self.port))                       #Connects to camera server

    def Send(self,Inst):                                            #Sends commands to camera server
        self.byte_Inst = Inst.encode("utf-8")                       #Encodeds commands as bytes
        self.s.send(self.byte_Inst)                                 #Sends command
    def Recive(self):
        self.bytes = ''.encode("utf-8")                             #Creates an empty start byte
        self.stop = False
        print("Image tranmission in progress, Please wait.")
        while (not self.stop) and (not len(self.bytes)>1000):       #Start loop
            self.incoming=self.s.recv(1000)                         #recived 1000 bytes of the image
            print(len(self.incoming))
            self.bytes+=self.incoming                               #add bytes to the empty byte
            if len(self.incoming) == 0 and len(self.bytes)>1000:    #repeates loop until no new bytes are reviced within the socket timeout period
                self.stop = True
        if len(self.bytes)!=0:
            self.new_image=open("Image_Original.jpg","wb")          #Opens a blank image
            self.new_image.write(self.bytes)                        #writes bytes to blank image
            self.new_image.close()                                  #Closes file stream
        print(colored("Image Recived!","green"))

class ImageProcesses():
    def __init__(self):
        self.model = tf.keras.models.load_model('model.h5')                                                #Loads trained classification model
        self.closingkernel = np.ones((5,5),np.uint8)                                                       # defines morphological closing kernal
        self.smoothingkernel = np.ones((5,5),np.float32)/25                                                # defines 2D smoothing kernal
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
    def changesize(self, img):
        self.width, self.height = img.shape[:-1]
        self.min = min([self.width, self.height])/500
        self.resized = cv2.resize(img, (int(self.height/self.min), int(self.width/self.min)))
        return self.resized
    def classify(self, board_segments):
        for self.segment in board_segments:
            self.segment = self.segment.astype(np.float32) # Using float16 encoded data may lead to significant performace increases
            self.segment = cv2.resize(self.segment,(200,200)) #the model is trained on 100x100 images so cannot take images of higher resolution
            self.segment  = np.reshape(self.segment,(1,200,200,1))/255.0 #tf is stupid so you have to reshape the array so i advise you dont touch this
            self.prediction = self.model.predict(self.segment)#runs the images though the CNN and returnes a probability distrobution
            self.num = np.argmax(self.prediction)#returns the index of the greatest probability
            return chr(65+self.num)
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
           self.angles.append((((np.arctan2([self.y1-self.y2], [self.x1-self.x2])/np.pi)*180)+180).astype(np.int)) #np.arctan2 can be changed to cv2.fastacrtan for better performance
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
        self.board = self.classify(np.reshape(self.cubes,(16,100,100))))
        return self.board
class Printer(object):
    def __init__(self):
        self.font = TTFont('C://Windows//Fonts//712_serif.ttf')
        self.cmap = self.font['cmap']
        self.t = self.cmap.getBestCmap()
        self.s = self.font.getGlyphSet()
        self.outputfile = open("Printeroutput","w")
        self.spaces = [" "," "," "," "," "]
        self.dividers=["╔═╦═╦═╦═╗","╠═╬═╬═╬═╣","╠═╬═╬═╬═╣","╠═╬═╬═╬═╣","╚═╩═╩═╩═╝"]
        self.dividercharacter = "║"
    def pixel_width(unicode_text):
        if ord(self.c) in self.t and self.t[ord(self.c)] in self.s:
            return self.s[self.t[ord(self.c)]].width
        else:
            return self.s['.notdef'].width

    def formatboard(self):
        for div in self.dividers:
            self.outputfile.write(div+"\n")

    def printlist(self, board, array):
        self.formatboard(board)
        cv2.imwrite("printimg.png", self.filledboard)
        self.printer.image("printimg.png")
        for element in array:
            self.outputfile.write(element+"\n")
        print(self.printer.output)
class Board(object):
    def __init__(self):
        self.boggle_Board = []
        self.words = [x.strip() for x in open('words_alpha.txt','r')]
    '''
    isadjacent() and find_word() have been blackboxed
    '''
    def isadjecent(self, coordset1, coordset2):
        '''Checks if contencts of coordset1 is adjacent to coordset2'''
        self.offsets = [[x,y] for y in range(1,-2,-1) for x in range(1,-2,-1)]
        self.returnarray = []
        for set1 in coordset1:
            self.alladjacentcells = np.array(list(np.add(np.tile(set1,(9,1)), self.offsets)))
            for set2 in coordset2:
                if set2.tolist() in self.alladjacentcells.tolist():
                    self.returnarray.append(set2)
        return self.returnarray
    def find_word(self, word):
        self.printer.printer.text("Testing: "+word+"\n")
        if np.all(list(map(lambda letter:(letter in self.boggle_Board) and word.count(letter)<=np.count_nonzero(self.boggle_Board ==letter),word))): #checks if letter in word are in board and that there are the correct number, working
            ## TODO: check for adjacency of consecutive letters
            self.letterindecies = list(map(lambda letter:np.array(np.where(self.boggle_Board == letter)).flatten('F'), list(word))) #finds position of letters on board
            for index in range(len(self.letterindecies)):
                self.setlength = len(self.letterindecies[index])
                if self.setlength > 2:
                    self.chunks = self.setlength/2
                    self.letterindecies[index] = np.split(self.letterindecies[index],self.chunks)
                else:
                    self.letterindecies[index] = [self.letterindecies[index]] #working
            self.previous = self.isadjecent(self.letterindecies[1], self.letterindecies[0])
            self.history = np.array([self.previous])
            for index in range(0, len(self.letterindecies)-1):
                self.history = np.reshape(self.history,(-1,2))

                self.previous = self.isadjecent(self.previous, self.letterindecies[index+1])
                self.newprevious = []
                for element in self.previous:
                    if element.tolist() not in self.history.tolist():
                        self.newprevious.append(element)
                self.previous = self.newprevious
                if len(self.previous) == 0:
                    break
                self.history = np.append(self.history,self.previous, axis = 0)
            else:
                return [word, self.history.tolist()]
    def find_words(self):
        with Pool(8) as pool:
            self.output = pool.map(self.find_word, tqdm(self.words))
        self.correctedarray = []
        for element in self.output:
            if element != None:
                self.correctedarray.append(element)
        return self.correctedarray
class Main(object):
    def __init__(self):
        self.client = Client()
        self.imageprocesses = ImageProcesses()
        self.board = Board()
        self.printer = Printer()
        self.mainloop()

    def mainloop(self):
        #input("Ready. Press Enter To Start")
        #self.client.Send("Get_Img")
        #self.client.Recive()
        self.boardimage = cv2.imread("Image_Original.jpg")                                                     # read in board image
        self.resized = self.imageprocesses.changesize(self.boardimage)                                                    # adjust the size of image for further processing
        self.grayscale = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)                                    # get grayscale version of image
        self.bluechannel, self.greenchannel, self.redchannel = cv2.split(self.resized)                                                  # get indivudual colour channels of input image
        self.mask, self.edges = self.imageprocesses.FindCroppingMask(self.bluechannel)
        self.maskedimages = self.imageprocesses.ApplyMask(self.mask, [self.resized, self.edges, self.grayscale, self.mask])
        self.amounttorotate = self.imageprocesses.Straighten(self.grayscale, self.resized)
        self.boggle_Board = self.imageprocesses.FindGrid(self.maskedimages[2], self.maskedimages[3])
        self.board.boggle_Board = self.boggle_Board
        self.foundwords = self.board.findwords()
        self.printer.printlist(self.boggle_Board, self.foundwords)
if __name__ == '__main__':
    Main()
