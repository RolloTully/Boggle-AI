import socket, pytesseract, cv2
import math as maths
from playsound import playsound
from PIL import Image
from colorama import init
from termcolor import colored
from gtts import gTTS
import numpy as np
import time
import shutil
import tkinter
init()

''' Old attempt to get this to have a thin linux client to simplify the interaction wih the enviroment so I dont have to drag a huge desktop around'''

class Process():
    def __init__(self):
        self.kernel_sharpening_filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 2
        self.params.maxThreshold = 200
        self.params.filterByArea = True
        self.params.minArea = 1500
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.5
        self.params.maxCircularity = 0.9
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.5
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.01

    def Filter(self, image):
        print("Processing, Please wait.")
        self.detector = cv2.SimpleBlobDetector_create(self.params)
        self.Img = cv2.imread(image)
        for self.y in self.Img:
            for self.pixel in self.y:
                self.pixel=[0,255-self.pixel[1],0]
        self.Grey_Img = cv2.resize(cv2.cvtColor(self.Img, cv2.COLOR_BGR2GRAY),(960,640))
        self.key_areas = self.detector.detect(self.Grey_Img)
        if len(self.key_areas)>=2:
            print(len(self.key_areas),"Markers found.")
            self.key_areas_coords = list(map(lambda x:list(map(lambda y:int(y),x.pt)),self.key_areas))
            self.Grouped_Key_Areas = [[self.y,self.x] for self.y in self.key_areas_coords for self.x in self.key_areas_coords]
            self.Grouped_Key_Areas_Range = [maths.hypot(self.Group[0][0]-self.Group[1][0],self.Group[0][1]-self.Group[1][1]) for self.Group in self.Grouped_Key_Areas]
            self.True_Markers = [self.Grouped_Key_Areas[self.Grouped_Key_Areas_Range.index(max(self.Grouped_Key_Areas_Range))]]
            self.Mid_Point =list(map(lambda x:int(x/2),list(map(lambda x:sum(x),zip(self.True_Markers[0][0],self.True_Markers[0][1])))))
            self.Radius = int(maths.hypot(self.Mid_Point[0]-self.True_Markers[0][0][0],self.Mid_Point[1]-self.True_Markers[0][0][1]))*0.9
            for self.x in range(len(self.Grey_Img)):
                for self.y in range(len(self.Grey_Img[self.x])):
                    self.R = ((self.x-self.Mid_Point[1])**2)+((self.y-self.Mid_Point[0])**2)
                    if self.R > self.Radius**2:
                        self.Grey_Img[self.x][self.y]=255
            self.F = 60#int(input("threshold: "))
            self.thresh, self.BAW = cv2.threshold(self.Grey_Img, self.F, 255, cv2.THRESH_BINARY)
            cv2.imwrite("Image_BAW.jpg",self.BAW)
            self.BAW = cv2.filter2D(self.BAW, -20, self.kernel_sharpening_filter)
            cv2.imwrite("Image_Sharp.jpg",self.BAW)
            self.BAW = cv2.medianBlur(self.BAW,5)
            cv2.imwrite("Image_Blur.jpg",self.BAW)
            self.BAW = cv2.resize(self.BAW,(960,1440))
            cv2.imwrite("Filtered_Image.jpg",self.BAW)
            return 1
        else:
            print("Error: Marker not found!")
            return 0

    def Show_Img(self,title,Img):
        cv2.imshow(title,cv2.resize(cv2.imread(Img,1),(1920,1080)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Board():
    def __init__(self):
        self.lang="en"
        self.Board=[]
    def find_word(self):
        pass
    def interpertate_Board(self):
        print("Content of Board: \n",pytesseract.image_to_string(Image.open("Filtered_Image.jpg")))
    def Read_Answers(self):
        self.text = ' '.join(self.Board)
        tts = gTTS(text=self.text,lang=self.lang)
        tts.save("Boggle_Answers.wav")
        playsound("Boggle_Answers.wav")

class Client():
    def __init__(self):
        self.port=49152
        self.host="192.168.1.74"
    def Send(self,Inst):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host,self.port))
        self.byte_Inst = Inst.encode("utf-8")
        self.s.send(self.byte_Inst)
        self.s.close()

    def Recive(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host,self.port))
        self.bytes = ''.encode("utf-8")
        self.stop = False
        print("Image tranmission in progress, bytes long, Please wait.")
        while (not self.stop) and (not len(self.bytes)>1000):
            self.incoming=self.s.recv(1000)
            print(len(self.incoming))
            self.bytes+=self.incoming
            if len(self.incoming) == 0 and len(self.bytes)>1000:
                self.stop = True
        self.s.close()
        if len(self.bytes)!=0:
            self.new_image=open("Image_Original.jpg","wb")
            self.new_image.write(self.bytes)
            self.new_image.close()
        print(colored("Image Recived!","green"))
class GUI():
    def __init__(self):
        self.client = Client()
        self.process = Process()
        self.board = Board()


class Main():
    def __init__(self):

        self.Mainloop()
    def Mainloop(self):
        while True:
            input("Ready. Press Enter To Start")
            self.client.Send("Get_Img")
            self.client.Recive()
            if self.process.Filter("Image_Original.jpg"):
                self.process.Show_Img("Filtered","Filtered_Image.jpg")

            self.board.interpertate_Board()
            c,r = shutil.get_terminal_size()
            print(colored("-"*r,"red"))

if __name__ == "__main__":
    main=Main()
