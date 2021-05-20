from escpos.printer import Usb     #for usb printers
from escpos.printer import Network #for network printers
from escpos.printer import Dummy   #for testing
from multiprocessing import Pool   #just to speed things up
import numpy as np                 # speed again
import cv2
from tqdm import tqdm
cv2.setUseOptimized(True)
'''

This is all the logic behind the word searching, i never ended up melding toghter this and the letter recognition so make it fully functional

'''

'''
IMPORTANT NOTE TO FUTURE !!!
RMEMBER THAT THIS PROGRAM USES 27 LETTERS IN ITS APLHABET WITH THE 27TH BEING @ -> Qu and all q's have been removed bringing it back down to 26
'''

class Printer(object):
    def __init__(self):
        self.blankboard = self.makeblankboard()
        self.printer = Dummy()#Usb(0x04b8, 0x0202, 0)# VendorID, ProductID, TImeout
        self.printer.text("test")

    def makeblankboard(self):
        self.blankimg = np.ones((401, 401))*255
        for x in range(0,500,100):
            cv2.line(self.blankimg,(0,x),(400,x),0,2)
            cv2.line(self.blankimg,(x,0),(x,400),0,2)
        return self.blankimg
    def makeboard(self, board):
        for y in range(0,4,1):
            for x in range(0,4,1):
                cv2.putText(self.blankboard,board[x,y],((y*100)+15,(x*100)+70),cv2.FONT_HERSHEY_COMPLEX,3,0,4)
        self.filledboard = self.blankboard
    def drawlines(self, array):
        return self.blankboard

    def printlist(self, board, array):
        cv2.imwrite("printimg.png", self.filledboard)
        self.printer.image("printimg.png")
        for element in array:
            self.printer.text("Word: "+element[0]+"\n")
            self.img = self.drawlines(element[1])
            cv2.imwrite("printimg.png", self.img)

            self.printer.image("printimg.png")

        self.printer.cut()

        print(self.printer.output)


class Main(object):
    def __init__(self):
        self.printer = Printer()
        self.boggle_Board = np.array([['e','o','s','l'],
                                      ['c','e','y','b'],
                                      ['d','s','n','h'],
                                      ['i','e','b','d']])

        self.boggle_Board = self.boggle_Board.ravel()
        np.random.shuffle(self.boggle_Board)
        self.boggle_Board = np.reshape(self.boggle_Board,(4,4))
        self.printer.makeboard(self.boggle_Board)
        print(self.boggle_Board)
        self.boardelements = np.unique(self.boggle_Board)
        self.words = [x.strip() for x in open('words_alpha.txt','r')]
        self.mainloop()

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

    def mainloop(self):
        with Pool(8) as pool:
            self.output = pool.map(self.find_word, tqdm(self.words))

        self.correctedarray = []
        for element in self.output:
            if element != None:
                self.correctedarray.append(element)
        self.printer.printlist(self.boggle_Board, self.correctedarray)




if __name__ == '__main__':
    Main()
