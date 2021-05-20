import os, socket, subprocess, pytesseract

from termcolor import colored
from colorama import init

from PIL import Image
init()
class Server():
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Host = "192.168.1.73"
        self.port = 49152
        self.socket.bind(("192.168.1.73",port))
    def recive_image(self):
        c,addr = s.accept()
        c.settimeout(10)
        print(colored(("Got connection from "+str(addr)),"green"))
        bytes = ''.encode("utf-8")
        for i in range(500):
            try:
                incoming=c.recv(20000000)
                bytes+=incoming
            except socket.timeout:
                break
        if len(bytes)!=0:
            new_image=open("Image.jpg","wb")
            new_image.write(bytes)
            new_image.close()
        c.close()





class Main():
    def __init__(self):
        self.camera = Camera()
        self.board = Board()
        self.server = Server()
        self.mainloop()

    def mainloop(self):
        self.camera.Take_Photo()
if __name__ =="__main__":
    main=Main()
