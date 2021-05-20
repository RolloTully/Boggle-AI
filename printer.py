from escpos.printer import Usb     #for usb printers
printer = Usb(0x04b8, 0x0202, 10)# VendorID, ProductID, TImeout
printer.text("Hello World")
