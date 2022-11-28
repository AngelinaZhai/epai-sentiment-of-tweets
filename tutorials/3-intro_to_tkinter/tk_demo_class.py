import tkinter as tk
from tkinter import *

'''
In an actual development team, we can't just have spaghetti code like in the simple case. 
It is better to organize each GUI window/feature into a class.
'''

class Main_Window():

    def __init__(self):

        self.root = tk.Tk() #Creates an instance of a Tk object

        self.root.geometry("800x600") #Sets the size of the window
        self.root.title("Tkinter Demo Class") #Sets the title of the window

        #Define the frame
        self.my_frame = tk.Frame(self.root) 

        #define title label
        self.my_label = tk.Label(self.my_frame, text="Type something here!", font=("Arial", 24))
        self.my_label.grid(row=0, column=0, rowspan=1, columnspan=3)

        #define textbox
        self.my_textbox = tk.Text(self.my_frame)
        self.my_textbox.grid(row=1, column=0, columnspan=3, rowspan=2, sticky="nsew")

        #define button
        self.my_button = tk.Button(self.my_frame, text="Submit", command=self.button_click) 
        #DO NOT CALL self.button_click() here. It will execute the function immediately.
        
        self.my_button.grid(row=3, column=2, columnspan=1, rowspan=1, sticky="e")

        self.my_frame.pack()

        self.root.mainloop()

    '''
    Define function inside the class so that variable names could be reused without conflict.
    Here we define a function that will be called when the button is clicked.
    '''
    def button_click(self):
        
        input = self.my_textbox.get('1.0', END)
        self.popup = tk.Tk() #Create a tk instance for the popup window
        self.popup.geometry("400x200")
        self.popup.title("Popup Msg")
        self.popup_label = tk.Label(self.popup, text=input)
        self.popup_label.pack()



Main_Window()