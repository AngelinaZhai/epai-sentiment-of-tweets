import tkinter as tk 
from tkinter import * 

root = tk.Tk() #Creates an instance of a Tk object
root.title("Tkinter Demo") #Sets the title of the window
root.geometry("600x800") #Sets the size of the window

#Creates a label widget
my_label = tk.Label(root, text="Hello World!", font=("Arial", 40), fg='blue', bg='yellow')
my_label.pack()

#Create a textbox
my_textbox = tk.Text(root, height=10, width=50) 
my_textbox.pack(padx=100, pady=10)

#create a frame
my_frame = tk.Frame(root)
# my_frame.columnconfigure(0, weight=1)
# my_frame.rowconfigure(0, weight=1)s

#Create a button
my_button = tk.Button(my_frame, text="Click Me!")
my_button.grid(row=0, column=0, padx=10, pady=10)

#Create a second button
my_button2 = tk.Button(my_frame, text="Click Me!")
my_button2.grid(row=0, column=1, padx=10, pady=10)

my_frame.pack()

root.mainloop()