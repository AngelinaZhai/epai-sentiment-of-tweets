# create a tkiner gui with a textbox using grid layout and a button
# when the button is clicked, the text in the textbox is printed to the console
# the button is disabled after it is clicked

import tkinter as tk

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        # self.count_text()

        #create event handler for textbox
        self.textbox.bind("<Key>", self.count_text)



    def create_widgets(self):
        self.textbox = tk.Text(self)
        #configure the textbox to take up 3 columns and 3 rows
        self.textbox.grid(row=0, column=0, columnspan=3, rowspan=3)
        self.button = tk.Button(self, text="Click Me", command=self.print_text)
        #configure the button to be in the 4th row and 3rd column
        self.button.grid(row=4, column=3)
        # create a label to display the number of characters in the textbox
        self.label = tk.Label(self, text="Number of characters: ")
        self.label.grid(row=4, column=0)


        
    def print_text(self):
        # print(self.textbox.get())
        text = self.textbox.get("1.0", "end")
        print(text)
        self.count_text()

    #update counter as the user types
    def count_text(self, event):
        #get the text from the textbox
        text = self.textbox.get("1.0", "end")
        #count the number of characters in the text
        count = len(text)
        #display the number of characters in the label
        self.label["text"] = "Number of characters: " + str(count)

        #change the counter as the user types
        # self.textbox.bind("<Key>", self.count_text)



root = tk.Tk()
app = App(master=root)
app.mainloop()