# create a tkiner gui with a textbox using grid layout and a button
# when the button is clicked, the text in the textbox is printed to the console
# the button is disabled after it is clicked

import tkinter as tk

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        #ensure window is not resizable
        self.master.resizable(False, False)
        self.pack()
        self.create_widgets()
        # self.count_text()

        #create event handler for textbox
        self.textbox.bind("<Key>", self.count_text)

        #create event handler for button
        self.button["command"] = self.button_click



    def create_widgets(self):
        #change the title of the window
        self.master.title("Twitter Sentiment Analysis")
        self.textbox = tk.Text(self)
        #configure the textbox to take up 3 columns and 3 rows
        self.textbox.grid(row=0, column=0, columnspan=2, rowspan=3)
        #make textbox more narrow (can be tuned later)
        self.textbox["width"] = 50
        self.button = tk.Button(self, text="Analyze", command=self.print_text)
        #configure the button to be in the 4th row and 0th column, on the left side
        self.button.grid(row=4, column=0, sticky="w")
        # create a label to display the number of characters in the textbox
        self.label = tk.Label(self, text="/280")
        #configure the label to be in the 4th row and 3rd column, on the right side
        self.label.grid(row=4, column=3, sticky="e")

        #create a panel to the right of the textbox with grid layout
        self.panel = tk.Frame(self)
        self.panel.grid(row=0, column=3, rowspan=3, sticky="nsew")
        #create a title for the panel
        self.panel_title = tk.Label(self.panel, text="Summary of Analysis")
        #add padding to the title
        self.panel_title["padx"] = 15
        self.panel_title.grid(row=0, column=0)
        
    def print_text(self):
        # print(self.textbox.get())
        text = self.textbox.get("1.0", "end")
        print(text)
        self.count_text()

    #update counter as the user types
    def count_text(self, event=None):
        #get the text from the textbox
        text = self.textbox.get("1.0", "end")
        #count the number of characters in the text
        count = len(text)
        #display the number of characters in the label
        self.label["text"] = str(count)+" / 280"

        #disable button if character count is greater than 280
        if count > 280:
            self.button["state"] = "disabled"
        else:
            self.button["state"] = "normal"
        return count

    #disable button if character count is greater than 280
    def disable_button(self, event=None):
        if self.count_text() > 280:
            self.button["state"] = "disabled"
        else:
            self.button["state"] = "normal"

    # update text in panel when user clicks button
    def update_panel(self):
        #define the text to be displayed in the panel
        text = "This is the summary of the analysis"
        #create a label to display the text
        self.panel_text = tk.Label(self.panel, text=text)
        #add padding to the label
        self.panel_text["padx"] = 15
        #add the label to the panel
        self.panel_text.grid(row=1, column=0)

    # define button click event handler
    def button_click(self):
        if self.count_text() > 280:
            self.button["state"] = "disabled"
        else:
            self.button["state"] = "normal"
            self.update_panel()



root = tk.Tk()
app = App(master=root)
app.mainloop()