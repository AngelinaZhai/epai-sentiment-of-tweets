# create a tkiner gui with a textbox using grid layout and a button
# when the button is clicked, the text in the textbox is printed to the console
# the button is disabled after it is clicked

import tkinter as tk
from tkmacosx import Button as button #should be cross platform


theme = 1

if theme == 1: #light theme
    TEXTBOX_COLOUR = "#ebfcff"
    TEXTBOX_BORDER = "#317f8c"
    TEXTBOX_TEXT_COLOUR = "#353A3F"
    BUTTON_COLOUR = "#c73e08"
    BUTTON_TEXT_COLOUR = "#FFFFFF"
    PANEL_COLOUR = "#FFFFFF"
    PANEL_TEXT_COLOUR = "#c73e08"

elif theme == 2: #dark theme
    TEXTBOX_COLOUR = "#eff3f4"
    # TEXTBOX_COLOUR = "#081f30"
    TEXTBOX_BORDER = "#1d9bf0"
    TEXTBOX_TEXT_COLOUR = "#18222e"
    BUTTON_COLOUR = "#f75b2f"
    BUTTON_TEXT_COLOUR = '#ffffff'
    PANEL_COLOUR = "#243447"
    PANEL_TEXT_COLOUR = "#ffffff"
    

FONT = ("Cambria", 12)
TTLE_FONT = ("Cambria", 16)


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

        self.textbox = tk.Text(self, width=48, fg = TEXTBOX_TEXT_COLOUR, bg=TEXTBOX_COLOUR, font=FONT)
        self.textbox.grid(row=0, column=0, columnspan=2, rowspan=3, sticky = "nsew", padx = 1)


        # create a label to display the number of characters in the textbox
        self.label = tk.Label(self, text="/280")
        self.label.grid(row=4, column=1, sticky="we")


        #create a button
        self.button = button(self, text="Analyze", command=self.print_text, bg=BUTTON_COLOUR, highlightbackground=PANEL_COLOUR, fg=BUTTON_TEXT_COLOUR, font=FONT, borderless=1)
        #configure the button to be in the 4th row and 0th column, on the left side
        self.button.grid(row=4, column=1, sticky="e", pady=5)


        #create a panel to the right of the textbox with grid layout
        self.panel = tk.Frame(self)
        self.panel.grid(row=0, column=3, rowspan=3, sticky="nsew")
        

        #create a title for the panel
        self.panel_title = tk.Label(self.panel, text="Summary of Analysis")
        #add padding to the title
        self.panel_title.grid(row=0, column=0, columnspan=2, sticky="we", padx=15, pady=15)


        #create a string to display classification labels in the panel
        text = "Respect: \n\nInsult: \n\nHumiliate: \n\nStatus: \n\nDehumanize:\n\nViolence:\n\nGenocide:\n\nAttack Defend:\n\n"
        #create a label to display the text
        self.panel_text = tk.Label(self.panel, text=text, bg=PANEL_COLOUR, fg=PANEL_TEXT_COLOUR, font=FONT, justify="left")
        #add the label to the panel
        self.panel_text.grid(row=1, column=0, sticky="w", padx=15)


        #set background color of elements
        self["bg"] = PANEL_COLOUR

        self.label["bg"] = PANEL_COLOUR
        self.label["fg"] = PANEL_TEXT_COLOUR
        self.label["font"] = FONT

        self.textbox["bg"] = TEXTBOX_COLOUR
        self.textbox["highlightbackground"] = TEXTBOX_BORDER
        self.textbox["fg"] = TEXTBOX_TEXT_COLOUR
        self.textbox["highlightthickness"] = 3
        self.textbox["borderwidth"] = 15 #padding around the text inside the textbox
        self.textbox["font"] = FONT

        self.panel["bg"] = PANEL_COLOUR

        self.panel_title["bg"] = PANEL_COLOUR
        self.panel_title["fg"] = PANEL_TEXT_COLOUR
        self.panel_title["font"] = TTLE_FONT


    #print the text in the textbox to the console when the button is clicked
    def print_text(self):
        text = self.textbox.get("1.0", "end")
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
        text = "{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n".format('--', '--', '--', '--', '--', '--', '--', '--')
        self.result_text = tk.Label(self.panel, text=text, bg=PANEL_COLOUR, fg=PANEL_TEXT_COLOUR, font=FONT, justify="right")
        self.result_text.grid(row=1, column=1, sticky="e", padx=15)

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