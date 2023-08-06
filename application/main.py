# create a tkiner gui with a textbox using grid layout and a button
# when the button is clicked, the text in the textbox is printed to the console
# the button is disabled after it is clicked

import tkinter as tk
from tkmacosx import Button as button #should be cross platform
import torch
import torch.nn as nn
import os
import pickle


class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        # ensure window is not resizable
        self.master.resizable(False, False)
        self.pack()
        self.create_menu()
        self.create_widgets()

        # create event handler for textbox
        self.textbox.bind("<Key>", self.count_text)

        # create event handler for button
        self.button["command"] = self.button_click

        # load in lists for words + indices
        self.index_to_word = {}
        self.word_to_index = {}
        self.load_word_arrays()

        # load pretrained model
        model_path = os.path.realpath(os.path.join(os.getcwd(), 'model.pth'))
        self.model = self.load_network()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval

    # Create menu bar to switch between light and dark modes
    def create_menu(self):
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)
        
        visualMenu = tk.Menu(self.menubar, tearoff=0)
        visualMenu.add_command(label = "Light Mode", command = self.to_light)
        visualMenu.add_command(label = "Dark Mode", command = self.to_dark)
        self.menubar.add_cascade(label = "Appearance", menu = visualMenu)

    # switches the theme to light mode
    def to_light(self):
        global theme
        
        if theme == 1:
            return
        else:
            theme = 1
            self.destroy()
            self.__init__(self.master)

    # switches the theme to dark mode
    def to_dark(self):
        global theme
        
        if theme == 2:
            return
        else:
            theme = 2
            self.destroy()
            self.__init__(self.master)

    # creates the widgets for the UI
    def create_widgets(self):

        # calling global variables
        # global textbox_colour, textbox_border, textbox_text_colour, button_colour, button_text_colour, panel_colour, panel_text_colour
        global theme

        if theme == 1: #light theme
            textbox_colour = "#ebfcff"
            textbox_border = "#317f8c"
            textbox_text_colour = "#353A3F"
            button_colour = "#c73e08"
            button_text_colour = "#FFFFFF"
            panel_colour = "#FFFFFF"
            panel_text_colour = "#c73e08"

        elif theme == 2: #dark theme
            textbox_colour = "#eff3f4"
            textbox_border = "#1d9bf0"
            textbox_text_colour = "#18222e"
            button_colour = "#f75b2f"
            button_text_colour = '#ffffff'
            panel_colour = "#243447"
            panel_text_colour = "#ffffff"



        # change the title of the window
        self.master.title("Twitter Sentiment Analysis")

        self.textbox = tk.Text(self, width=48, fg = textbox_text_colour, bg=textbox_colour, font=FONT)
        self.textbox.grid(row=0, column=0, columnspan=2, rowspan=3, sticky = "nsew", padx = 1)


        # create a label to display the number of characters in the textbox
        self.label = tk.Label(self, text="/280")
        self.label.grid(row=4, column=1, sticky="we")


        # create a button
        self.button = button(self, text="Analyze", command=self.print_text, bg=button_colour, highlightbackground=panel_colour, fg=button_text_colour, font=FONT, borderless=1)
        # configure the button to be in the 4th row and 0th column, on the left side
        self.button.grid(row=4, column=1, sticky="e", pady=5)


        # create a panel to the right of the textbox with grid layout
        self.panel = tk.Frame(self)
        self.panel.grid(row=0, column=3, rowspan=3, sticky="nsew")
        

        # create a title for the panel
        self.panel_title = tk.Label(self.panel, text="Summary of Analysis")
        # add padding to the title
        self.panel_title.grid(row=0, column=0, columnspan=2, sticky="we", padx=15, pady=15)


        # create a string to display classification labels in the panel
        text = "Respect: \n\nInsult: \n\nHumiliate: \n\nStatus: \n\nDehumanize:\n\nViolence:\n\nGenocide:\n\nAttack Defend:\n\n"
        # create a label to display the text
        self.panel_text = tk.Label(self.panel, text=text, bg=panel_colour, fg=panel_text_colour, font=FONT, justify="left")
        # add the label to the panel
        self.panel_text.grid(row=1, column=0, sticky="w", padx=15)


        # set background color of elements
        self["bg"] = panel_colour

        self.label["bg"] = panel_colour
        self.label["fg"] = panel_text_colour
        self.label["font"] = FONT

        self.textbox["bg"] = textbox_colour
        self.textbox["highlightbackground"] = textbox_border
        self.textbox["fg"] = textbox_text_colour
        self.textbox["highlightthickness"] = 3
        self.textbox["borderwidth"] = 15 # padding around the text inside the textbox
        self.textbox["font"] = FONT

        self.panel["bg"] = panel_colour

        self.panel_title["bg"] = panel_colour
        self.panel_title["fg"] = panel_text_colour
        self.panel_title["font"] = TTLE_FONT


    # print the text in the textbox to the console when the button is clicked
    def print_text(self):
        text = self.textbox.get("1.0", "end")
        self.count_text()


    # update counter as the user types
    def count_text(self, event=None):
        # get the text from the textbox
        text = self.textbox.get("1.0", "end")
        # count the number of characters in the text
        count = len(text)
        # display the number of characters in the label
        self.label["text"] = str(count)+" / 280"

        #disable button if character count is greater than 280
        if count > 280:
            self.button["state"] = "disabled"
        else:
            self.button["state"] = "normal"
        return count
    

    # retrieve output from model using the text 
    def get_output(self, text):
        # convert text to tensor
        tensor = self.text_to_tensor(text)
        # get output from model
        output = self.model(tensor)
        print (output)


    # disable button if character count is greater than 280
    def disable_button(self, event=None):
        if self.count_text() > 280:
            self.button["state"] = "disabled"
        else:
            self.button["state"] = "normal"


    # update text in panel when user clicks button
    def update_panel(self, results):

        global theme

        if theme == 1: # light theme
            panel_colour = "#FFFFFF"
            panel_text_colour = "#c73e08"

        elif theme == 2: # dark theme
            panel_colour = "#243447"
            panel_text_colour = "#ffffff"


        # multiply and round all entries of the results array
        if len(results) == 0:
            text = "{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n".format("--", "--", "--", "--", "--", "--", "--", "--")
        else:
            results = [str(round(i * 100, 2)) for i in results]
            text = "{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n{}%\n\n".format(results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7])
        self.result_text = tk.Label(self.panel, text=text, bg=panel_colour, fg=panel_text_colour, font=FONT, justify="right")
        self.result_text.grid(row=1, column=1, sticky="e", padx=15)


    # define button click event handler
    def button_click(self):
        if self.count_text() > 280:
            self.button["state"] = "disabled"
        else:
            self.button["state"] = "normal"
            scores = self.predict_sentiment(self.model, self.textbox.get("1.0", "end"))
            self.update_panel(scores)


    # load index-to-word and word-to-index arrays from pickle files
    def load_word_arrays(self):
        # create appropriate paths
        iw_loc = os.path.realpath(os.path.join(os.getcwd(), 'index_to_word.pkl'))
        wi_loc = os.path.realpath(os.path.join(os.getcwd(), 'word_to_index.pkl'))

        with open(iw_loc, 'rb') as f:
            while True:
                try:
                    self.index_to_word = pickle.load(f)
                except EOFError:
                    break

        with open(wi_loc, 'rb') as f:
            while True:
                try:
                    self.word_to_index = pickle.load(f)
                except EOFError:
                    break

        print("Successfully loaded info arrays.")


    # load and return a specified network architecture
    def load_network(self):
        NETWORK_TYPE = 'GRU' # specify the specific network type
        INPUT_DIM=int(len(self.word_to_index)) # size of the vocabulary (number of words, arbitrary, but 10k is a good number)
        EMBEDDING_DIM = 256
        HIDDEN_DIM = 100
        OUTPUT_DIM = 8 # size of the output layer. Fixed to 8 for this project
        N_LAYERS = 2 # number of stacked RNN type layers. Please note that this is not the number of layers in the network. Only use 1 or 2 or else the network becomes too complex
        BIDIRECTIONAL = True # whether to use a bidirectional network
        DROPOUT = 0.35 
        return NETWORK(NETWORK_TYPE, INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


    # split tweet into a list of words (punctuation is preserved)
    def split_tweet(self, tweet):
        # separate punctuations
        tweet = tweet.replace(".", " . ") \
                    .replace(",", " , ") \
                    .replace(";", " ; ") \
                    .replace("?", " ? ")
        return tweet.lower().split()

    # predict the sentiment for a given sentence using the provided neural network.
    def predict_sentiment(self, net, sentence):
        
        idxs = [self.word_to_index[w]        # lookup the index of word
                        for w in self.split_tweet(sentence)
                        if w in self.word_to_index] # keep words that has an embedding
        tensor = torch.tensor(idxs)  # convert sentence to tensor
        tensor = tensor.unsqueeze(0)  # change shape from [n_words] to [n_words, 1]
        output = net(tensor)  # get predictions from network

        if len(output) != 0:
            # convert to probabilities
            output = torch.sigmoid(output)
            return output.tolist()[0]
        else:
            return []


class NETWORK(nn.Module):
    """
    The class object for the model network.
    Attributes:
    emb: the type of embedding
    hidden_size: the number of layers
    nn: the actual neural network
    fc: the activation layer
    """

    # param: type:str
    # param: vocab_size:int
    # param: embedding_dim:int
    # param: hidden_dim:int
    # param: output_dim:int
    # param: n_layers:int
    # param: bidirectional:bool
    # param: dropout:float
    # return: void
    # initializes the neural network with the given parameters
    def __init__(self, type, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.type = type
        if self.type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif self.type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif self.type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        else:
            raise Exception("Invalid RNN type")
        self.layers = n_layers
        self.fc = nn.Linear(hidden_dim*n_layers, 256)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    # param: text:torch.Tensor
    # return: torch.Tensor
    # forward pass of the neural network
    def forward(self, text):
        try:
            embedded = self.embedding(text)
            if self.type == "LSTM":
                output, (hidden, cell) = self.rnn(embedded)
            elif self.type == "GRU":
                output, hidden = self.rnn(embedded)
            if self.layers == 1:
                hidden = self.dropout(hidden[0:,:])
            else:
                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            hidden = self.relu(self.fc(hidden))
            hidden = self.relu(self.fc1(hidden))
            return self.fc2(hidden)
        except:
            return []


FONT = ("Cambria", 12)
TTLE_FONT = ("Cambria", 16)

theme = 1

root = tk.Tk()
app = App(master=root)
app.mainloop()