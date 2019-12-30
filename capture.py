import wx
import os
import sys
import threading
import pandas as pd
import queue
import sounddevice as sd
import soundfile as sf
import numpy


class SoundMachine:
    q = queue.Queue()
    stop_recording = False

    # Configuration Item Candidates
    sample_rate = 22050
    sub_type = "PCM_16"

    def __init__(self):
        self.reset()

    def reset(self):
        self.stop_recording = False
        sd.default.samplerate = self.sample_rate

    def record(self, filename):

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            print('.')
            if status:
                print(status, file=sys.stderr)
            self.q.put(indata.copy())

        try:
            # Make sure the file is opened before recording anything:
            with sf.SoundFile(filename, mode='x', samplerate=self.sample_rate, channels=1, subtype=self.sub_type) as file:
                with sd.InputStream(callback=callback):
                    while not self.stop_recording:
                        file.write(self.q.get())

                    sd.stop(True)
                    self.reset()
        except KeyboardInterrupt:
            print('\nRecording finished: ' + repr(filename))
        except Exception as e:
            exit(type(e).__name__ + ': ' + str(e))

    def stop(self):
        self.stop_recording = True

    def play(self, filename):
        try:
            data, fs = sf.read(filename, dtype='float32')
            sd.play(data)
            status = sd.wait()
        except Exception as e:
            exit(type(e).__name__ + ': ' + str(e))

        if status:
            exit('Error during playback: ' + str(status))

    def save(self):
        pass


class ViewModel:
    # Configuration Item Candidates
    record_id_prefix = "RR001"
    record_id_separator = "-"
    record_id_num_pad = 4
    record_subdir = "wav"
    collection_dir = "RRSpeech-1.0"
    data_dir = "data"

    def __init__(self):
        self.ready = False
        self.selected_sentence = "unknown"
        self.sentences = None
        self.max_index = 0
        self.current_sentence_index = 0
        self.sentence_filename = ""

        # setup data directory structure
        self.data_dir_structure = os.path.join(self.data_dir, self.collection_dir, self.record_subdir)
        self.ensureDir(self.data_dir_structure)

    def reset(self):
        self.ready = False
        self.selected_sentence = "unknown"
        self.sentences = None
        self.max_index = 0
        self.current_sentence_index = 0
        self.sentence_filename = ""

        # TODO: Check if we need to setup data directory structures here as well as in the init function

    def ensureDir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def previous(self):
        if not self.ready:
            return
        if self.current_sentence_index > 0:
            self.current_sentence_index = self.current_sentence_index - 1

    def next(self):
        if self.ready:
            if self.current_sentence_index < self.max_index:
                self.current_sentence_index = self.current_sentence_index + 1

    def loadSentenceFile(self, filename):
        self.reset()
        self.sentence_filename = filename
        self.sentences = pd.read_csv(self.sentence_filename, sep='|', header=None)
        self.sentences.columns = ['key', 'spoken', 'normalised']
        self.max_index = len(self.sentences) - 1
        self.ready = True

    def getCurrentSentence(self):
        if self.ready:
            sentence = self.sentences['spoken'][self.current_sentence_index]
            self.selected_sentence = sentence
            return {'text': sentence, 'id': self.current_sentence_index}

    def getRecordingFilename(self):
        index_format = '{2:0' + str(self.record_id_num_pad) + '}'
        name_format = '{0}{1}' + index_format
        name = name_format.format(self.record_id_prefix, self.record_id_separator, self.current_sentence_index)
        return os.path.join(self.data_dir_structure,name)


class AppFrame(wx.Frame):
    view_model = ViewModel()
    sound_machine = SoundMachine()
    sentence_text = None
    sentence_id = None

    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, 'Capture', size=(1200, 400))
        self.panel = wx.Panel(self)

        # Action Buttons

        # close_button = wx.Button(panel, label='Close', pos=(500, 330), size=(76, -1))
        # close_button.SetDefault()
        # self.Bind(wx.EVT_BUTTON, self.closeButton, close_button)
        self.Bind(wx.EVT_CLOSE, self.closeWindow)

        # Instructions
        title = wx.StaticText(self.panel, -1, "Say the following sentence...", (10, 10), (1180, -1), wx.ALIGN_CENTER)
        title.SetBackgroundColour('Dark Grey')
        title.SetForegroundColour('White')

        self.sentence_id = wx.StaticText(self.panel, -1, "-", (10, 150), (1180, -1), wx.ALIGN_LEFT)
        self.sentence_text = wx.StaticText(self.panel, -1, "Please load sentence file.", (10, 150), (1180, -1), wx.ALIGN_CENTER)

        margin_left = 400
        button_width = 76
        button_gap = 4
        button_panel_height = 260

        previous_button = wx.Button(self.panel, label='Prev', pos=(margin_left, button_panel_height), size=(button_width, -1))
        previous_button.SetBackgroundColour('Dark Grey')
        self.Bind(wx.EVT_BUTTON, self.previousButton, previous_button)
        next_button = wx.Button(self.panel, label='Next', pos=(margin_left + button_width + button_gap, button_panel_height), size=(button_width, -1))
        next_button.SetBackgroundColour('Dark Grey')
        self.Bind(wx.EVT_BUTTON, self.nextButton, next_button)
        rec_button = wx.Button(self.panel, label='Rec', pos=(margin_left + (button_width + button_gap) * 2, button_panel_height), size=(button_width, -1))
        rec_button.SetBackgroundColour('red')
        self.Bind(wx.EVT_BUTTON, self.recButton, rec_button)
        stop_button = wx.Button(self.panel, label='Stop', pos=(margin_left + (button_width + button_gap) * 3, button_panel_height), size=(button_width, -1))
        stop_button.SetBackgroundColour('Dark Grey')
        self.Bind(wx.EVT_BUTTON, self.stopButton, stop_button)
        play_button = wx.Button(self.panel, label='Play', pos=(margin_left + +(button_width + button_gap) * 4, button_panel_height), size=(button_width, -1))
        play_button.SetBackgroundColour('Dark Green')
        self.Bind(wx.EVT_BUTTON, self.playButton, play_button)

        # create a menu bar
        self.makeMenuBar()

        # and a status bar
        self.CreateStatusBar()
        self.SetStatusText("(c) 2020 Reece Robinson (Reece@TheRobinsons.gen.nz)")

    def makeMenuBar(self):
        """
        A menu bar is composed of menus, which are composed of menu items.
        This method builds a set of menus and binds handlers to be called
        when the menu item is selected.
        """

        # Make a file menu with Hello and Exit items
        fileMenu = wx.Menu()
        openItem = fileMenu.Append(wx.ID_ANY, "Open...", "Open sentence file with the required format...")
        fileMenu.AppendSeparator()
        # When using a stock ID we don't need to specify the menu item's
        # label
        exitItem = fileMenu.Append(wx.ID_EXIT)

        # Now a help menu for the about item
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        # Make the menu bar and add the two menus to it. The '&' defines
        # that the next letter is the "mnemonic" for the menu item. On the
        # platforms that support it those letters are underlined and can be
        # triggered from the keyboard.
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")

        # Give the menu bar to the frame
        self.SetMenuBar(menuBar)

        # Finally, associate a handler function with the EVT_MENU event for
        # each of the menu items. That means that when that menu item is
        # activated then the associated handler function will be called.
        self.Bind(wx.EVT_MENU, self.closeWindow, exitItem)
        self.Bind(wx.EVT_MENU, self.openFile, openItem)
        return menuBar

    def playButton(self, event):
        rec_filename = "{0}.wav".format(self.view_model.getRecordingFilename())
        self.sound_machine.play(rec_filename)

    def stopButton(self, event):
        self.sound_machine.stop()

    def recButton(self, event):
        rec_filename = "{0}.wav".format(self.view_model.getRecordingFilename())
        if os.path.exists(rec_filename):
            os.remove(rec_filename)
        x = threading.Thread(target=self.sound_machine.record, args=(rec_filename, ))
        x.start()

    def openFile(self, event):
        filename = wx.FileSelector("Choose a sentence file to open", wildcard="*.csv")
        self.view_model.loadSentenceFile(filename)
        self.displayCurrentSentence()
        self.SetStatusText("Loaded {0}".format(filename))

    def displayCurrentSentence(self):
        sentence = self.view_model.getCurrentSentence()
        self.sentence_text.SetLabelText(sentence['text'])
        self.sentence_id.SetLabelText("ID: {0}".format(sentence['id']))

    def previousButton(self, event):
        self.view_model.previous()
        self.displayCurrentSentence()

    def nextButton(self, event):
        self.view_model.next()
        self.displayCurrentSentence()

    def closeButton(self, event):
        self.Close(True)

    def closeWindow(self, event):
        self.Destroy()


if __name__ == '__main__':
    app = wx.App()
    frame = AppFrame(parent=None, id=1)
    frame.Show()
    app.MainLoop()
