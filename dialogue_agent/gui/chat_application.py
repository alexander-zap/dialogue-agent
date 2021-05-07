import os
import tkinter as tk

from dialogue_agent.locations import RESOURCES_PATH, GUI_RESOURCES_PATH
from dialogue_agent.user.speech_to_text import SpeechToText

GRAY_COLOR = "#ABB2B9"
DARK_BLUE_COLOR = "#17202A"
WHITE_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class ChatApplication:
    def __init__(self):
        self.window = tk.Tk()
        self._setup_main_window()
        self.wait_state = False
        self.last_message = ""
        self.speech_to_text = SpeechToText(
            model_file_path=os.path.join(RESOURCES_PATH, "deepspeech-model/german_output_graph.pbmm"),
            scorer_file_path=os.path.join(RESOURCES_PATH, "deepspeech-model/german_kenlm.scorer"),
            beam_width=500, lm_alpha=0.75, lm_beta=1.85)

    def _setup_main_window(self):
        self.window.title("Webpage")
        self.window.resizable(width=True, height=True)
        self.window.configure(width=1536, height=864)
        self.window.bind("<Control-f>", self.toggle_fullscreen)
        self.window.bind("<Escape>", self.end_fullscreen)

        background_image = tk.PhotoImage(file=os.path.join(GUI_RESOURCES_PATH, "webpage.png"))
        background_label = tk.Label(master=self.window, image=background_image)
        background_label.photo = background_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # create nested chat frame
        nested_frame = tk.Canvas(master=self.window, highlightbackground="black", highlightthickness=1)
        nested_frame.place(relwidth=0.22, relheight=0.75, relx=0.75, rely=0.2)

        chat_assistant_image_frame = tk.Frame(master=nested_frame)
        chat_assistant_image_frame.place(relwidth=0.98, relheight=0.33, relx=0.01, rely=0.01)
        chat_assistant_image = tk.PhotoImage(file=os.path.join(GUI_RESOURCES_PATH, "chat_assistant.gif"))
        chat_assistant_label = tk.Label(master=chat_assistant_image_frame, image=chat_assistant_image)
        chat_assistant_label.photo = chat_assistant_image
        chat_assistant_label.place(x=0, y=0, relwidth=1, relheight=1)

        chat_frame = tk.Frame(master=nested_frame, bg=WHITE_COLOR)
        chat_frame.place(relwidth=0.98, relheight=0.65, relx=0.01, rely=0.34)

        # head label
        head_label = tk.Label(master=chat_frame, bg=DARK_BLUE_COLOR, fg=WHITE_COLOR, text="Welcome", font=FONT_BOLD)
        head_label.place(relwidth=1, rely=0.01)

        # text widget
        self.text_widget = tk.Text(master=chat_frame, width=20, height=2, bg=DARK_BLUE_COLOR, fg=WHITE_COLOR,
                                   font=FONT,
                                   padx=5, pady=5, wrap=tk.WORD)
        self.text_widget.place(relheight=0.745, relwidth=0.96, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=tk.DISABLED)

        # scroll bar
        scrollbar = tk.Scrollbar(master=chat_frame)
        scrollbar.place(relheight=0.825, relx=0.95)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = tk.Label(master=chat_frame, bg=GRAY_COLOR)
        bottom_label.place(relwidth=1, relheight=2.4, rely=0.825)

        # message entry box
        self.msg_entry = tk.Entry(bottom_label, bg="#2C3E50", fg=WHITE_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.98, relheight=0.06, rely=0.006, relx=0.01)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

    def toggle_fullscreen(self, event):
        self.window.attributes("-fullscreen", True)
        return "break"

    def end_fullscreen(self, event):
        self.window.attributes("-fullscreen", False)
        return "break"

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self.insert_message(msg, "You")

    def insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, tk.END)

        msg1 = f"{sender}: {msg}\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg1)
        self.text_widget.configure(state=tk.DISABLED)

        self.text_widget.see(tk.END)

        self.last_message = msg
        self.wait_state = False

    def wait_for_user_message(self):
        self.wait_state = True
        while self.wait_state:
            self.window.update()
        return self.last_message

    def wait_for_speech_to_text(self):
        self.speech_to_text.start_speech_to_text()

        self.wait_state = True
        while self.wait_state:
            self.window.update()
            self.msg_entry.delete(0, tk.END)
            self.msg_entry.insert(0, self.speech_to_text.transcribed_text)

        self.speech_to_text.reset()

        return self.last_message

    def reset_text_widget(self):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.configure(state=tk.DISABLED)
        self.window.update()
