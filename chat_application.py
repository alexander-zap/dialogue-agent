import tkinter as tk

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

    def _setup_main_window(self):
        self.window.title("Dialogue Agent")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=600, bg=DARK_BLUE_COLOR)

        # head label
        head_label = tk.Label(master=self.window, bg=DARK_BLUE_COLOR, fg=WHITE_COLOR, text="Welcome", font=FONT_BOLD,
                              pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = tk.Label(master=self.window, width=450, bg=GRAY_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = tk.Text(master=self.window, width=20, height=2, bg=DARK_BLUE_COLOR, fg=WHITE_COLOR,
                                   font=FONT,
                                   padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=tk.DISABLED)

        # scroll bar
        scrollbar = tk.Scrollbar(master=self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = tk.Label(master=self.window, bg=GRAY_COLOR, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = tk.Entry(bottom_label, bg="#2C3E50", fg=WHITE_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = tk.Button(master=bottom_label, text="Send", font=FONT_BOLD, width=20, bg=GRAY_COLOR,
                                command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

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
