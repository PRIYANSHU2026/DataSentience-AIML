# Importing necessary libraries
import sys
import os
from dotenv import dotenv_values
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTextEdit, QVBoxLayout, QLabel, QPushButton,
                             QHBoxLayout, QStackedWidget, QSizePolicy, QFrame, QScrollBar)
from PyQt5.QtGui import QIcon, QPainter, QMovie, QFont, QTextCharFormat, QTextBlockFormat, QColor, QPixmap
from PyQt5.QtCore import Qt, QSize, QTimer

# Load environment variables
env_vars = dotenv_values(".env")
Assistantname = env_vars.get("Assistantname")

# Define paths
current_dir = os.getcwd()
TempDirPath = rf"{current_dir}\Frontend\Files"
GraphicsDirPath = rf"{current_dir}\Frontend\Graphics"

# Utility Functions
def AnswerModifier(answer):
    lines = answer.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def QueryModifier(query):
    new_query = query.lower().strip()
    if not new_query.endswith("?"):
        new_query += "?"
    return new_query.capitalize()

def SetMicrophoneStatus(command):
    with open(rf"{TempDirPath}\Mic.data", "w", encoding="utf-8") as file:
        file.write(command)

def GetMicrophoneStatus():
    with open(rf"{TempDirPath}\Mic.data", "r", encoding="utf-8") as file:
        return file.read()

def SetAssistantStatus(status):
    with open(rf"{TempDirPath}\Status.data", "w", encoding="utf-8") as file:
        file.write(status)

def GetAssistantStatus():
    with open(rf"{TempDirPath}\Status.data", "r", encoding="utf-8") as file:
        return file.read()

def GraphicsDirectoryPath(filename):
    return rf"{GraphicsDirPath}\{filename}"

def TempDirectoryPath(filename):
    return rf"{TempDirPath}\{filename}"

def ShowTextToScreen(text):
    with open(rf"{TempDirPath}\Responses.data", "w", encoding='utf-8') as file:
        file.write(text)

class ChatSection(QWidget):
    def __init__(self):
        super(ChatSection, self).__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 40, 40, 100)
        layout.setSpacing(10)

        self.chat_text_edit = QTextEdit()
        self.chat_text_edit.setReadOnly(True)
        self.chat_text_edit.setTextInteractionFlags(Qt.NoTextInteraction)
        self.chat_text_edit.setFrameStyle(QFrame.NoFrame)
        self.chat_text_edit.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.chat_text_edit)

        self.gif_label = QLabel()
        self.gif_label.setStyleSheet("border: none;")
        movie = QMovie(GraphicsDirectoryPath('Jarvis.gif'))
        movie.setScaledSize(QSize(500, 500))
        self.gif_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.gif_label.setMovie(movie)
        movie.start()
        layout.addWidget(self.gif_label)

        self.label = QLabel("Initializing...")
        self.label.setStyleSheet("color: white; font-size: 16px; margin-right: 195px;")
        self.label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label)

        font = QFont()
        font.setPointSize(13)
        self.chat_text_edit.setFont(font)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loadMessages)
        self.timer.timeout.connect(self.SpeechRecogText)
        self.timer.start(500)

        self.chat_text_edit.viewport().installEventFilter(self)

    def loadMessages(self):
        try:
            with open(TempDirectoryPath("Responses.data"), "r", encoding="utf-8") as file:
                messages = file.read()
                if messages:
                    self.addMessage(messages, color="white")
        except Exception as e:
            pass

    def SpeechRecogText(self):
        try:
            with open(TempDirectoryPath("Status.data"), "r", encoding="utf-8") as file:
                messages = file.read()
                self.label.setText(messages)
        except Exception as e:
            pass

    def addMessage(self, message, color):
        cursor = self.chat_text_edit.textCursor()
        format = QTextCharFormat()
        formatm = QTextBlockFormat()
        formatm.setTopMargin(10)
        formatm.setLeftMargin(10)
        format.setForeground(QColor(color))
        cursor.setCharFormat(format)
        cursor.setBlockFormat(formatm)
        cursor.insertText(message + "\n")
        self.chat_text_edit.setTextCursor(cursor)

class InitialScreen(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 150)

        gif_label = QLabel()
        movie = QMovie(GraphicsDirectoryPath("Jarvis.gif"))
        screen_width = QApplication.desktop().screenGeometry().width()
        movie.setScaledSize(QSize(screen_width, 560))
        gif_label.setAlignment(Qt.AlignCenter)
        gif_label.setMovie(movie)
        movie.start()

        self.label = QLabel("Hello, I am your assistant")
        self.label.setStyleSheet("color: white; font-size: 16px; margin-bottom: 0;")
        self.label.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel()
        pixmap = QPixmap(GraphicsDirectoryPath("Mic_on.png"))
        new_pixmap = pixmap.scaled(60, 60)
        self.icon_label.setPixmap(new_pixmap)
        self.icon_label.setFixedSize(150, 150)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.toggled = True
        self.icon_label.mousePressEvent = self.toggle_icon

        layout.addWidget(gif_label)
        layout.addWidget(self.label)
        layout.addWidget(self.icon_label)

        self.setLayout(layout)
        self.setStyleSheet("background-color: black;")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.SpeechRecogText)
        self.timer.start(500)

    def SpeechRecogText(self):
        try:
            with open(TempDirectoryPath("Status.data"), "r", encoding="utf-8") as file:
                messages = file.read()
                self.label.setText(messages)
        except Exception as e:
            pass

    def toggle_icon(self, event=None):
        if self.toggled:
            pixmap = QPixmap(GraphicsDirectoryPath("Mic_off.png")).scaled(60, 60)
            SetMicrophoneStatus("False")
        else:
            pixmap = QPixmap(GraphicsDirectoryPath("Mic_on.png")).scaled(60, 60)
            SetMicrophoneStatus("True")
        self.icon_label.setPixmap(pixmap)
        self.toggled = not self.toggled

class MessageScreen(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Live Status")
        self.label.setStyleSheet("color: white;")
        layout.addWidget(self.label)
        self.chat_section = ChatSection()
        layout.addWidget(self.chat_section)
        self.setLayout(layout)
        self.setStyleSheet("background-color: black;")

class CustomTopBar(QWidget):
    def __init__(self, parent, stacked_widget):
        super().__init__(parent)
        self.parent = parent
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignRight)

        title_label = QLabel(f"{Assistantname} AI")
        title_label.setStyleSheet("color: black; font-weight: bold;")

        home_button = QPushButton("Home")
        home_button.setIcon(QIcon(GraphicsDirectoryPath("Home.png")))
        home_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        chat_button = QPushButton("Chat")
        chat_button.setIcon(QIcon(GraphicsDirectoryPath("Chats.png")))
        chat_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        # Minimize Button
        minimize_button = QPushButton()
        minimize_icon = QIcon(GraphicsDirectoryPath("Minimize2.png"))  # Make sure this file exists
        minimize_button.setIcon(minimize_icon)
        minimize_button.setStyleSheet("background-color: white;")
        minimize_button.clicked.connect(self.minimizeWindow)

        # Maximize Button
        self.maximize_button = QPushButton()
        self.maximize_icon = QIcon(GraphicsDirectoryPath("Maximize.png"))  # Use a separate icon for restore if you like
        self.restore_icon = QIcon(GraphicsDirectoryPath("Restore.png"))    # Optional
        self.maximize_button.setIcon(self.maximize_icon)
        self.maximize_button.setStyleSheet("background-color: white;")
        self.maximize_button.clicked.connect(self.maximizeWindow)

        close_button = QPushButton("Close")
        close_button.setIcon(QIcon(GraphicsDirectoryPath("Close.png")))
        close_button.clicked.connect(self.parent.close)
        
        for btn in [home_button, chat_button, close_button]:
            btn.setStyleSheet("background-color: white; color: black; height: 30px;")

        layout.addWidget(title_label)
        layout.addStretch(1)
        layout.addWidget(home_button)
        layout.addWidget(chat_button)
        layout.addWidget(minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(close_button)


    def minimizeWindow(self):
         self.parent.showMinimized()

    def maximizeWindow(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_button.setIcon(self.maximize_icon)
        else:
            self.parent.showMaximized()
            self.maximize_button.setIcon(self.restore_icon)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.initUI()

    def initUI(self):
        screen = QApplication.desktop().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height())

        self.stacked_widget = QStackedWidget()
        self.initial_screen = InitialScreen()
        self.message_screen = MessageScreen()

        self.stacked_widget.addWidget(self.initial_screen)
        self.stacked_widget.addWidget(self.message_screen)

        self.setCentralWidget(self.stacked_widget)

        top_bar = CustomTopBar(self, self.stacked_widget)
        self.setMenuWidget(top_bar)
        self.setStyleSheet("background-color: black;")

def GraphicalUserInterface():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    GraphicalUserInterface()
