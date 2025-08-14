# Corrected version of your integrated assistant backend and frontend launch

import sys
import os
import json
import subprocess
import threading
import asyncio
from time import sleep
from dotenv import dotenv_values

# Import GUI functions
from Frontend.GUI import (
    GraphicalUserInterface,
    SetAssistantStatus,
    ShowTextToScreen,
    TempDirectoryPath,
    SetMicrophoneStatus,
    AnswerModifier,
    QueryModifier,
    GetMicrophoneStatus,
    GetAssistantStatus
)

# Import backend functions
from Backend.Model import FirstLayerDMM
from Backend.RealtimeSearchEngine import RealtimeSearchEngine
from Backend.Automation import Automation
from Backend.SpeechToText import SpeechRecognition
from Backend.Chatbot import ChatBot
from Backend.TextToSpeech import TextToSpeech

# Load environment variables
env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")

# Default greeting message
DefaultMessage = f"""
{Username} : Hello {Assistantname}, How are you?
{Assistantname} : Welcome {Username}. I am doing well. How may I help you?
"""

Functions = ["open", "close", "play", "system", "content", "google search", "youtube search"]

# Show default greeting if no chats exist
def ShowDefaultChatIfNoChats():
    try:
        with open(r'Data/ChatLog.json', 'r', encoding='utf-8') as file:
            if len(file.read()) < 5:
                with open(TempDirectoryPath("Database.data"), 'w', encoding='utf-8') as db_file:
                    db_file.write("")
                with open(TempDirectoryPath("Responses.data"), 'w', encoding='utf-8') as res_file:
                    res_file.write(DefaultMessage)
    except:
        pass

# Read ChatLog.json
def ReadChatLogJson():
    with open(r'Data/ChatLog.json', 'r', encoding='utf-8') as file:
        return json.load(file)

# Format chat history into readable form
def ChatLogIntegration():
    json_data = ReadChatLogJson()
    formatted_chatlog = ""
    for entry in json_data:
        if entry['role'] == "user":
            formatted_chatlog += f"User: {entry['content']}\n"
        elif entry['role'] == "assistant":
            formatted_chatlog += f"Assistant: {entry['content']}\n"
    formatted_chatlog = formatted_chatlog.replace("User", Username + " :")
    formatted_chatlog = formatted_chatlog.replace("Assistant", Assistantname + " :")
    with open(TempDirectoryPath("Database.data"), 'w', encoding='utf-8') as file:
        file.write(AnswerModifier(formatted_chatlog))

# Display chat on GUI

def ShowChatsOnGUI():
    try:
        with open(TempDirectoryPath("Database.data"), 'r', encoding='utf-8') as file:
            data = file.read()
        if data:
            lines = data.split('\n')
            result = '\n'.join(lines)
            with open(TempDirectoryPath("Responses.data"), 'w', encoding='utf-8') as file:
                file.write(result)
    except:
        pass

# Run initial chat setup
def InitialExecution():
    SetMicrophoneStatus("False")
    ShowTextToScreen("")
    ShowDefaultChatIfNoChats()
    ChatLogIntegration()
    ShowChatsOnGUI()

# Main execution logic
def MainExecution():
    TaskExecution = False
    ImageExecution = False
    ImageGenerationQuery = ""

    SetAssistantStatus("Listening...")
    Query = SpeechRecognition()
    ShowTextToScreen(f"{Username} : {Query}")
    SetAssistantStatus("Thinking...")
    Decision = FirstLayerDMM(Query)

    G = any(i.startswith("general") for i in Decision)
    R = any(i.startswith("realtime") for i in Decision)

    MergedQuery = " and ".join(
        [" ".join(i.split()[1:]) for i in Decision if i.startswith("general") or i.startswith("realtime")]
    )

    for query in Decision:
        if "generate" in query:
            ImageGenerationQuery = query
            ImageExecution = True

    for query in Decision:
        if not TaskExecution and any(query.startswith(func) for func in Functions):
            asyncio.run(Automation(Decision))
            TaskExecution = True

    if ImageExecution:
        with open(TempDirectoryPath("ImageGeneration.data"), "w") as file:
            file.write(f"{ImageGenerationQuery}, True")
        try:
            subprocess.Popen(['python', r'Backend\ImageGeneration.py'], shell=False)
        except Exception as e:
            print(f"Error launching ImageGeneration.py: {e}")

    if (G and R) or R:
        SetAssistantStatus("Searching...")
        Answer = RealtimeSearchEngine(QueryModifier(MergedQuery))
        ShowTextToScreen(f"{Assistantname} : {Answer}")
        SetAssistantStatus("Answering...")
        TextToSpeech(Answer)
        return True

    for query in Decision:
        if "general" in query:
            SetAssistantStatus("Thinking...")
            FinalQuery = query.replace("general ", "")
            Answer = ChatBot(QueryModifier(FinalQuery))
            ShowTextToScreen(f"{Assistantname} : {Answer}")
            SetAssistantStatus("Answering...")
            TextToSpeech(Answer)
            return True

        elif "realtime" in query:
            SetAssistantStatus("Searching...")
            FinalQuery = query.replace("realtime ", "")
            Answer = RealtimeSearchEngine(QueryModifier(FinalQuery))
            ShowTextToScreen(f"{Assistantname} : {Answer}")
            SetAssistantStatus("Answering...")
            TextToSpeech(Answer)
            return True

        elif "exit" in query:
            Answer = ChatBot("Okay, Bye!")
            ShowTextToScreen(f"{Assistantname} : {Answer}")
            SetAssistantStatus("Answering...")
            TextToSpeech(Answer)
            SetAssistantStatus("Shutting down")
            os._exit(1)

# Thread for continuous mic status check
def FirstThread():
    while True:
        current_status = GetMicrophoneStatus()
        if current_status == "True":
            MainExecution()
        else:
            if "Available" in GetAssistantStatus():
                sleep(0.1)
            else:
                SetAssistantStatus("Available...")

# GUI thread
def SecondThread():
    GraphicalUserInterface()

# Entry point
if __name__ == "__main__":
    thread2 = threading.Thread(target=FirstThread, daemon=True)
    thread2.start()
    SecondThread()