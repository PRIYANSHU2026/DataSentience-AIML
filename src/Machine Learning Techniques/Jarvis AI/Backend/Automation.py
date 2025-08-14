from AppOpener import close, open as appopen
from webbrowser import open as webopen
from pywhatkit import search, playonyt
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from rich import print
from groq import Groq
import webbrowser
import subprocess
import requests
import keyboard
import asyncio
import os

# Load environment variables
env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")

# Initialize Groq client
client = Groq(api_key=GroqAPIKey)

# System message to provide context to the chatbot
SystemChatBot = [{"role": "system", "content": f"Hello, I am {os.environ.get('Username', 'User')}, You're a content writer."}]

# HTML parser classes for scraping
classes = [
    "zCubwf", "IZ6rdc", "LwkfKe", "hgkElc", "LTKOO", "sY7ric", "ZOLCW",
    "gst vk_bk FzvwSb YwPhnf", "pclqee", "tw-Data-text tw-text-small tw-ta",
    "05uR6d LTKO0", "VLzY6d", "VQF4g", "qv3Wpe", "webanswers-webanswers_table_webanswers-table",
    "dDoNo ikb4Bb gsrt", "sLaDe", "kno-rdesc", "SPZz6b"
]

# User-Agent
useragent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
)

# Function to perform Google search
def GoogleSearch(Topic):
    search(Topic)
    return True

# Function to open Notepad
def OpenNotepad(File):
    default_text_editor = 'notepad.exe'
    subprocess.Popen([default_text_editor, File])

# Function to generate content using AI and save it to a file
def Content(Topic):
    def ContentWriterAI(prompt):
        messages = SystemChatBot.copy()
        messages.append({"role": "user", "content": prompt})
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            stream=True
        )
        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content
        Answer = Answer.replace("</s>", "")
        messages.append({"role": "assistant", "content": Answer})
        return Answer

    Topic = Topic.replace("content", "")
    ContentByAI = ContentWriterAI(Topic)
    filepath = rf"Data\{Topic.lower().replace(' ', '')}.txt"
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(ContentByAI)
    OpenNotepad(filepath)
    return True

# Function to search YouTube
def YouTubeSearch(Topic):
    url = f"https://www.youtube.com/results?search_query={Topic}"
    webbrowser.open(url)
    return True

# Function to play video on YouTube
def PlayYoutube(query):
    playonyt(query)
    return True

# Function to open an application
def OpenApp(app, sess=requests.session()):
    try:
        appopen(app, match_closest=True, output=True, throw_error=True)
        return True
    except:
        return False

# Function to extract links from HTML
def extract_links(html):
    if html is None:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a')
    return [link.get('href') for link in links]

# Function to search and open the first link
def search_google(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": useragent}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print("Failed to retrieve search results.")
        return None

# Function to close application
def CloseApp(app):
    if "chrome" in app:
        pass
    else:
        try:
            close(app, match_closest=True, output=True, throw_error=True)
            return True
        except:
            return False

# System control functions
def System(command):
    def mute():
        keyboard.press_and_release("volume mute")

    def unmute():
        keyboard.press_and_release("volume mute")

    def volume_up():
        keyboard.press_and_release("volume up")

    def volume_down():
        keyboard.press_and_release("volume down")

    if command == "mute":
        mute()
    elif command == "unmute":
        unmute()
    elif command == "volume up":
        volume_up()
    elif command == "volume down":
        volume_down()
    return True

# Asynchronous command translator
async def TranslateAndExecute(commands: list[str]):
    funcs = []
    for command in commands:
        if command.startswith("open ") and "open it" not in command and "open file" not in command:
            fun = asyncio.to_thread(OpenApp, command.removeprefix("open "))
            funcs.append(fun)
        elif command.startswith("close "):
            fun = asyncio.to_thread(CloseApp, command.removeprefix("close "))
            funcs.append(fun)
        elif command.startswith("play "):
            fun = asyncio.to_thread(PlayYoutube, command.removeprefix("play "))
            funcs.append(fun)
        elif command.startswith("content "):
            fun = asyncio.to_thread(Content, command.removeprefix("content "))
            funcs.append(fun)
        elif command.startswith("google search "):
            fun = asyncio.to_thread(GoogleSearch, command.removeprefix("google search "))
            funcs.append(fun)
        elif command.startswith("youtube search "):
            fun = asyncio.to_thread(YouTubeSearch, command.removeprefix("youtube search "))
            funcs.append(fun)
        elif command.startswith("system "):
            fun = asyncio.to_thread(System, command.removeprefix("system "))
            funcs.append(fun)
        else:
            print(f"No function found for '{command}'")
    results = await asyncio.gather(*funcs)
    for result in results:
        if isinstance(result, str):
            yield result
        else:
            yield result

# Asynchronous automation entry point
async def Automation(commands: list[str]):
    async for result in TranslateAndExecute(commands):
        pass
    return True

if __name__ == "__main__":
    user_commands = input("🧠 Enter your commands separated by commas:\n➡ ").split(",")
    user_commands = [cmd.strip() for cmd in user_commands if cmd.strip()]
    
    print(f"\n🚀 Executing: {user_commands}\n")
    asyncio.run(Automation(user_commands))
