from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values

# Load environment variables from the .env file.
env_vars = dotenv_values(".env")

# Retrieve values
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

# Initialize Groq client
client = Groq(api_key=GroqAPIKey)

# Message history
messages = []

# System instruction
System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""

SystemChatBot = [{"role": "system", "content": System}]

# Load existing chat log
try:
    with open(r"ChatLog.json", "r") as f:
     messages = load(f)
except FileNotFoundError:
    with open(r"ChatLog.json", "w") as f:
        dump([], f)

# Real-time info function
def RealtimeInformation():
    now = datetime.datetime.now()
    return (
        f"Please use this real-time information if needed,\n"
        f"Day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d')}\n"
        f"Month: {now.strftime('%B')}\n"
        f"Year: {now.strftime('%Y')}\n"
        f"Time: {now.strftime('%H')} hours :{now.strftime('%M')} minutes :{now.strftime('%S')} seconds.\n"
    )

# Answer formatter
def AnswerModifier(Answer):
    lines = Answer.split('\n')
    return '\n'.join([line for line in lines if line.strip()])

# Chatbot logic
def ChatBot(Query):
    """This function sends the user's query to the chatbot and returns the AI's response."""
    try:
        with open(r"Data\ChatLog.json", "r") as f:
            messages = load(f)

        messages.append({"role": "user", "content": Query})

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        Answer = Answer.replace("</s>", "")
        messages.append({"role": "assistant", "content": Answer})

        with open(r"Data\ChatLog.json", "w") as f:
            dump(messages, f, indent=4)

        return AnswerModifier(Answer)

    except Exception as e:
        print(f"Error: {e}")
        with open(r"Data\ChatLog.json", "w") as f:
            dump([], f, indent=4)
        return ChatBot(Query)

# Main program loop
if __name__ == "__main__":
    while True:
        user_input = input("Enter Your Question: ")
        print(ChatBot(user_input))
