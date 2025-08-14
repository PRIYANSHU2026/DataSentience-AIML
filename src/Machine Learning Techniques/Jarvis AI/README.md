# 🤖 Jarvis – AI Desktop Voice Assistant

Jarvis is an AI-powered voice assistant designed for desktops, capable of executing voice commands to launch apps, perform web searches, control the system, and even generate AI-based content. The assistant features a modern PyQt5 GUI with animated visuals, voice feedback, and seamless AI integration using the Groq API.

---

![Jarvis GIF](assets/Jarvis.gif)

## 🎯 Features

- 🎙️ Real-time voice recognition and response
- 💻 Open and close desktop applications via voice
- 🔍 Voice-based Google and YouTube searches
- ✍️ AI-generated content using the Groq API
- 🪟 Custom PyQt5 GUI with frameless design and animation
- ⚙️ System control functions (shutdown, restart, etc.)
- 🔄 Multithreading for smooth GUI–backend communication

---

## 🧠 Tech Stack

| Technology        | Purpose                             |
|-------------------|-------------------------------------|
| Python            | Core programming language           |
| PyQt5             | GUI design and integration          |
| SpeechRecognition | Voice input processing              |
| gTTS / pyttsx3    | Text-to-speech voice output         |
| Groq API          | AI text generation                  |
| Subprocess        | App and system control              |
| Multithreading    | GUI performance and non-blocking UX |

---
## 🖼️ Screenshots

> *(Optional – add images in the `/assets` folder and update the links below)*

| GUI Interface | Voice Command in Action |
|---------------|------------------------|
| ![GUI Interface](https://github.com/user-attachments/assets/291e54ca-35d3-46ac-a828-d1d192c8ebe1) | ![Voice Command](https://github.com/user-attachments/assets/de0ac056-0f05-4081-93ee-5b7a3a27ca12) |

---



All tasks are executed through **natural speech**, providing a real-time interactive assistant experience.

---

## 💡 Motivation
With the rise of AI-powered tools, voice assistants enhance accessibility and convenience in daily tasks. AI-JARVIS aims to **bring a flexible, real-time assistant** to users, similar to a human helper.

---

## 🧰 Tech Stack
- **Programming Language:** Python 3.9+  
- **Libraries Used:**  
  `pyttsx3`, `speech_recognition`, `wikipedia`, `webbrowser`, `smtplib`, `PyQt5`, `geopy`, `bs4`, `yahoo_fin`, etc.  
- **Framework/Tools:**  
  - PyQt5 for GUI  
  - SpeechRecognition & pyttsx3 for voice interface  
  - Custom Python modules for assistant functionality  

---

## 🧠 Features
AI-JARVIS currently supports **19+ voice-controlled features**, including:  

- Weather and temperature updates  
- Media control (e.g., play a random song)  
- Open applications or websites  
- Wikipedia & Google search  
- Email and location services  
- Stock market information  
- Match score prediction  
- Jokes, news updates, and more  

---

## 🏁 Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/repo-name.git
cd src/Miscellaneous/Jarvis AI
```
## Install Dependencies
```bash
pip install -r requirements.txt
```

 ## Run the Assistant
 ```bash
 python Main.py
```

## ✅ Folder Structure
```bash 
AI-JARVIS/
├── Jarvis_GUI.py
├── Jarvis_Core.py
├── requirements.txt
├── README.md
└── assets/
    ├── Jarvis.gif
    └── icons/
```





 ## 🛠 Contribution Guidelines
- New contributors can enhance features, optimize the GUI, add unit tests, or improve voice recognition.
- Bug reports, UI improvements, and feature requests are welcome.

## 🔮 Future Scope
- ML/DL-based personalization

- Integration with AR or smart home devices

- Multi-language support

- Gesture & chatbot integration









