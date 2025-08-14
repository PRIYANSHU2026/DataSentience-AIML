import asyncio
from random import randint
from PIL import Image
import requests
from dotenv import get_key
import os
from time import sleep

# API config
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
API_KEY = get_key(".env", "HuggingFaceAPIKey")
headers = {"Authorization": f"Bearer {API_KEY}"}


# Function to open and display generated images
def open_images(prompt):
    folder_path = "Data"
    prompt_safe = prompt.replace(" ", "_")
    files = [f"{prompt_safe}_{i}.jpg" for i in range(1, 5)]
    
    for file in files:
        image_path = os.path.join(folder_path, file)
        try:
            img = Image.open(image_path)
            print(f"🖼️ Opening: {image_path}")
            img.show()
            sleep(1)
        except IOError:
            print(f"❌ Could not open image: {image_path}")


# Async function to query Hugging Face API
async def query(payload):
    response = await asyncio.to_thread(requests.post, API_URL, headers=headers, json=payload)
    return response.content


# Async function to generate and save images
async def generate_images(prompt: str):
    prompt_safe = prompt.replace(" ", "_")
    os.makedirs("Data", exist_ok=True)
    tasks = []

    for _ in range(4):
        payload = {
            "inputs": f"{prompt}, quality=4K, ultra high detail, high resolution, seed={randint(0, 1000000)}"
        }
        task = asyncio.create_task(query(payload))
        tasks.append(task)

    images = await asyncio.gather(*tasks)

    for i, image_data in enumerate(images):
        filename = f"Data/{prompt_safe}_{i+1}.jpg"
        with open(filename, "wb") as f:
            f.write(image_data)


# Combined wrapper
def GenerateImages(prompt: str):
    asyncio.run(generate_images(prompt))
    open_images(prompt)


# Monitor loop for external image generation trigger
while True:
    try:
        with open("Frontend/Files/ImageGeneration.data", "r") as f:
            data = f.read().strip()
            if not data:
                sleep(1)
                continue

            prompt, status = map(str.strip, data.split(","))

        if status.lower() == "true":
            print("🎨 Generating images for prompt:", prompt)
            GenerateImages(prompt)

            # Reset status
            with open("Frontend/Files/ImageGeneration.data", "w") as f:
                f.write("False,False")
            break
        else:
            sleep(1)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        sleep(2)
