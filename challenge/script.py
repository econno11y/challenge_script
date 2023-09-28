import json
import os

import replicate
import requests
from dotenv import load_dotenv

load_dotenv()
NASA_API_KEY = os.environ.get("NASA_API_KEY")
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL")


def main():
    image_json = get_nasa_image()
    llm_caption = get_llm_caption(image_json.get("url"))

    dictionary = {
        "title": image_json.get("title") or "No title provided",
        "explanation": image_json.get("explanation") or "No explanation provided",
        "url": image_json.get("url") or "No url provided",
        "image2prompt_output": llm_caption.strip().capitalize(),
    }

    return json.dumps(dictionary, indent=4)


def get_nasa_image():
    r = requests.get(f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}")
    if r.status_code != 200:
        print(r.status_code)
        raise Exception(f"Error fetching image from NASA API: {r.text}")
    return r.json()


def get_llm_caption(url, model=AI_MODEL):
    if not url:
        raise Exception("No image URL provided")


    print("....time to multi-task....this may take a few minutes...")
    try:
        client = replicate.Client(api_token=REPLICATE_API_KEY)
        output = client.run(
            model,
            input={"image": url, "prefix": "This is a picture of "},
        )
        return output
    except Exception as e:
        return f"Error processing image: {e}"


if __name__ == "__main__":
    print(main())
