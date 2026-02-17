import os
from io import BytesIO
import openai                  # for handling error types
from datetime import datetime  # for formatting date returned with images
import base64                  # for decoding images if recieved in the reply
import requests                # for downloading images from URLs
from PIL import Image          # pillow, for processing image types
import tkinter as tk           # for GUI thumbnails of what we got
from PIL import ImageTk        # for GUI thumbnails of what we got
from openai import OpenAI
import storytelling as st

OpenAI.api_key = "YOUR_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
client = OpenAI() 

def generate_image(prompt): 
    image_params = {
    "model": "dall-e-3",  # Defaults to dall-e-2
    "n": 1,               # Between 2 and 10 is only for DALL-E 2
    "size": "1024x1024",  # 256x256, 512x512 only for DALL-E 2 - not much cheaper
    "prompt" :  prompt + "Do not include any text in the image."

      ,     # DALL-E 3: max 4000 characters, DALL-E 2: max 1000
    "user": "myName",     # pass a customer ID to OpenAI for abuse monitoring
    "response_format": "b64_json"
    }

    try:
        images_response = client.images.generate(**image_params)
    except openai.APIConnectionError as e:
        print("Server connection error: {e.__cause__}")  # from httpx.
        raise
    except openai.RateLimitError as e:
        print(f"OpenAI RATE LIMIT error {e.status_code}: (e.response)")
        raise
    except openai.APIStatusError as e:
        print(f"OpenAI STATUS error {e.status_code}: (e.response)")
        raise

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    if images_response.data:
        # Access the first image data object
        image_data_object = images_response.data[0]

        # If image_data_object is a dictionary, use .get() method
        if isinstance(image_data_object, dict):
            image_data = image_data_object.get("b64_json")
        else:
            # If it's an object, access the attribute directly
            image_data = image_data_object.b64_json

        if image_data:
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image_name = 'DALLE.png'
            image.save(image_name)
            print(f"Image saved as {image_name}")
            return image
        else:
            print("No image data obtained")
            return None
    else:
        print("No data found in images_response")
        return None
    


def generate_image_hint(story, question,answer,name): 
    image_params = {
    "model": "dall-e-3",  # Defaults to dall-e-2
    "n": 1,               # Between 2 and 10 is only for DALL-E 2
    "size": "1024x1024",  # 256x256, 512x512 only for DALL-E 2 - not much cheaper
    "prompt": f"""given this question {question}
                generate an image that gives the hint about the answer to the answer
                here's the answer {answer}
                here's the summary of the story {story}
                keep the image and story accurate to eachother
                Do not include any text in the image.
                Image style: Kid's storybook """
      ,     # DALL-E 3: max 4000 characters, DALL-E 2: max 1000
    "user": "myName",     # pass a customer ID to OpenAI for abuse monitoring
    "response_format": "b64_json"
    }

    try:
        images_response = client.images.generate(**image_params)
    except openai.APIConnectionError as e:
        print("Server connection error: {e.__cause__}")  # from httpx.
        raise
    except openai.RateLimitError as e:
        print(f"OpenAI RATE LIMIT error {e.status_code}: (e.response)")
        raise
    except openai.APIStatusError as e:
        print(f"OpenAI STATUS error {e.status_code}: (e.response)")
        raise

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    if images_response.data:
        # Access the first image data object
        image_data_object = images_response.data[0]

        # If image_data_object is a dictionary, use .get() method
        if isinstance(image_data_object, dict):
            image_data = image_data_object.get("b64_json")
        else:
            # If it's an object, access the attribute directly
            image_data = image_data_object.b64_json

        if image_data:
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image_name = name
            image.save(image_name)
            print(f"Image saved as {image_name}")
            return image
        else:
            print("No image data obtained")
            return None
    else:
        print("No data found in images_response")
        return None
    


  
#function to create content prompt to generate imeges out of the story 
def generateImagePrompt(story_segment, age_group_prompt, model= "gpt-3.5-turbo", temperature = 0.7, max_tokens = 1000):
    system_prompt = (f"""Given the following story : 
                     ```
                     {story_segment}
                     ```
                     Generate an image scenario under then 4000 characters, really descriptive of the situation 
                     for the given age group properties: 
                    ```
                     {age_group_prompt}
                    ````
                    keep in mind following reminders while generating the prompt                 
    
                    - be accurate with the details
    
                    -dont add too much complexity
                    
                    -show actions and emotions thought the story
                    
                    -be detailed and descriptive 
                    
                    - if you want a certain number of objects in the image, specify that number in the prompt
                    
                    - provide context or a setting of the environment that the main subject to be in
                    
                    -if possible, include references or examples in your prompts to give clear idea
                    
                    -ensure that prompt aligns with the resolution
                    
                      """)

    response = client.chat.completions.create(
        model=model,
        messages= [
            { "role": "system",
             "content" : system_prompt },
        ],       
        n=3,
        temperature=temperature,
        max_tokens=max_tokens,
    ) 
    return response.choices[0].message.content


def chooseTarget(ageGroup):

    if ageGroup=="Toddlers":
        prompt = """ -give most importance to colors blue, green , red , orange
                        -give less presence to colors red, orange, pink and green 
                        -High Contrast and Simple Images
                        - Simple, recognizable shapes and faces are common.
                        - visual stimulation
                        - image should be things of action” rather than as “objects of contemplation
                        -cartoon style illustration 
                        -imagine you are 1 years old trying to understand the story from the image
                        -make the story understandable by just looking at the image
                        -use imagery of things that they see in the real life
                        -create focus on the main scenario
                        -dont provide too  much distraction
                        -remind the prompt that when generating image based on this prompt in the future steps dont put any text in the image
                        -define the atmosphere of the story , specify at the end 
                        -Use Descriptive Adjectives: Adjectives help in refining the image. For example, instead of saying “a dog,” say “a fluffy, small, brown dog
                        -describe actions or movements. For instance, “a cat jumping over a fence” is more dynamic than just “a cat.”
                        - Short imperative phrases – Quickly convey major elements e.g. “An alien octopus playing drums”. Goes for brevity.
                        – Adds nuance and context e.g. “An enormous blue alien octopus skillfully playing a glowing futuristic drum kit on a rock concert stage.”
                        -highly complex scenes and compositions e.g. describing multiple characters, detailed clothing and expressions, extensive environmental context.
                    """
        
    elif ageGroup== "Preschoolers":
        prompt="""-takes pictures as symbols
            -use symbols( For example; bone symbolizes dog )
            -make use of the representational relation between a picture and the story
            -explain the story in the picture 
            -provide ability to form meta-representatin
            -provide explicit reasoning about pictures as symbols continue to develop throughout the preschool and elementary years
            -provide mapping between word and a picture
            -playful atmosphere 
            -clear depiction of key elements
            -Bright Colors and Clear Shapes
            -Interactive Elements: Pictures that encourage interaction (like flaps or textures) help with motor skills and engagement.
            -use symbols( For example; bone symbolizes dog)
            -Example: taught an unfamiliar label (“whisk”) for a small line drawing of an unfamiliar object (a whisk).
            -remind the prompt that when generating image based on this prompt in the future steps dont put any text in the image
            -define the atmosphere of the story , specify at the end 
            -Use Descriptive Adjectives: Adjectives help in refining the image. For example, instead of saying “a dog,” say “a fluffy, small, brown dog
            -describe actions or movements. For instance, “a cat jumping over a fence” is more dynamic than just “a cat.”
            - Short imperative phrases – Quickly convey major elements e.g. “An alien octopus playing drums”. Goes for brevity.
            – Adds nuance and context e.g. “An enormous blue alien octopus skillfully playing a glowing futuristic drum kit on a rock concert stage.”
            -highly complex scenes and compositions e.g. describing multiple characters, detailed clothing and expressions, extensive environmental context.

            """
            
        
    elif ageGroup== "Early Elementary":
        prompt =  """ provide story in the picture to help them understand the story
            -The images often depict actions or events mentioned in the text.
            -provide subtext messages mentioned in the story 
            -provide the humor in the pictures, related to the story
            -give most importance to colors blue, green , red , orange
            -give less presence to colors red, orange, pink and green 
            -avoid image or context repetition
            -avoid the use of complex symbolism
            -remind the prompt that when generating image based on this prompt in the future steps dont put any text in the image
            
            """

    elif ageGroup== "Preteens":
        prompt=""" 
        -extract the key elements of the story and concentrate on that
        -for example if a process is being explained; explain the process in the image prompt and tell a story with just seeing the image
        -give most importance to colors blue, green , red , orange
        -give less presence to colors red, orange, pink and green 
        -remind the prompt that when generating image based on this prompt in the future steps dont put any text in the image
        -stay away from the too abstract descriptions
        -dont overwhelm with the details 
        -include relatable elements like , a child observing, or joining into the action with the storyline
        -decrease the complexity of the background
        -make the focal point the main character of the storyline
        -include visual cues; symbols, arrows . For example if a photosynthesis being explained; put arrows showing the directions of the sunlight , water and air or illustrate the release of the oxygen

        """

    elif ageGroup== "Late Elementary":
        prompt= """ 
        -extract the key elements of the story and concentrate on that
        -for example if a process is being explained; explain the process in the image prompt and tell a story with just seeing the image
        -give most importance to colors blue, green , red , orange
        -give less presence to colors red, orange, pink and green 
        -remind the prompt that when generating image based on this prompt in the future steps dont put any text in the image
        -stay away from the too abstract descriptions
        -dont overwhelm with the details 
        -include relatable elements like , a child observing, or joining into the action with the storyline
        -decrease the complexity of the background
        -make the focal point the main character of the storyline
        -include visual cues; symbols, arrows . For example if a photosynthesis being explained; put arrows showing the directions of the sunlight , water and air or illustrate the release of the oxygen
        """

    else:
        raise ValueError("choose target error")

    return prompt

def generate_image_begin(story_segment, age_group):     
    prompt91= generateImagePrompt(story_segment, chooseTarget(age_group))
    print("Image generation prompt" + prompt91)
    generate_image(prompt91)
    print("image is generated and saved under the name 'DALLE' ")

