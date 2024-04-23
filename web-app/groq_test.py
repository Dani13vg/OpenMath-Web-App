# run following command on terminal to set the api key
# export GROQ_API_KEY=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

import os
import random
from groq import Groq

age = input("input your age: ")  #Esto se guarda al perfil al registrarse
liked_activities = input("input your liked activities: ") 
#this is a list of liked activities, #Esto se guarda al perfil al registrarse como una lista	

#turn string into list
liked_activities = liked_activities.split(" ")

area = input("input the area you want to understand: ")  #segun donde pulsa
concept =  input("input the concept you want to understand: ") #

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("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"),
)

liked_activity = random.choice(liked_activities)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"explain for a {age} year old child that likes {liked_activity} in a way that they can understand, act like a friend, not a teacher, dont be superb and let the child ask questions and be curious."
        },
        {
            "role": "user",
            "content": f"Explain this concept: {concept} of {area} to me in a way that I can understand.",
        }
    ],
    model="llama3-70b-8192",
)

print(chat_completion.choices[0].message.content)