import os
from groq import Groq
import re

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Class to define colors for terminal output
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# Function to make an api call and get the response from the model
def get_model_response(user_input, memory=None, user_data=None):
    # Make an API call to the chat endpoint
    messages = []
    
    # Include past conversation history if it exists
    if memory:
        for entry in memory:
            messages.append({
                "role": entry['role'],  # 'user' or 'assistant'
                "content": entry['content']
            })
    
    if user_data:
        messages.append({
            "role": "system",
            "content": f"You are talking to a person of {user_data['age']} years old, is a person who likes the following topics: {user_data['likes']} and prefers {user_data['learning_preference']}% of theory and {100 - user_data['learning_preference']}% of examples and practice in the explanations. Take into account this information to provide better explanations."
        })
    # Append the current user input to the message history
    messages.append({
        "role": "user",
        "content": user_input,
    })
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
    )

    # Return the response from the model
    return chat_completion.choices[0].message.content

# Function to format the response and replace **text** with bold text
def format_response(response):
    # This regular expression matches pairs of '**' surrounding any text
    #bold_pattern = re.compile(r'\*\*(.*?)\*\*')

    # Replace '**text**' with ANSI codes to make 'text' bold
    #response = bold_pattern.sub(r'{}{}\{}'.format(color.BOLD, r'\1', color.END), response)

    #response = bold_pattern.sub(r'{}{}\{}'.format("", r'\1', ""), response)
    
    response = response.replace("**", "")
    
    return response


if __name__ == "__main__":

    # Example of usage:
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="mixtral-8x7b-32768",
    )
    
    print(chat_completion.choices[0].message.content)

    # Example string to test formatting
    test_string = "This is a **bold** statement and here's **another one** to test."
    
    # Format the response
    formatted_response = format_response(test_string)
    
    # Print the formatted response to see the effect in the terminal
    print(formatted_response)
