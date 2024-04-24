import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Function to make an api call and get the response from the model
def get_model_response(user_input):
    # Make an API call to the chat endpoint
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="llama3-70b-8192",
    )

    # Return the response from the model
    return chat_completion.choices[0].message.content

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