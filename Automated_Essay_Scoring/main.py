from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from models import predict_score, assess_creativity_and_similarity
import json
import math
import random

app = FastAPI()

class EssayInput(BaseModel):
    essay_content: str

@app.post("/")
def predict_essay_score(essay_input: EssayInput):
    essay_content = essay_input.essay_content
    predicted_score = predict_score(essay_content)                     # Predict the score

    def load_messages(filename):
        with open(filename, 'r') as file:
            messages = json.load(file)
        return messages

    # Define the message file path
    message_file = "feedback.json"

    file_path = './Data/reference.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        reference_essay = file.read()

    if predicted_score > 10:
        predicted_score = 10
    elif predicted_score< 0:
        predicted_score = 0
        
    creativity_score, similarity_score = assess_creativity_and_similarity(essay_content, reference_essay)
    formatted_creativity_score = round(creativity_score * 10, 1)
    formatted_similarity_score = round(similarity_score * 10, 1)
    total_score = predicted_score + formatted_creativity_score - formatted_similarity_score
    rounded_total_score = round(total_score)

    messages = load_messages(message_file)

    # Update the score range conditions
    if 0 <= rounded_total_score <= 4:
        message = random.choice(messages["0-4"])
    elif 5 <= rounded_total_score <= 7:
        message = random.choice(messages["5-7"])
    elif 8 <= rounded_total_score <= 11:
        message = random.choice(messages["8-11"])
    elif 12 <= rounded_total_score <= 15:
        message = random.choice(messages["12-15"])
    else:
        message = "Score is out of range."

    print(f"TOTAL SCORE = {int(math.ceil(rounded_total_score))} / 15")
    print("FEEDBACK : ", message)

    return {
    "predicted_essay_score": f"{predicted_score} / 10",
    "creativity_score": f"{formatted_creativity_score} / 5",
    "similarity_score": f"{formatted_similarity_score} / 10",
    "total_score": f"{int(math.ceil(rounded_total_score))} / 15",
    "feedback": message
    }


if __name__ == "__main__":
    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)
