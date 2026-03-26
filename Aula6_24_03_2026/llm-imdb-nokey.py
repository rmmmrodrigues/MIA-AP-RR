# -*- coding: utf-8 -*-

from datasets import load_dataset
from groq import Groq
import random


GROQ_API_KEY = "..." ## put your key here 
SEED = 42
random.seed(SEED)

TRAIN_SUBSET = 3
TEST_SUBSET  = 5

def ask_llama(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return chat.choices[0].message.content

def zero_shot(test_data): 
    base_prompt_zero = "We will provide a set of movie reviews. You should classify them as positive or negative and explain why."
    instruction = "Each review will be given by an integer index between [], followed by the text. The output should be a list of 0 (if negative) or 1 (if positive)"
    prompt = base_prompt_zero + " " + instruction
    
    for i in range(TEST_SUBSET):
        sample = test_data[i]        
        text = sample["text"][:200]
        prompt += "[" + str(i) + "]" + text
    
    try:
        print(ask_llama(prompt))
    except Exception as e:
        print(f"Error: {e}")
    print()
    
def few_shot(train_data, test_data): 
    base_prompt = "I will provide a set of example movie reviews. with the text and a label saying if they are positive (1) or negative (0). Each review starts with R: and ends [0] for negative or [1] for positive label."
    instruction1 = "After that, we will put a line with --------. We will then provide a set of reviews to classify as positive or negative and explain why."
    instruction2 = "Each review will be given by an integer index between [], followed by the text. The output in the end should be a list of 0 (if negative) or 1 (if positive) for each index"
    prompt = base_prompt + " " + instruction1 + " " + instruction2
    for i in range(TRAIN_SUBSET):
        sample = train_data[i]        
        text = sample["text"][:200]
        prompt += "R-" + text + " [" + str(train_data[i]['label'])+ "]."
        prompt += "\n------------\n"
        
    for i in range(TEST_SUBSET):
        sample = test_data[i]        
        text = sample["text"][:200]
        prompt += "[" + str(i) + "]" + text
        
    try:
        print(ask_llama(prompt))
    except Exception as e:
        print(f"Error: {e}")
    print()

### Main

print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_data = dataset["train"].shuffle(SEED)
test_data  = dataset["test"].shuffle(SEED)
train_data = train_data.select(range(TRAIN_SUBSET))
test_data = test_data.select(range(TEST_SUBSET))

#zero_shot(test_data)
few_shot(train_data, test_data)

correct = []
for i in range(TEST_SUBSET):
    correct.append(test_data[i]['label'])

print("Correct labels:", correct)



