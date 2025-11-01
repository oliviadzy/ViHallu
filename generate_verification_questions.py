import os
import json
import openai
from tqdm import tqdm

# Read API key from environment to avoid hard-coded secrets
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_BASE = "https://api.deepseek.com"

openai.api_key = DEEPSEEK_API_KEY
openai.api_base = DEEPSEEK_API_BASE


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# NOTE: Update the input path to your local dataset before running.
input_data = read_json('/data/dzy/MLLM/test/output_split_3.json')
output_data = []

for entry in tqdm(input_data, desc="Processing images", unit="image"):
    prompt = f"sentence: {entry.get('answer', '')}"

    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": (
                "You are a language assistant that helps generate relevant yes-or-no questions about a sentence. "
                "Provide up to five concise (<=20 words) verification questions. Each line should start with '&' followed by the question."
            )},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    entry["response"] = response['choices'][0]['message']['content']
    output_data.append(entry)
    save_json(output_data, '/data/dzy/MLLM/test/zancun2/qwen_eval_ds_question3.json')


print("Data with responses saved to 'qwen_eval_ds_question3.json'.")

