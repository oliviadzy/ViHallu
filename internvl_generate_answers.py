import json
import os
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Load input questions file (path configurable via env variable)
input_questions = os.environ.get('INTERNVL_INPUT_JSON', 'cleaned_question.json')
with open(input_questions, 'r', encoding='utf-8') as f:
    data = json.load(f)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model_path = os.environ.get('INTERNVL_MODEL_PATH', '/data/dzy/MLLM/OpenGVLab/InternVL2-26B')
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

image_folder = os.environ.get('INTERNVL_IMAGE_FOLDER', '/data/dzy/MLLM/dataset/cc_sbu_align/image')

results_with_answers = {"results": []}

for entry in tqdm(data.get("annotations", []), desc="Processing images and questions"):
    image_id = entry.get("Image ID")
    question = entry.get("Processed Question")
    prompt_text = "Answer the following question based on the picture. First give a yes or no answer, then give a simple explanation, don't answer too long: " + (question or "")

    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    if os.path.exists(image_path):
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=50, do_sample=True)
        response = model.chat(tokenizer, pixel_values, prompt_text, generation_config)

        results_with_answers["results"].append({
            "Image ID": image_id,
            "Processed Question": question,
            "Answer": response
        })
    else:
        print(f"Image {image_id}.jpg not found.")

output_file = os.environ.get('INTERNVL_OUTPUT_JSON', 'results_with_answers.json')
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results_with_answers, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")

