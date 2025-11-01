import os
import json
import random
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from controlnet_aux import SamDetector
from diffusers import EulerAncestralDiscreteScheduler
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline
from tqdm import tqdm

# Set up device
device = torch.device('cuda:0')

# Load models and configuration
eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("/data/models/stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("/data/dzy/MLLM/model/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
controlnet_model = ControlNetModel_Union.from_pretrained("/data/models/xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)

pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    "/data/models/stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
    vae=vae, torch_dtype=torch.float16, scheduler=eulera_scheduler
)
pipe = pipe.to(device)

processor = SamDetector.from_pretrained('/data/dzy/MLLM/model/dhkim2810/MobileSAM', model_type="vit_t", filename="mobile_sam.pt")

# Define constants
input_image_folder = "/data/dzy/MLLM/dataset/val2014_500/"
output_image_folder = "/data/dzy/MLLM/t2i/ControlNetPlus/output2_1"
json_file_path = "/data/dzy/MLLM/test2/zancun/captions.json"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
results_json_path = "/data/dzy/MLLM/test2/zancun/results.json" 

# Load JSON file
with open(json_file_path, 'r') as f:
    images_data = json.load(f)

results = []
# Process each image and corresponding prompt
for item in tqdm(images_data, desc="Processing images", unit="image"):
    image_path = os.path.join(input_image_folder, item['image'])
    rewritten_captions = item['rewritten_captions']
    for caption in rewritten_captions:
        save_path = os.path.join(output_image_folder, item['image'].replace(".jpg", f"_{caption[:10]}{i}.jpg"))


        # Read image
        controlnet_img = cv2.imread(image_path)
        if controlnet_img is None:
            print(f"Cannot read image: {image_path}")
            continue

        controlnet_img = processor(controlnet_img, output_type='cv2')

        # Resize image to 1024x1024
        height, width, _ = controlnet_img.shape
        ratio = np.sqrt(1024. * 1024. / (width * height))
        new_width, new_height = int(width * ratio), int(height * ratio)
        controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
        controlnet_img = Image.fromarray(controlnet_img)

        # Generate random seed
        seed = random.randint(0, 2147483647)
        generator = torch.Generator('cuda').manual_seed(seed)

        # Use control network to generate image
        images = pipe(
            prompt=[caption] * 1,
            image_list=[0, 0, 0, 0, 0, controlnet_img],
            negative_prompt=[negative_prompt] * 1,
            generator=generator,
            width=new_width,
            height=new_height,
            num_inference_steps=30,
            union_control=True,
            union_control_type=torch.Tensor([0, 0, 0, 0, 0, 1]),
        ).images


        # Save the generated image
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        images[0].save(save_path)
        print(f"Saved image: {save_path}")
        # Append results for JSON output
        results.append({
            "image_filename": save_path,
            "prompt": caption
        })

# Save results to JSON
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4)

    #CUDA_VISIBLE_DEVICES=0 nohup python test1.py > log.txt 2>&1 &
