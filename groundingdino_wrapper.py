"""
Grounding DINO helper wrapper.
See: https://github.com/IDEA-Research/GroundingDINO
"""

import logging
from typing import Union

import numpy as np
import torch
from groundingdino.util.inference import (
    load_model as _load_model,
    predict as _predict,
    load_image,
    annotate
)
from groundingdino.util import box_ops
from PIL import Image

def load_model(
    config_path : str = "groundingdino_swint_ogc.py",
    model_path : str = "groundingdino_swint_ogc.pth"):
    return _load_model(config_path, model_path)

def predict(
    model           : torch.nn.Module,
    image_path      : str = None,
    image           : Image = None,
    box_threshold   : float = 0.35,
    text_threshold  : float = 0.25,
    prompt          : str = None,
    device          : Union[torch.device, str] = None
):

    if image_path:
        logging.info('Loading image %s', image_path)
        _, image = load_image(image_path)

    assert torch.is_tensor(image), 'Image parameter should be a tensor'

    if device:
        image = image.to(device)

    boxes, logits, phrases = _predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    logging.info('Total number of %d boxes found', len(boxes))
    return boxes, logits, np.array(phrases)

