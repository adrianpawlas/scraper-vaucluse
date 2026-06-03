"""Embedding module using google/siglip-base-patch16-384 for image and text embeddings."""

import io
import logging
import threading
import time
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    DOWNLOAD_TIMEOUT,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Global model cache (load once, reuse) – thread-safe
_model = None
_processor = None
_model_lock = threading.Lock()


def get_model_and_processor():
    """Lazy-load the SigLIP model and processor (thread-safe singleton)."""
    global _model, _processor
    if _model is None or _processor is None:
        with _model_lock:
            # Double-check after acquiring lock
            if _model is None or _processor is None:
                logger.info(f"Loading model: {EMBEDDING_MODEL}")
                _model = AutoModel.from_pretrained(EMBEDDING_MODEL)
                _processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL)
                _model.eval()
                logger.info(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")
    return _model, _processor


def download_image(url: str) -> Optional[Image.Image]:
    """Download an image from a URL and return as PIL Image."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed to download {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def generate_image_embedding(image: Image.Image, normalize: bool = True) -> Optional[list[float]]:
    """Generate a 768-dimensional image embedding using SigLIP.

    Args:
        image: PIL Image to embed.
        normalize: Whether to L2-normalize the embedding.

    Returns:
        List of floats (768-dim) or None if generation fails.
    """
    try:
        model, processor = get_model_and_processor()
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # get_image_features returns BaseModelOutputWithPooling; extract pooler_output
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings[0].tolist()
    except Exception as e:
        logger.error(f"Failed to generate image embedding: {e}")
        return None


def generate_image_embedding_from_url(url: str, normalize: bool = True) -> Optional[list[float]]:
    """Download image from URL and generate embedding."""
    image = download_image(url)
    if image is None:
        logger.warning(f"Could not download image from {url}")
        return None
    return generate_image_embedding(image, normalize=normalize)


def generate_text_embedding(text: str, normalize: bool = True) -> Optional[list[float]]:
    """Generate a 768-dimensional text embedding using SigLIP.

    Args:
        text: Text to embed.
        normalize: Whether to L2-normalize the embedding.

    Returns:
        List of floats (768-dim) or None if generation fails.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding.")
        return None

    try:
        model, processor = get_model_and_processor()
        inputs = processor(
            text=[text],
            padding="max_length",
            max_length=64,  # SigLIP uses 64 tokens max
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            # get_text_features returns BaseModelOutputWithPooling; extract pooler_output
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings[0].tolist()
    except Exception as e:
        logger.error(f"Failed to generate text embedding: {e}")
        return None


def generate_embeddings_for_product(
    product: dict,
) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """Generate both image and text embeddings for a product.

    Args:
        product: Parsed product dict with 'image_url' and 'info_text' fields.

    Returns:
        Tuple of (image_embedding, info_embedding).
    """
    # Generate image embedding from the main product image
    image_embedding = None
    image_url = product.get("image_url")
    if image_url:
        logger.info(f"Generating image embedding for {product['title']}...")
        image_embedding = generate_image_embedding_from_url(image_url)
        if image_embedding:
            logger.info(f"Image embedding generated (dim={len(image_embedding)})")
        else:
            logger.warning(f"Image embedding failed for {product['title']}")

    # Generate text embedding from product info
    info_text = product.get("info_text", "")
    if info_text:
        logger.info(f"Generating text embedding for {product['title']}...")
        info_embedding = generate_text_embedding(info_text)
        if info_embedding:
            logger.info(f"Text embedding generated (dim={len(info_embedding)})")
        else:
            logger.warning(f"Text embedding failed for {product['title']}")
    else:
        info_embedding = None

    return image_embedding, info_embedding
