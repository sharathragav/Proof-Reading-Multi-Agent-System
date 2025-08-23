import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os

logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

def load_models(
    small_name=None,
    big_name=None,
    use_8bit=False
):
    """
    Loads the small and big models. If not provided, defaults to:
    - small_name: download flan-t5-small into ./models/flan-t5-small
    - big_name: use local coedit-large-local in ./models/coedit-large-local
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default paths
    if small_name is None:
        small_name = "./models/flan-t5-small"
    if big_name is None:
        big_name = "./models/coedit-large-local"

    # Download flan-t5-small if not present locally
    if not os.path.exists(small_name):
        logger.info(f"Downloading flan-t5-small to {small_name}")
        small_tok = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=small_name)
        small_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=small_name)
    else:
        logger.info(f"Loading flan-t5-small from {small_name}")
        small_tok = AutoTokenizer.from_pretrained(small_name)
        small_model = AutoModelForSeq2SeqLM.from_pretrained(small_name)
    small_model.to(device)

    logger.info(f"Loading big model from {big_name} (use_8bit={use_8bit})")
    try:
        if use_8bit:
            big_tok = AutoTokenizer.from_pretrained(big_name)
            big_model = AutoModelForSeq2SeqLM.from_pretrained(big_name, load_in_8bit=True, device_map="auto")
        else:
            big_tok = AutoTokenizer.from_pretrained(big_name)
            big_model = AutoModelForSeq2SeqLM.from_pretrained(big_name)
            big_model.to(device)
    except Exception as exc:
        logger.exception("Big model load failed, retry default CPU load: %s", exc)
        big_model = AutoModelForSeq2SeqLM.from_pretrained(big_name)
        big_model.to(device)
    return small_tok, small_model, device, big_tok, big_model, device