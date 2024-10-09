import torch
import logging
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_trained_model():
    """Load the trained model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained('./models/t5_model')
    tokenizer = T5Tokenizer.from_pretrained('./models/t5_model')

    if torch.cuda.is_available():
        model = model.to('cuda')

    return model, tokenizer


def infer(model, tokenizer, description=None, naics=None, psc=None):
    """Run inference with the trained model."""
    logging.info(f"Inference started with inputs - Description: {description}, NAICS: {naics}, PSC: {psc}")

    # Determine which fields are provided and which are missing
    fields = {'Description': description, 'NAICS': naics, 'PSC': psc}
    provided = [k for k, v in fields.items() if v]
    missing = [k for k, v in fields.items() if not v]

    if not provided:
        raise ValueError("No fields provided. Please provide at least one field.")

    # Create input text based on provided fields
    input_text = " ".join(f"{k}: {v}" for k, v in fields.items() if v)

    # Tokenize and encode the input (consistent with training)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    # Generate output
    output = model.generate(input_ids, max_length=512, num_return_sequences=1)

    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Parse the output (improved parsing logic)
    output_fields = {}
    for field in missing:  # Only parse for fields that were NOT provided in the input
       match = re.search(rf"{field}:\s*(\S+)", decoded_output) # Non-greedy match, extracts first word/code
       if match:
           output_fields[field] = match.group(1)
       else:
           output_fields[field] = f"[Unable to infer {field}]"
           
    return output_fields