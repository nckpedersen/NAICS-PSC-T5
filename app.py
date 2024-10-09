import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from utils.etl import load_data, preprocess_data, split_data
from utils.train import train_model, test_model
from utils.inference import load_trained_model, infer
from transformers import T5Tokenizer

# Initialize FastAPI
app = FastAPI()

# Global variable to handle early stop requests
STOP_TRAINING = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define request body model for inference
class InferenceRequest(BaseModel):
    description: Optional[str] = None
    naics: Optional[str] = None
    psc: Optional[str] = None

@app.post("/train")
def train():
    global STOP_TRAINING
    STOP_TRAINING = False  # Reset early stop flag

    try:
        logging.info("Starting training process...")

        # Step 1: Load the dataset
        logging.info("Loading the dataset from './data/dataset.csv'.")
        data = load_data('./data/dataset.csv')
        logging.info(f"Dataset loaded successfully. Number of records: {len(data)}.")

        # Step 2: Initialize the tokenizer
        logging.info("Loading T5 tokenizer from 't5-small'.")
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        logging.info("Tokenizer loaded successfully.")

        # Step 3: Preprocess the data and create multitask inputs/outputs
        logging.info("Preprocessing the dataset and creating multitask input-output pairs.")
        preprocessed_data = preprocess_data(data, tokenizer)
        logging.info(f"Preprocessing completed. Number of preprocessed records: {len(preprocessed_data)}.")

        # Step 4: Split data into train/validation/test
        logging.info("Splitting the data into training, validation, and test sets.")
        train_data, val_data, test_data = split_data(preprocessed_data)
        logging.info(f"Data splitting completed. Training set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}.")

        # Step 5: Train the model with early stopping and regularization
        logging.info("Starting the model training with early stopping.")
        train_model(train_data, val_data)
        logging.info("Model training completed successfully.")

        # Step 6: Test the model
        logging.info("Starting test on the test dataset.")
        test_model(test_data)

        return {"message": "Training and testing completed successfully!"}
    except Exception as e:
        logging.error(f"An error occurred during the training process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/early_stop")
def early_stop():
    global STOP_TRAINING
    STOP_TRAINING = True
    return {"message": "Early stop signal received."}

@app.post("/infer")
async def inference(request: InferenceRequest):
    try:
        model, tokenizer = load_trained_model()
        result = infer(model, tokenizer, request.description, request.naics, request.psc)
        return {"prediction": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))