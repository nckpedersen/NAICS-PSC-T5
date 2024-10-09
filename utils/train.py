import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import Dataset
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to simulate user-triggered stop
STOP_TRAINING = False

def check_early_stop():
    """This function checks whether to stop training early."""
    global STOP_TRAINING
    return STOP_TRAINING  # Set STOP_TRAINING to True when you want to stop

class CustomEarlyStoppingCallback(TrainerCallback):
    """Custom callback to stop training early based on user trigger or validation loss."""
    def __init__(self, stop_condition_func):
        self.stop_condition_func = stop_condition_func

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_condition_func():
            logging.info("Early stopping triggered by user!")
            control.should_training_stop = True
        return control

def load_model():
    """Load pre-trained T5 model."""
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    return model, tokenizer

def prepare_dataset(data):
    """Prepare dataset for HuggingFace Trainer."""
    logging.info(f"Preparing dataset. Type of data received: {type(data)}")
    if isinstance(data, pd.DataFrame):
        return Dataset.from_pandas(data)  # Convert Pandas DataFrame to HuggingFace Dataset
    else:
        logging.error(f"Error: Expected a Pandas DataFrame, but got {type(data)} instead.")
        raise ValueError(f"Expected a Pandas DataFrame, but got {type(data)}")

def train_model(train_data, val_data):
    """Train the model with multitask input-output pairs."""
    logging.info("Loading the model and tokenizer...")
    model, tokenizer = load_model()

    # Check if GPU is available and set to GPU if possible
    if torch.cuda.is_available():
        model.to('cuda')
        logging.info("Model is running on GPU.")
    else:
        logging.info("Model is running on CPU.")

    # Prepare dataset
    logging.info("Preparing the datasets...")
    try:
        train_dataset = prepare_dataset(train_data)
        val_dataset = prepare_dataset(val_data)
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")
        raise

    # Log a few examples from the training dataset
    logging.info("Here are a few examples of input-output pairs from the training data:")
    for i in range(3):  # Print the first 3 examples
        input_ids = train_dataset[i]['input_ids']
        labels = train_dataset[i]['labels']
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        output_text = tokenizer.decode(labels, skip_special_tokens=True)
        logging.info(f"Example {i + 1}:")
        logging.info(f"Input: {input_text}")
        logging.info(f"Output: {output_text}")

    # Use steps for both evaluation and saving
    training_args = TrainingArguments(
        output_dir='./models/t5_model',
        per_device_train_batch_size=16,  # Slightly larger batch size for more efficient training
        per_device_eval_batch_size=16,  # Larger eval batch size for faster validation
        num_train_epochs=3,  # Increase to 3 for better training coverage
        evaluation_strategy="steps",  # Evaluate periodically during training
        save_strategy="steps",  # Save model periodically during training
        save_steps=500,  # Save every 500 steps (less frequent than quick test)
        eval_steps=500,  # Evaluate every 500 steps
        logging_dir='./logs',
        logging_steps=50,  # Log every 50 steps (more detailed logging)
        logging_first_step=True,
        load_best_model_at_end=True,  # Ensure best model is loaded at the end
        weight_decay=0.01,  # Regularization to avoid overfitting
        fp16=True,  # Mixed precision for faster training
        warmup_steps=500,  # Add warmup steps for gradual learning rate increase
        save_total_limit=1,  # Limit to only the most recent checkpoint
        learning_rate=5e-5,  # A medium learning rate, adjustable based on performance
        report_to="none"
    )

    logging.info(f"Starting the training with {len(train_dataset)} training examples and {len(val_dataset)} validation examples...")

    # Initialize Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomEarlyStoppingCallback(check_early_stop)]  # Custom callback for early stopping
    )

    # Train the model
    logging.info("Training has started.")
    trainer.train()
    logging.info("Training has finished.")

    # Save the final model and tokenizer
    logging.info("Saving the trained model and tokenizer.")
    model.save_pretrained('./models/t5_model')
    tokenizer.save_pretrained('./models/t5_model')

    logging.info("Model saved successfully.")


def test_model(test_data):
    """Evaluate the model on the test dataset."""
    logging.info("Loading the trained model and tokenizer for testing...")
    model, tokenizer = load_model()

    # Check if GPU is available and set to GPU if possible
    if torch.cuda.is_available():
        model.to('cuda')
        logging.info("Model is running on GPU for testing.")
    else:
        logging.info("Model is running on CPU for testing.")

    # Prepare the test dataset
    logging.info("Preparing the test dataset...")
    test_dataset = prepare_dataset(test_data)

    # Initialize Trainer with EarlyStoppingCallback for testing
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir='./models/t5_model', per_device_eval_batch_size=4),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomEarlyStoppingCallback(check_early_stop)]
    )

    # Evaluate the model on the test set
    logging.info("Evaluating the model on the test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    logging.info(f"Test set evaluation results: {test_results}")

