import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from transformers import BitsAndBytesConfig

MODEL_LINK = "Felladrin/Llama-160M-Chat-v1"
MODEL_NAME = "Llama-160M-Chat-v1"
FINE_TUNED_MODEL_NAME = "Llama-160M-Chat-v1-fraud-detection"
FINE_TUNED_ONNX_MODEL_NAME = "Llama-160M-Chat-v1-fraud-detection-onnx"

# Title of the app
st.title("CSV Upload and Model Training with Fine-Tuning")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    # Let the user select the feature column(s) and target column
    text_column = st.selectbox("Select the text column", data.columns)
    target_column = st.selectbox("Select the target column", data.columns)

    if text_column and target_column:
        # Split data into features and target
        X = data[text_column]
        y = data[target_column]

        # Split into train and test sets
        test_size = st.slider("Select the percentage of data for testing", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Load the base model and tokenizer
        base_model_name =  MODEL_LINK
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            return_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Preprocess the data
        def preprocess_data(texts, labels):
            encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=512, return_tensors="pt")
            labels = torch.tensor(labels.values)
            return encodings, labels

        train_encodings, train_labels = preprocess_data(X_train, y_train)
        test_encodings, test_labels = preprocess_data(X_test, y_test)

        # Fine-tuning setup
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Define Trainer
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = self.labels[idx]
                return item

        train_dataset = CustomDataset(train_encodings, train_labels)
        test_dataset = CustomDataset(test_encodings, test_labels)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        # Train the model
        st.write("Fine-tuning the model...")
        trainer.train()

        # Evaluate the model
        st.write("Evaluating the model...")
        predictions = trainer.predict(test_dataset)
        y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()

        # Calculate accuracy
        fine_tuned_accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy after fine-tuning: {fine_tuned_accuracy:.2f}")

        # Compare with pre-fine-tuned accuracy
        st.write("Evaluating the base model (before fine-tuning)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            return_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        trainer.model = base_model  # Use the base model for evaluation
        predictions = trainer.predict(test_dataset)
        y_pred_base = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
        base_accuracy = accuracy_score(y_test, y_pred_base)
        st.write(f"Accuracy before fine-tuning: {base_accuracy:.2f}")

        # Display comparison
        st.write(f"Accuracy improvement: {fine_tuned_accuracy - base_accuracy:.2f}")
    else:
        st.error("Please select both a text column and a target column.")