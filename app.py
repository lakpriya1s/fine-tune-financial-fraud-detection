import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
import requests  # For uploading the model
import os  # For file handling

MODEL_LINK = "Felladrin/Llama-160M-Chat-v1"
UPLOAD_URL = "https://your-upload-url.com/upload"  # Replace with your URL

# Title of the app
st.title("Fraud Detection Model Training")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    # User selects the text and target columns
    text_column = st.selectbox("Select the text column", data.columns)
    target_column = st.selectbox("Select the target column", data.columns)

    if text_column and target_column:
        # Automatically process the dataset
        st.write("Processing the dataset...")
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data
        train_size = 0.8
        eval_size = 0.1

        # Split the data
        train_end = int(train_size * len(data))
        eval_end = train_end + int(eval_size * len(data))

        train_data = data[:train_end]
        eval_data = data[train_end:eval_end]
        test_data = data[eval_end:]

        # Generate prompts
        def generate_prompt(data_point):
            return f"""Is the following text fraudulent? Answer Yes or No only.
            Text: {data_point[text_column]}
            Fraud: {data_point[target_column]}""".strip()

        train_data["text"] = train_data.apply(generate_prompt, axis=1)
        eval_data["text"] = eval_data.apply(generate_prompt, axis=1)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_data[["text"]])
        eval_dataset = Dataset.from_pandas(eval_data[["text"]])

        # Button to start training
        if st.button("Start Training"):
            steps = [
                "Downloading the model...",
                "Preprocessing the data...",
                "Fine-tuning the model...",
                "Evaluating the model...",
                "Uploading the model...",
            ]
            progress_bar = st.progress(0)  # Initialize progress bar
            status_placeholder = st.empty()  # Placeholder for step status

            # Update the status dynamically
            for i, step in enumerate(steps):
                # Update progress percentage
                progress_percentage = int((i / len(steps)) * 100)
                progress_bar.progress(progress_percentage)

                # Display the current step
                status_placeholder.markdown(f"**{step}**")
                st.write("")  # Add spacing

                if i == 0:  # Step 1: Downloading the model
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_LINK,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        return_dict=True,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_LINK)
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                elif i == 1:  # Step 2: Preprocessing the data
                    def preprocess_data(dataset):
                        encodings = tokenizer(
                            list(dataset["text"]),
                            truncation=True,
                            padding=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        return encodings

                    train_encodings = preprocess_data(train_dataset)
                    eval_encodings = preprocess_data(eval_dataset)

                elif i == 2:  # Step 3: Fine-tuning the model
                    training_args = TrainingArguments(
                        output_dir="./results",
                        num_train_epochs=1,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing=True,
                        learning_rate=2e-5,
                        weight_decay=0.001,
                        logging_steps=1,
                        fp16=False,
                        bf16=False,
                        max_grad_norm=0.3,
                        warmup_ratio=0.03,
                        lr_scheduler_type="cosine",
                        evaluation_strategy="steps",
                        eval_steps=0.2,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                    )

                    trainer.train()

                elif i == 3:  # Step 4: Evaluating the model
                    predictions = trainer.predict(eval_dataset)
                    y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
                    fine_tuned_accuracy = accuracy_score(eval_data[target_column], y_pred)
                    st.write(f"Accuracy after fine-tuning: {fine_tuned_accuracy:.2f}")

                elif i == 4:  # Step 5: Uploading the model
                    status_placeholder.markdown("**Uploading the model...**")
                    model_dir = "./results"
                    files = {
                        "model": open(os.path.join(model_dir, "pytorch_model.bin"), "rb"),
                        "config": open(os.path.join(model_dir, "config.json"), "rb"),
                        "tokenizer": open(os.path.join(model_dir, "tokenizer_config.json"), "rb"),
                    }
                    response = requests.post(UPLOAD_URL, files=files)
                    if response.status_code == 200:
                        st.success("Model uploaded successfully! ✔")
                    else:
                        st.error(f"Failed to upload the model. Status code: {response.status_code}")

                # Mark the step as done
                status_placeholder.markdown(f"**{step} ✔ Done**")

            # Complete the progress bar
            progress_bar.progress(100)
            st.success("Training complete! ✔")