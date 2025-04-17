from dotenv import load_dotenv
from realTimeProgressBar import StreamlitTrainerCallback
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
import requests
import os

load_dotenv()

MODEL_LINK = "Felladrin/Llama-160M-Chat-v1"
TOKEN = os.getenv("TOKEN") 
st.title("Fraud Detection Model Training")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)
    data['Class'] = data['Class'].map({1: 'Yes', 0: 'No'})
    data.head()

    text_column = st.selectbox("Select the text column", data.columns)
    target_column = st.selectbox("Select the target column", data.columns)

    if text_column and target_column:
        st.write("Processing the dataset...")
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = 0.8
        eval_size = 0.1

        train_end = int(train_size * len(data))
        eval_end = train_end + int(eval_size * len(data))

        train_data = data[:train_end]
        eval_data = data[train_end:eval_end]
        test_data = data[eval_end:]

        def generate_prompt(data_point):
            return f"""Is the following text fraudulent? Answer Yes or No only.
            Text: {data_point[text_column]}
            Fraud: {data_point[target_column]}""".strip()

        train_data["text"] = train_data.apply(generate_prompt, axis=1)
        eval_data["text"] = eval_data.apply(generate_prompt, axis=1)

        train_dataset = Dataset.from_pandas(train_data[["text"]])
        eval_dataset = Dataset.from_pandas(eval_data[["text"]])

        if st.button("Start Training"):
            steps = [
                "Downloading the model...",
                "Preprocessing the data...",
                "Fine-tuning the model...",
                "Evaluating the model...",
                "Uploading the model...",
            ]

            step_status = [st.empty() for _ in steps]
            progress_bars = [st.empty() for _ in steps]

            # --- Step 1: Downloading the model ---
            step_status[0].markdown(f"**{steps[0]}**")
            bar = progress_bars[0].progress(0)
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
            bar.progress(100)
            step_status[0].markdown(f"**{steps[0]} ✔ Done**")
            progress_bars[0].empty() 

            # --- Step 2: Preprocessing the data ---
            step_status[1].markdown(f"**{steps[1]}**")
            bar = progress_bars[1].progress(0)

            def preprocess_data(dataset):
                return dataset.map(
                    lambda x: tokenizer(
                        x["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                    ),
                    batched=True,
                )

            train_dataset = preprocess_data(train_dataset)
            eval_dataset = preprocess_data(eval_dataset)

            bar.progress(100)
            step_status[1].markdown(f"**{steps[1]} ✔ Done**")
            progress_bars[1].empty()

            # --- Step 3: Fine-tuning the model ---
            step_status[2].markdown(f"**{steps[2]}**")
            bar = progress_bars[2].progress(0)

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

            
            total_steps = len(train_dataset) * training_args.num_train_epochs
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[StreamlitTrainerCallback(total_steps=total_steps)]
            )

            trainer.train()

            bar.progress(100)
            step_status[2].markdown(f"**{steps[2]} ✔ Done**")
            progress_bars[2].empty()

            # --- Step 4: Evaluating the model ---
            step_status[3].markdown(f"**{steps[3]}**")
            bar = progress_bars[3].progress(0)

            predictions = trainer.predict(eval_dataset)
            y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
            fine_tuned_accuracy = accuracy_score(eval_data[target_column], y_pred)

            st.write(f"**Accuracy after fine-tuning:** `{fine_tuned_accuracy:.2f}`")

            bar.progress(100)
            step_status[3].markdown(f"**{steps[3]} ✔ Done**")
            progress_bars[3].empty()

            # --- Step 5: Uploading the model ---
            step_status[4].markdown(f"**{steps[4]}**")
            bar = progress_bars[4].progress(0)

            try:
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
            except Exception as e:
                st.error(f"Upload failed: {e}")

            bar.progress(100)
            step_status[4].markdown(f"**{steps[4]} ✔ Done**")
            progress_bars[4].empty()

            st.success("🎉 Training complete!")

