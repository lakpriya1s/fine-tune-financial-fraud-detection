from dotenv import load_dotenv
import wandb
from realTimeProgressBar import StreamlitTrainerCallback
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
from transformers import pipeline
from tqdm import tqdm
from plot_result import evaluate
from trl import SFTTrainer
from peft import LoraConfig, PeftConfig
from log_stream import StreamToStreamlit
import sys


load_dotenv()

TOKEN = os.getenv("TOKEN") 
MODEL_LINK = "Felladrin/Llama-160M-Chat-v1"
MODEL_NAME = "Llama-160M-Chat-v1"
FINE_TUNED_MODEL_NAME = "Llama-160M-Chat-v1-fraud-detection"
FINE_TUNED_ONNX_MODEL_NAME = "Llama-160M-Chat-v1-fraud-detection-onnx"
wb_token = os.getenv('WANDB_API_KEY')

st.title("Fraud Detection Model Training")
os.environ["WANDB_MODE"] = "offline"

# wandb.login(key=wb_token)
run = wandb.init(
    project='Fraud Detection',
    job_type="training",
    anonymous="allow")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Class'] = data['Class'].map({1: 'Yes', 0: 'No'})
    data.rename(columns={'Class': 'Fraud'}, inplace=True)
    st.write("Uploaded Data:")
    st.write(data)
    data.head()

    text_column = st.selectbox("Select the text column", data.columns)
    target_column = st.selectbox("Select the target column", data.columns)

    if text_column and target_column:
        st.write("Processing the dataset...")
        data = data.sample(frac=1, random_state=42).reset_index(drop=True).head(4000)
        train_size = 0.8
        eval_size = 0.1

        train_end = int(train_size * len(data))
        eval_end = train_end + int(eval_size * len(data))

        train_data = data[:train_end]
        eval_data = data[train_end:eval_end]
        test_data = data[eval_end:]

        def generate_prompt(data_point):
            return f"""
        Is the following text fraudulent? Answer Yes or No only.
        Text: {data_point[text_column]}
        Fraud: {data_point["Fraud"]}""".strip()

        def generate_test_prompt(data_point):
            return f"""
        Is the following text fraudulent? Answer Yes or No only.
        Text: {data_point[text_column]}
        Fraud: """.strip()

        # train_data.loc[:,"text"] = train_data.apply(generate_prompt, axis=1)
        # eval_data.loc[:,"text"] = eval_data.apply(generate_prompt, axis=1)

    
        # test_data = pd.DataFrame(test_data.apply(generate_test_prompt, axis=1), columns=["text"])
        

        
        if st.button("Start Training"):
            steps = [
                "Downloading the model...",
                "Generating prompts...",
                "Predicting...",
                "Fine-tuning the model...",
                "Predicting after the fine-tuning...",
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
                torch_dtype=torch.float32, #changed 16 to 32
                return_dict=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            model.config.use_cache = False
            model.config.pretraining_tp = 1

            tokenizer = AutoTokenizer.from_pretrained(MODEL_LINK)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
            bar.progress(100)
            step_status[0].markdown(f"**{steps[0]} ✔ Done**")
            progress_bars[0].empty() 

            # --- Step 2: Generating prompts ---
            step_status[1].markdown(f"**{steps[1]}**")
            bar = progress_bars[1].progress(0)

            # for i in range(len(train_data)):
            #     train_data.loc[i, "text"] = generate_prompt(train_data.iloc[i])
            #     bar.progress(int((i + 1) / len(train_data) * 100))

            # for i in range(len(eval_data)):
            #     eval_data.loc[i, "text"] = generate_prompt(eval_data.iloc[i])
            #     bar.progress(int((i + 1) / len(eval_data) * 100))
            
            # y_true = test_data['Fraud']
            # st.write("True labels 1:", y_true)
            # for i in range(len(test_data)):
                # test_data.loc[i, "text"] = generate_test_prompt(test_data.iloc[i])
                # bar.progress(int((i + 1) / len(test_data) * 100))

            # comment for now to fine tuning
            train_data.loc[:,'text'] = train_data .apply(generate_prompt, axis=1)
            eval_data.loc[:,'text'] = eval_data.apply(generate_prompt, axis=1)

            # Generate test prompts and extract true labels
            y_true = test_data['Fraud']
            # st.write("True labels:", y_true)
            test_data = pd. DataFrame(test_data.apply(generate_test_prompt, axis=1), columns=["text"])
            
            train_dataset = Dataset.from_pandas(train_data[["text"]])
            eval_dataset = Dataset.from_pandas(eval_data[["text"]])

            bar.progress(100)
            step_status[1].markdown(f"**{steps[1]} ✔ Done**")
            progress_bars[1].empty()

            # --- Step 3: Predicting ---

            # step_status[2].markdown(f"**{steps[2]}**")
            # bar = progress_bars[2].progress(0)

            # progress_text = st.empty()

            def predict(test, model, tokenizer):
                
                y_pred = []
                categories = ["Yes", "No"]

                pipe = pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=10,
                    temperature=0.1
                )

                for i in tqdm(range(len(test))):
                    prompt = test.iloc[i]["text"]
                    result = pipe(prompt)

                    # Remove the prompt from the generated text
                    generated_text = result[0]['generated_text']
                    answer = generated_text[len(prompt):].strip()


                    print("prompt:" + prompt)
                    print("answer:" + answer)

                    # Determine the predicted category
                    for category in categories:
                        if category.lower() in answer.lower():
                            y_pred.append(category)
                            break
                    else:
                        y_pred.append("none")
                
                    progress = int((i + 1) / len(test) * 100)
                    bar.progress(progress)
                    progress_text.text(f"Progress: {progress}%")

                return y_pred

            # y_pred_base = predict(test_data, model, tokenizer)
            # y_pred_base = pd.Series(y_pred_base)

            # bar.progress(100)
            # progress_text.text("Progress: 100%")
            # step_status[2].markdown(f"**{steps[2]} ✔ Done**")
            # progress_bars[2].empty()

            
            # --- Step 4: FineTuning the model ---

            step_status[3].markdown(f"**{steps[3]}**")
            train_bar = progress_bars[3].progress(0)

            def find_all_linear_names_cpu(model):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[-1])  # use last part for clarity
                if 'lm_head' in lora_module_names:
                    lora_module_names.remove('lm_head')  # still skip the output head
                return list(lora_module_names)
            

            modules = find_all_linear_names_cpu(model)

            output_dir=MODEL_NAME

            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=modules,
            )

            training_arguments = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                optim="adamw_torch",  # ✅ CPU-compatible optimizer
                logging_steps=1,
                learning_rate=2e-5,
                weight_decay=0.001,
                fp16=False,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=False,
                lr_scheduler_type="cosine",
                report_to="none",  # Disable wandb if you want faster no-network training
            )


            trainer = SFTTrainer(
                model=model,
                args=training_arguments,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                # tokenizer=tokenizer,
            )

            log_output = st.empty()

            stream = StreamToStreamlit(log_output, progress_bar=train_bar)
            original_stdout = sys.stdout
            sys.stdout = stream

            try:
                trainer.train()
            finally:
                sys.stdout = original_stdout

            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            train_bar.progress(100)
            step_status[3].markdown(f"**{steps[3]} ✔ Done**")
            progress_bars[3].empty()

            # ---step 5: Predicting with the fine-tuned model ---
            step_status[4].markdown(f"**{steps[4]}**")
            bar = progress_bars[4].progress(0)

            progress_text = st.empty()

            y_pred_fine = predict(test_data, model, tokenizer)
            bar.progress(100)
            progress_text.text("Progress: 100%")
            step_status[4].markdown(f"**{steps[4]} ✔ Done**")
            progress_bars[4].empty()

            # -- step 6: Evaluating the model ---

            step_status[5].markdown(f"**{steps[5]}**")
            bar = progress_bars[5].progress(0)
            
            labels = ["Yes", "No"]

            
            # base_metrics = evaluate(y_true, y_pred_base, labels=labels, measure_time=True, measure_memory=True)
            fine_tuned_metrics = evaluate(y_true, y_pred_fine, labels=labels, measure_time=True, measure_memory=True)

            bar.progress(100)
            step_status[5].markdown(f"**{steps[5]} ✔ Done**")
            progress_bars[5].empty()
            # Access and print metrics
            print("Overall Accuracy:", fine_tuned_metrics["overall_accuracy"])
            print("Accuracy per Label:", fine_tuned_metrics["label_accuracies"])
            print("Classification Report:")
            print(fine_tuned_metrics["classification_report"])
            print("Confusion Matrix:")
            print(fine_tuned_metrics["confusion_matrix"])
            print("ROC-AUC:", fine_tuned_metrics["roc_auc"])
            print("PR-AUC:", fine_tuned_metrics["pr_auc"])
            print("Inference Time:", fine_tuned_metrics["inference_time"])
            print("Memory Usage (MB):",fine_tuned_metrics['memory_usage'])
            

            # print("Overall Accuracy:", base_metrics["overall_accuracy"])
            # print("Accuracy per Label:", base_metrics["label_accuracies"])
            # print("Classification Report:")
            # print(base_metrics["classification_report"])
            # print("Confusion Matrix:")
            # print(base_metrics["confusion_matrix"])
            # print("ROC-AUC:", base_metrics["roc_auc"])
            # print("PR-AUC:", base_metrics["pr_auc"])
            # print("Inference Time:", base_metrics["inference_time"])
            # print("Memory Usage (MB):",base_metrics['memory_usage'])
            #  # --- Step 5: Uploading the model ---
            # step_status[4].markdown(f"**{steps[4]}**")
            # bar = progress_bars[4].progress(0)

            # try:
            #     model_dir = "./results"
            #     files = {
            #         "model": open(os.path.join(model_dir, "pytorch_model.bin"), "rb"),
            #         "config": open(os.path.join(model_dir, "config.json"), "rb"),
            #         "tokenizer": open(os.path.join(model_dir, "tokenizer_config.json"), "rb"),
            #     }
            #     response = requests.post(UPLOAD_URL, files=files)
            #     if response.status_code == 200:
            #         st.success("Model uploaded successfully! ✔")
            #     else:
            #         st.error(f"Failed to upload the model. Status code: {response.status_code}")
            # except Exception as e:
            #     st.error(f"Upload failed: {e}")

            # bar.progress(100)
            # step_status[4].markdown(f"**{steps[4]} ✔ Done**")
            # progress_bars[4].empty()

            # st.success("🎉 Training complete!")