from transformers import TrainerCallback
import time
import streamlit as st

# Initialize placeholders globally
progress_bar_live = st.empty()
status_text_live = st.empty()

# Custom Callback
class StreamlitTrainerCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def on_step_end(self, args, state, control, **kwargs):
        current = state.global_step
        loss = state.log_history[-1]["loss"] if state.log_history else 0
        pct = int((current / self.total_steps) * 100)

        progress_bar_live.progress(min(pct, 100))
        status_text_live.markdown(
            f"**Fine-tuning progress:** {current} / {self.total_steps} steps — Loss: `{loss:.4f}`"
        )
