import random
import os
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    print(f"Model loaded: {model}")
    if hasattr(model, 'model_dir'):
        print(f"Attempting to inspect model_dir: {model.model_dir}")
    if hasattr(model, 'cache_dir'):
        print(f"Attempting to inspect cache_dir: {model.cache_dir}")
    # You can also try printing the model's __dict__ if it's a custom class and might reveal paths
    # print(f"Model __dict__: {model.__dict__}")
    return model


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
        print(f"Model loaded in generate: {model}")
        if hasattr(model, 'model_dir'):
            print(f"Attempting to inspect model_dir in generate: {model.model_dir}")
        if hasattr(model, 'cache_dir'):
            print(f"Attempting to inspect cache_dir in generate: {model.cache_dir}")
        # You can also try printing the model's __dict__ if it's a custom class and might reveal paths
        # print(f"Model __dict__ in generate: {model.__dict__}")

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())


with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
