import os
import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = ChatterboxVC.from_pretrained(DEVICE)
print(f"VC Model loaded: {model}")
if hasattr(model, 'model_dir'):
    print(f"Attempting to inspect VC model_dir: {model.model_dir}")
if hasattr(model, 'cache_dir'):
    print(f"Attempting to inspect VC cache_dir: {model.cache_dir}")
# print(f"VC Model __dict__: {model.__dict__}") # Optional: if further inspection is needed

def generate(audio, target_voice_path):
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
