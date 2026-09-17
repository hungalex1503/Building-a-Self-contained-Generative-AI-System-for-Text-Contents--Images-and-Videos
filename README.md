# Generative AI Applications for Text, Images & Video

A learning project exploring generative AI through a local conversational application and separate notebooks for image-conditioned writing, image generation, and video generation.

The repository combines a **Streamlit + Ollama chatbot** with **Gemini API** and **Hugging Face Diffusers** workflows. The components currently run independently; image and video generation are not integrated into the chatbot interface.

## Features

- **Local text generation:** chat with Phi-3 Mini through Ollama and display responses incrementally.
- **Custom personas:** change the system prompt from the sidebar to control the assistant's writing style and role.
- **Conversation management:** maintain conversation context within a Streamlit session, clear the chat, and download its text history.
- **Image-conditioned writing:** provide an image and instructions to Gemini 2.5 Flash to generate blog content.
- **Text-to-image generation:** generate images with Stable Diffusion v1.5.
- **Text-to-video generation:** generate short clips with ModelScope and export them as MP4 files.
- **Video memory configuration:** use FP16 weights and CPU offloading in the video notebook.

## Components

| Component | Model / service | Main libraries | Entry point |
| --- | --- | --- | --- |
| Local chatbot | `phi3:mini` through Ollama | Streamlit, Ollama Python client | [`main/app.py`](main/app.py) |
| Image-conditioned writing | `gemini-2.5-flash` | `google-generativeai`, Pillow | [`main/Exercise2_DMCP (1).ipynb`](main/Exercise2_DMCP%20%281%29.ipynb) |
| Image generation | `stable-diffusion-v1-5/stable-diffusion-v1-5` | PyTorch, Diffusers, Transformers, Accelerate | [`main/Tutorial_17.ipynb`](main/Tutorial_17.ipynb) |
| Video generation | `damo-vilab/text-to-video-ms-1.7b` | PyTorch, Diffusers, Accelerate | [`main/Tutorial18_Hung.ipynb`](main/Tutorial18_Hung.ipynb) |

All models are used for inference. This repository does not implement model training, fine-tuning, RAG, or an autonomous tool-using agent.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and setup instructions |
| `main/app.py` | Streamlit chatbot and Ollama integration |
| `main/Exercise2_DMCP (1).ipynb` | Gemini text and image-input experiments |
| `main/Tutorial_17.ipynb` | Stable Diffusion image generation |
| `main/Tutorial18_Hung.ipynb` | ModelScope video generation and MP4 export |

## Run the Local Chatbot

### 1. Clone the repository

```bash
git clone https://github.com/hungalex1503/Building-a-Self-contained-Generative-AI-System-for-Text-Contents--Images-and-Videos.git
cd Building-a-Self-contained-Generative-AI-System-for-Text-Contents--Images-and-Videos
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the chatbot dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install streamlit ollama
```

### 3. Prepare Ollama

Install the Ollama application separately, ensure its local service is running, and download the model configured in `main/app.py`:

```bash
ollama pull phi3:mini
```

If the service is not already running, start it in a separate terminal:

```bash
ollama serve
```

The Python package alone does not install the Ollama application or download model weights.

### 4. Start Streamlit

```bash
python -m streamlit run main/app.py
```

Open the local URL printed by Streamlit. Enter a message, optionally change the persona in the sidebar, and use the history controls to clear or download the conversation.

**Image upload status:** the interface includes an image uploader and code for attaching image bytes. However, the configured model is `phi3:mini`, while a source comment refers to `llava-phi3`. Treat the current default as a text chatbot. Image understanding requires selecting, downloading, and testing an appropriate vision-capable model before relying on this feature.

## Run the Notebooks

The notebooks contain notebook-specific installation commands and were written for interactive use, including Google Colab. Open them in Colab or a local Jupyter environment and review their setup cells before execution.

For a local notebook environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

Model downloads require internet access. The diffusion notebooks are configured for GPU execution; they are not ready-to-run CPU-only examples. Dependency versions are not pinned, so the existing code may require adjustments for your environment.

### Image-conditioned writing with Gemini

Open `main/Exercise2_DMCP (1).ipynb`.

1. Install the notebook's dependencies: `google-generativeai` and Pillow.
2. Replace the hard-coded credential setup with a secret supplied by your environment.
3. Replace `/content/beach.png` with the path to your input image.
4. Edit the prompt to match the image and desired writing task.
5. Run the generation cells.

For example, replace the notebook's key assignment and configuration cell with:

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
```

Set `GEMINI_API_KEY` through your environment or notebook secret manager before running this cell. This is a recommended replacement; the checked-in notebook has not been modified by these README instructions.

The notebook uses `gemini-2.5-flash` and the `google-generativeai` package as written. API access, model availability, and SDK compatibility must be checked in your own environment. Text and image inputs in this workflow are sent to Google's API; this component does not run locally.

**Credential handling:** do not commit API keys or print them in notebook outputs. If a real key has already been committed, revoke or rotate it and remove it from Git history. Removing it only from the newest revision is insufficient.

### Image generation with Stable Diffusion

Open `main/Tutorial_17.ipynb`.

1. Select a suitable GPU runtime and install the notebook dependencies.
2. Load `stable-diffusion-v1-5/stable-diffusion-v1-5`.
3. Edit the text prompt.
4. Run the inference and display cells.

The current code requests CUDA placement and `bfloat16`. Choose a precision supported by your GPU and adjust the loading arguments if required by your installed Diffusers version. The notebook displays the generated image; it does not currently provide an integrated image-generation web interface.

### Video generation with ModelScope

Open `main/Tutorial18_Hung.ipynb` and run it in a compatible GPU environment.

The current configuration is:

| Setting | Value |
| --- | --- |
| Model | `damo-vilab/text-to-video-ms-1.7b` |
| Precision | `torch.float16` |
| Scheduler | `DPMSolverMultistepScheduler` |
| Memory handling | `enable_model_cpu_offload()` |
| Inference steps | 25 |
| Frames | 16 |
| Export frame rate | 8 FPS |
| Output | `huggingface_video.mp4` |

Edit `prompt` and `negative_prompt` before generation. With 16 frames exported at 8 FPS, the configured clip duration is approximately 2 seconds. This is playback duration, not generation latency. The video is exported to the notebook's working directory and displayed inline.

## Implementation Notes

- `ContentGenerator.generate_text_stream()` prepares the conversation messages and yields text chunks from `ollama.chat(..., stream=True)`.
- The Streamlit interface stores messages in `st.session_state`, adds the selected system prompt, and passes previous turns as context.
- Chat history is session-based; the application does not use a persistent conversation database.
- Streaming is implemented for chatbot text. It does not establish real-time image or video generation.
- FP16 and CPU offloading are configured in the video workflow, but their memory and latency effects have not been benchmarked in this repository.

## Current Limitations

- The chatbot and generation notebooks are separate workflows.
- The chatbot's image-upload feature needs a compatible model configuration and functional validation.
- The Gemini workflow depends on an external API.
- Dependencies and hardware requirements are not fully pinned or documented through reproducible benchmarks.
- There are no quantitative generation-quality, latency, or peak-memory evaluations.
- The project does not include production deployment, authentication, automated tests, or model fine-tuning.

## Planned Improvements

The following items are proposed future work, not completed features:

- Add tested dependency versions and reproducible environment instructions.
- Align the image-upload interface with a validated vision model.
- Integrate text, image, and video generation into a common interface.
- Add example outputs and a short application demo.
- Measure generation latency and peak GPU memory on documented hardware.
- Add input validation and clearer handling of model-loading and inference errors.

## Author

**Truong Tuan Hung** — [GitHub](https://github.com/hungalex1503)

## Acknowledgments and Licensing

This project uses Ollama, Streamlit, PyTorch, Hugging Face Diffusers, and Google's Gemini API, together with pretrained models from their respective providers. Review each model's license and each service's terms before reuse or distribution.

No project-level license file is currently included in this repository.
