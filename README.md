## Overview
This repo creates a simple web app that edits an image based on a given prompt, which:
- Uses LLM to understand the prompt;
- Uses CLIPSeg and SAM to identify the region to be edited;
- Uses DALL-E to edit accordingly.
- Uses CLIP and BLIP captioning to evaluate the quality of the edited images.

## Get Started
Install the required packages.
```bash
pip install -r requirements.txt
```

Make the environment file.
```bash
cp .env.example .env
```

Run the ui.
```bash
gradio ui.py
```