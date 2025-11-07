import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import clip
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
from segment_anything import sam_model_registry, SamPredictor
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from openai import OpenAI

import sys
sys.path.append("..")
import numpy as np
import json
import requests
import os
import heapq
import subprocess

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
project_id = "skills-network"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIPSeg Model
processor_clipseg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model_clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load SAM
# The URL of the SAM model parameters
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ac2ryJJQCwFcauPZLw9DRg/sam-vit-b-01ec64.pth"
# The desired output filename (optional, wget will infer if not provided)
file_name = "sam-vit-b-01ec64.pth"

if not os.path.exists(file_name):
    subprocess.run(["wget", url])

sam_checkpoint = "sam-vit-b-01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Load BLIP Captioning Model
processor_caption = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load CLIP Model
model_clip, preprocess_clip = clip.load("ViT-B/32", device="cpu")

# Load LLM
credentials = Credentials(
                url = "https://us-south.ml.cloud.ibm.com"
                )

# Get sample parameter values
sample_params = TextChatParameters.get_sample_params()

# Initialize the TextChatParameters object with the sample values
params = TextChatParameters(**sample_params)

model_chat = ModelInference(
    model_id='ibm/granite-3-3-8b-instruct',  #meta-llama/llama-3-3-70b-instruct
    credentials=credentials,
    project_id=project_id,
    params=params,
)

def caption(image_path):
    image = Image.open(image_path)
    inputs = processor_caption(images=image, return_tensors="pt")
    out = model_caption.generate(**inputs)

    caption = processor_caption.decode(out[0], skip_special_tokens=True)
    return caption

def similarity_score(original_caption, query, edited_image_path):
    image = preprocess_clip(Image.open(edited_image_path)).unsqueeze(0) 

    #Tokenize the text (caption)
    text = clip.tokenize([f"{original_caption}: {query}"]).to("cpu")

    #Encode both image and text into CLIP’s embedding space
    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(text)

    #Normalize embeddings (important for cosine similarity)
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    #Compute cosine similarity
    cosine_sim = (image_features @ text_features.T).item()

    return cosine_sim

def process_dalle_images(response, filename, image_dir):
    # save the images
    urls = [datum.url for datum in response.data]  # extract URLs
    images = [requests.get(url).content for url in urls]  # download images
    image_names = [f"{filename}_{i + 1}.png" for i in range(len(images))]  # create names
    filepaths = [os.path.join(image_dir, name) for name in image_names]  # create filepaths
    for image, filepath in zip(images, filepaths):  # loop through the variations
        with open(filepath, "wb") as image_file:  # open the file
            image_file.write(image)  # write the image to the file

    return filepaths

def fashion_design(query, image_path):
    #### Understand user's query
    error_flag = False
    prompt = f"""
    I give you a user query: {query}
    Fill in the blanks.
    Can you point to me (CONCISE): 
    (1) What needs to be changed: [BLANK]; 
    (2) Will be changed to what: [BLANK].

    Structure your answers to the blanks as a Python Dictionary, with keys '1' and '2'.
    """

    response = model_chat.chat(
    messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    )
    response_output = response['choices'][0]['message']['content']
    try:
        response_output = json.loads(response_output)
    except:
        error_warning = "Please format your query as: change A to B."
        error_flag = True
    
    if error_flag:
        return None, error_warning
    
    keyerr_flag = False
    try:
        to_be_changed = response_output["1"]
        will_change_to = response_output["2"]
    except:
        keyerr_flag = True
    
    if keyerr_flag:
        keys = list(response_output.keys())
        to_be_changed = response_output[keys[0]]
        will_change_to = response_output[keys[1]]

    #### Image Captioning
    caption_original = caption(image_path)

    #### CLIPSeg
    # Read as a uint8 tensor [C, H, W] in RGB (forces 3 channels; drops alpha if present)
    img_tensor = read_image(image_path, mode=ImageReadMode.RGB)

    # Convert to PIL.Image in RGB
    image_pil = to_pil_image(img_tensor)

    # Send the text_prompt and the image_pil to the model for prediction
    inputs = processor_clipseg(text=[to_be_changed], images=[image_pil], return_tensors="pt")
    with torch.no_grad():
        outputs = model_clipseg(**inputs)
        preds = torch.sigmoid(outputs.logits)  # mask probabilities

    # Convert the predicted mask to be a binary mask of only 0 or 1.
    mask = preds.squeeze().cpu().numpy()
    mask_binary = (mask > 0.05).astype(np.uint8)  # threshold mask

    # Generate the mask from the mask_binary
    mask_pil = Image.fromarray(mask_binary*255)
    rough_mask_np = np.array(mask_pil.convert("L")) > 128

    #### SAM
    predictor = SamPredictor(sam)
    numpy_img = torch.permute(img_tensor,[1,2,0]).numpy()
    predictor.set_image(numpy_img)

    w, h, _ = numpy_img.shape
    wm, hm = rough_mask_np.shape

    # Derive bounding box from rough mask
    y, x = np.where(rough_mask_np)
    bbox = np.array([min(x)/hm*h, min(y)/wm*w, max(x)/hm*h, max(y)/wm*w])  # [x0, y0, x1, y1]

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=True,
    )

    num_masks = len(masks)
    for i in range(num_masks):
        # We'll now reverse the mask so that it is clear and everything else is white
        chosen_mask = masks[i]
        chosen_mask = chosen_mask.astype("uint8")
        chosen_mask[chosen_mask != 0] = 255
        chosen_mask[chosen_mask == 0] = 1
        chosen_mask[chosen_mask == 255] = 0
        chosen_mask[chosen_mask == 1] = 255

        # create a base blank mask
        width = 1024
        height = 1024
        mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # create an opaque image mask

        # Convert mask back to pixels to add our mask replacing the third dimension
        pix = np.array(mask)
        pix[:, :, 3] = chosen_mask

        # Convert pixels back to an RGBA image and display
        new_mask = Image.fromarray(pix, "RGBA")
        new_mask.save(f"mask_candidates/mask_option{i}.png")

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_image_urls = []
    n = 3
    print('Editing....')
    for i in range(num_masks-1):
        mask_path = f"mask_candidates/mask_option{i}.png"
        response = client.images.edit(
            model='dall-e-2',
            image=open(image_path, "rb"),  # from the generation section
            mask=open(mask_path, "rb"),  # from right above
            prompt=will_change_to,  # provide a prompt to fill the space
            n=n,
            size="1024x1024",
            response_format="url",
        )
        edit_filepaths = process_dalle_images(response, f"edits_mask{i}", 'edited_images')

        for edit_filepath in edit_filepaths:
            sim_score = similarity_score(caption_original, query, edit_filepath)
            if sim_score > 0.29:
                heapq.heappush(all_image_urls,(-sim_score,edit_filepath))
    
    print('Finalizing....')
    finalized_urls = []
    while len(all_image_urls) > 0:
        _, pth = heapq.heappop(all_image_urls)
        finalized_urls.append(pth)

    print('Out....')
    if len(finalized_urls) == 0:
        message = 'Edited, but none of the generated images satisfy your request... Please try again by changing your prompt.'
    else:
        message = 'Edited Sucessfully! (Please note that the models are imperfect, so the image quality can vary significantly)'

    return finalized_urls, message


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Fashion Design Assistant")
    gr.Markdown("#### IMPORTANT: Make sure your uploaded image is less than 4 MB!")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload an Image", type="filepath")
            query_input = gr.Textbox(label="Enter your prompt", placeholder="e.g., change the shirt to a sweater")
            submit_button = gr.Button("DESIGN!", variant="primary")

        with gr.Column():
            gr.Markdown("### AI Fashion Gallery")
            gr.Markdown("#### Each editing takes about 120 seconds (2 minutes).")
            gr.Markdown('#### If you have ERROR, please retry.')
            gallery_output = gr.Gallery(
                label="AI Designs",
                show_label=False,
                elem_id="gallery",
                columns=[3],
                rows=[1],
                object_fit="contain",
                height="auto"
            )
            message_output = gr.Markdown(label="Status Message")

    # When button is clicked, call function f
    submit_button.click(
        fn=fashion_design,
        inputs=[query_input,image_input],
        outputs=[gallery_output,message_output]
    )

    # 🔹 Example (image, query) pairs
    examples = [
        ["example_images/cat-blue-cloth.png", "change the blue shirt to a shirt with pumpkins"],
        ["example_images/lady-white-shirt.png", "change the white shirt to a pink sweater"]
    ]

    # Built-in Gradio Examples block
    gr.Examples(
        examples=examples,
        inputs=[image_input, query_input],
        label="Try These Example Image–Prompt Pairs"
    )


# --------------------------------------------------
# Launch the app
# --------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)
