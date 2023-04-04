"""
# https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B
"""
import gc
import os
from subprocess import Popen, PIPE
import sys
import time
from typing import Optional
import uuid

from PIL import Image
import cv2
import torch
import transformers as tf
from flask import Flask, render_template, request, jsonify, Response
from flask_sse import sse

TEMP_IMAGE_SAVE_PATH = "tmp.jpg"

examples = [
    ["cute_pics.png", "Q: What is unusual about this picture? A:"],
    ["", '''Answer the following question by reasoning step-by-step.
The Doge had 23 bones. If he used 6 for breakfast and stole 8 from Cheems, how many bones does Doge now have?'''],
    ["sunset.jpg", "What time of the day is it?"],
    ["forbidden.webp", "What dynasty was this built in?"],
    ["pizza.jpg", "What ingredients do I need to cook it?"] # Generate more tokens.
    ["flower.jpg", "What is the name of this flower?"],
    ["", '''Which of these sentences doesn't make sense?
Options:
- Sentence A: "He raised a microwave with his hands."
- Sentence B: "He lifted a truck with his hands."'''],
]

def download_if_not_exists(filename, url):
    """Download a file from a URL if it doesn't exist in the current directory."""
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}")
        os.system(f"wget {url} -O {filename}")
    return filename


def process_input(input_file):
    print(input_file.name)
    extension = input_file.name.split(".")[-1]
    # Check if the input is an image.
    if extension in ["jpg", "jpeg", "png"]:
        save_temp_file_as_image(input_file, TEMP_IMAGE_SAVE_PATH)
        # buffered = io.BytesIO()
        # input_file.save(buffered, format="JPEG")
        # buffered.seek(0)
        image = Image.open(input_file)
        print("Image size:", image.size)
        # Save as a temporary image file `tmp.jpg`
        image.save(TEMP_IMAGE_SAVE_PATH)
        return True, "image"
    # Check if the input is a video
    elif extension in ["mp4", "mov", "avi"]:
        with open("temp_video.mp4", "wb") as video_file:
            video_file.write(input_file)
        video_capture = cv2.VideoCapture("temp_video.mp4")
        success, frame = video_capture.read()
        # Get the FPS of the video.
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))

        # Calculate the total number of frames in the first second
        frames_in_first_second = min(fps, int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Read the frames and store them in a list
        frames = []
        for i in range(frames_in_first_second):
            success, frame = video_capture.read()
            if not success:
                break
            frames.append(frame)

        # Extract the middle frame from the list
        middle_frame = frames[len(frames) // 2]
        # Save the middle frame as a temporary image file `tmp.jpg`
        cv2.imwrite(TEMP_IMAGE_SAVE_PATH, middle_frame)
        return True, "video"
    else:
        raise Exception(f"File type {extension} not supported.")
    return False, "None"

def evaluate(media_file, input_text, token_count, temperature, top_k, top_p, repeat_penalty, seed):
    # for i in range(10):
    #     yield f"Hello {i}"
    # yield "<EOS>"
    print(media_file)
    if media_file is not None:
        if isinstance(media_file, str):
            media_file = media_file.strip()
            if media_file == "":
                media_file = None
        else:
            raise Exception(f"media_file is not a string: {media_file}")

    # print(process_input(media_file))
    assert input_text.strip() != "", "Input text cannot be empty."
    input_ids = all_model_specific_stuff["tokenizer"].batch_encode_plus(
        [input_text.strip()], return_tensors="pt").input_ids
    inputs_embeds = all_model_specific_stuff["t5_embeddings"](input_ids)

    if media_file is not None:
        # Process the image from the media file
        print("Processing image...")
        image_raw = Image.open(f'static/images/{media_file}')
        image_processed = all_model_specific_stuff["processor"](
            image_raw, return_tensors="pt")["pixel_values"]
        image_embeds = all_model_specific_stuff["blip2_model"].vision_model(
            image_processed, return_dict=True).last_hidden_state
        yield "Processed image."

        # Get the query tokens
        print("\n\n")
        print("Getting query tokens...", image_embeds.shape)
        query_tokens = all_model_specific_stuff["blip2_model"].query_tokens.expand(
            image_embeds.shape[0], -1, -1)
        query_outputs = all_model_specific_stuff["blip2_model"].qformer(
            query_embeds=query_tokens, encoder_hidden_states=image_embeds,
            return_dict=True).last_hidden_state
        yield "Converted image."

        # Get the language model inputs
        print("\n\n")
        print("Getting language model inputs...", query_outputs.shape)
        language_model_inputs = all_model_specific_stuff["blip2_model"].language_projection(
            query_outputs)
        # Concatenate query embeddings with prompt embeddings
        print(all_model_specific_stuff["t5_embeddings"], input_ids.shape)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1) # pylint: disable=no-member

    yield "Loading the model."
    print("Starting Generation...", inputs_embeds.shape)
    _ = open("tmp_file.txt", 'w+').write(
        " ".join([str(round(x, 8)) for x in inputs_embeds.flatten().tolist()]))
     
    if os.name == 'nt':
        main_file = "./cpp/main.exe"
    else:
        main_file = "./cpp/main"
        
    command = [main_file, 't5v1_1', "-m", 'models/int4-fixed-zero.bin',
               "--prompt", "tmp_file.txt",
               "--seed", str(seed),
               "--threads", "8",
               "--n_predict", str(token_count),
               "--top_p", str(top_p),
               "--top_k", str(top_k),
               "--temp", str(temperature),
               "--repeat_penalty", str(repeat_penalty),
               ]
    print(" ".join(command))

    process = Popen(command, stdout=PIPE, stderr=PIPE)
    tokens_ids_so_far = []
    has_generation_begun = False
    token_id_buffer = ""
    all_stdout_so_far = ""
    out_str = ""
    for c in iter(lambda: process.stdout.read(1), b""):
        all_stdout_so_far += c.decode('utf-8')

        if not has_generation_begun:
            to_print = c.decode('utf-8')
        else:
            if ' ' in c.decode('utf-8') and token_id_buffer.strip():
                # We have a token id
                token_id = int(token_id_buffer.strip())
                tokens_ids_so_far.append(token_id)
                token_str = all_model_specific_stuff["tokenizer"].decode(tokens_ids_so_far)
                token_id_buffer = ""
                # Call the streaming output hooks
                out_str = token_str
                # print("\n", tokens_ids_so_far)
                # print("\n", out_str)
                yield out_str
                to_print = token_str
            else:
                token_id_buffer += c.decode('utf-8')
        if to_print:
            print(to_print, end="")
            to_print = ""
            sys.stdout.flush()
        if '<|BEGIN> ' in all_stdout_so_far:
            has_generation_begun = True
        # Check if the line is empty or matches the end marker
        if '<END|>' in all_stdout_so_far:
            print("\n---------------------\n")
            break

    gc.collect()
    yield out_str.strip()

print("I am loading the model...")

class Blip2ForConditionalGeneration(tf.Blip2PreTrainedModel):
    config_class = tf.Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: tf.Blip2Config):
        super().__init__(config)

        self.vision_model = tf.Blip2VisionModel(config.vision_config)

        self.query_tokens = torch.nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = tf.Blip2QFormerModel(config.qformer_config)

        self.language_projection = torch.nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def _tie_weights(self):
        return

    def forward(self,
                pixel_values: torch.FloatTensor,
                input_ids: torch.FloatTensor,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values=pixel_values,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states,
                                           return_dict=return_dict)
        image_embeds = vision_outputs[0]

        # Step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens,
                                     encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=image_attention_mask,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     return_dict=return_dict)
        query_output = query_outputs[0]

        # Step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        return inputs_embeds

# Make `models` directory if it doesn't exist
if not os.path.exists("models"):
    os.mkdir("models")

# If config.json doesn't exist, download it from https://huggingface.co/ayushk4/smol-gpt4/resolve/main/config.json
if not os.path.exists("models/config.json"):
    download_if_not_exists("models/config.json",
                            "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/config.json")

blip2_config = tf.Blip2Config.from_pretrained("models/config.json")
blip2_model = Blip2ForConditionalGeneration._from_config(blip2_config)

# Get int4-fixed-zero.bin (`language model weights)
download_if_not_exists("models/int4-fixed-zero.bin",
                       "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/int4-fixed-zero.bin")

# TODO: Switch to safeTensors
# Get bert.bin (Bert model weights)
blip2_model.qformer.load_state_dict(torch.load(download_if_not_exists(
    "models/bert.bin", "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/bert.bin")))
# Get query-tokens.bin (Query tokens embeddings)
blip2_model.query_tokens.data = torch.load(download_if_not_exists(
    "models/query-tokens.bin", "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/query-tokens.bin")).float()
# Get t5_project.bin (T5 projection weights)
blip2_model.language_projection = torch.load(download_if_not_exists(
    "models/t5_project.bin", "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/t5_project.bin")).float()
# Get vision-model.bin (Vision model weights)
blip2_model.vision_model.load_state_dict(torch.load(download_if_not_exists(
    "models/vision-model.bin", "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/vision-model.bin")))

# Get t5_embed.bin (T5 embeddings)
# t5_embeddings = torch.nn.Embedding(32128, 4096)
t5_embeddings = torch.load(download_if_not_exists(
    "models/t5_embed.bin", "https://huggingface.co/ayushk4/smol-gpt4/resolve/main/t5_embed.bin")).float()

# Convert all blip2_model parameters to fp32
for param in blip2_model.parameters():
    param.data = param.data.float()
    param.requires_grad = False
t5_embeddings.weight.data = t5_embeddings.weight.data.float()
t5_embeddings.requires_grad = False
blip2_model.eval()
t5_embeddings.eval()

tokenizer = tf.AutoTokenizer.from_pretrained('google/flan-t5-xxl')

processor = tf.AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")

all_model_specific_stuff = {
    "blip2_model": blip2_model, "t5_embeddings": t5_embeddings,
    "tokenizer": tokenizer, "processor": processor}


###############################################################################
##################################### Flask app ###############################
###############################################################################

app = Flask(__name__)
app.register_blueprint(sse, url_prefix='/stream')

@app.route('/')
def index():
    print("Hello world")
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        print(image)
        image.save('static/images/tmp.jpg')

        return jsonify({'success': True, 'image_path': 'tmp.jpg'})

sse_connections = {}

@app.route('/generate_text', methods=['POST'])
def generate_text():
    if request.method == 'POST':
        print(request)
        json = request.get_json()
        image_path = json['image_path']
        input = json['input']
        max_tokens = json['max_tokens']
        temperature = json['temperature']
        top_k = json['top_k']
        top_p = json['top_p']
        repeat_penalty = json['repeat_penalty']
        seed = json['seed']
        if image_path is not None and image_path.strip() != "":
            print(image_path)
            image_shape = Image.open(f'static/images/{image_path}').size
            print(image_shape)
        else:
            image_path = None
            image_shape = None
        print({
            'success': True,
            'output': f"image_shape: {image_shape}, image_path: {image_path}, input: {input}, max_tokens: {max_tokens}, temperature: {temperature}, top_k: {top_k}, top_p: {top_p}, repeat_penalty: {repeat_penalty}, seed: {seed}"})

        # Create a unique connection ID and store the data in the dictionary
        connection_id = str(uuid.uuid4())
        sse_connections[connection_id] = {
            'image_path': image_path,
            'input': input,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repeat_penalty': repeat_penalty,
            'seed': seed
        }

        # Return the connection ID in the response
        return jsonify(sseConnectionId=connection_id)

@app.route('/generate_text_sse', methods=['GET'])
def generate_text_sse():
    """SSE endpoint that streams the generated text back to the client."""
    connection_id = request.args.get('connection_id')
    if connection_id in sse_connections:
        data = sse_connections[connection_id]
        image_path = data['image_path']
        input = data['input']
        max_tokens = data['max_tokens']
        temperature = data['temperature']
        top_k = data['top_k']
        top_p = data['top_p']
        repeat_penalty = data['repeat_penalty']
        seed = data['seed']

        def generate_tokens_sse():
            for full_str in evaluate(image_path, input, max_tokens, temperature, top_k, top_p,
                                     repeat_penalty, seed):
                print('\n', full_str)
                yield f'data: {full_str}\n\n'
                # time.sleep(0.05) # Adjust this value to control the rate of sending events
            print("\nDone generating")
            yield "data: [DONE]"

        # Remove the connection from the dictionary after using it
        del sse_connections[connection_id]

        return Response(generate_tokens_sse(), mimetype="text/event-stream")
    else:
        return "Invalid connection ID", 400


if __name__ == '__main__':
    # run at port 4241
    app.run(debug=True, port=4241)
