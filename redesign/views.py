from django.shortcuts import render
from .forms import DesignForm
from .models import DesignRequest

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import MidasDetector, CannyDetector

import torch
from PIL import Image
import os, uuid
from diffusers import DPMSolverMultistepScheduler
import random
import hashlib

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32


# =========================
# LOAD CONTROLNETS
# =========================
depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=dtype
)

canny_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype
)

# =========================
# LOAD PIPELINES
# =========================
pipe_depth = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=depth_controlnet,
    torch_dtype=dtype
).to(device)

pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[depth_controlnet, canny_controlnet],
    torch_dtype=dtype
).to(device)


# =========================
# OPTIMIZATION
# =========================
def optimize(pipe):
    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    pipe.enable_attention_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    return pipe


pipe_depth = optimize(pipe_depth)
pipe_canny = optimize(pipe_canny)


# =========================
# LOAD DETECTORS
# =========================
depth_estimator = MidasDetector.from_pretrained("lllyasviel/Annotators")
canny_detector = CannyDetector()

if device.type == "cuda":
    depth_estimator.to(device)

print("Running on:", device)


# =========================
# FIXED HASH FUNCTION
# =========================
def get_hash(prompt, scene_type, uploaded_file):
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    return hashlib.md5(
        (prompt + scene_type).encode() + img_bytes
    ).hexdigest()


# =========================
# MAIN VIEW
# =========================
def redesign_home(request):
    output_path = None

    if request.method == 'POST':
        form = DesignForm(request.POST, request.FILES)

        if form.is_valid():

            obj = form.save(commit=False)

            uploaded_file = request.FILES['original_image']

            # ✅ CREATE HASH FIRST
            request_hash = get_hash(
                obj.prompt,
                obj.scene_type,
                uploaded_file
            )

            # ✅ CHECK CACHE
            old = DesignRequest.objects.filter(request_hash=request_hash).first()

            if old and old.output_image:
                return render(request, "redesign/form.html", {
                    "form": form,
                    "output_path": old.output_image
                })

            # save new request
            obj.request_hash = request_hash
            obj.save()

            # =========================
            # LOAD IMAGE
            # =========================
            input_image = Image.open(obj.original_image.path).convert("RGB")
            input_image = input_image.resize((512, 512))

            user_prompt = obj.prompt

            # =========================
            # INTERIOR / EXTERIOR
            # =========================
            if obj.scene_type == "interior":

                pipe = pipe_depth
                control_image = depth_estimator(input_image)

                system_prompt = (
                    "realistic interior design, keep same room layout, "
                    "same perspective, same camera angle, correct proportions, "
                    "natural lighting, soft shadows, well arranged furniture"
                )

                conditioning_scale = 0.6
                strength = 0.5
                guidance_scale = 6

            else:

                pipe = pipe_canny
                control_image = [
                    depth_estimator(input_image),
                    canny_detector(input_image)
                ]

                system_prompt = (
                    "photorealistic modern exterior, sleek lines, minimalist style, "
                    "large glass windows, neutral tones, natural daylight, "
                    "architectural photography, wide angle, full facade"
                )

                conditioning_scale = [0.6, 0.3]
                strength = 0.45
                guidance_scale = 7.5

            # =========================
            # PROMPT
            # =========================
            prompt = f"{user_prompt}. {system_prompt}"

            negative_prompt = (
                "low quality, blurry, distorted, deformed, "
                "wrong perspective, duplicate objects, cropped, watermark, text"
            )

            # =========================
            # GENERATION
            # =========================
            steps = 30
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)

            if obj.scene_type == "exterior":
                image_input = [input_image, input_image]
            else:
                image_input = input_image

            with torch.inference_mode():
                output_image = pipe(
                    prompt=prompt,
                    image=image_input,
                    control_image=control_image,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    generator=generator,
                    controlnet_conditioning_scale=conditioning_scale
                ).images[0]

            # =========================
            # SAVE OUTPUT
            # =========================
            filename = f"{uuid.uuid4()}.png"
            output_dir = "media/outputs/"
            os.makedirs(output_dir, exist_ok=True)

            output_image.save(os.path.join(output_dir, filename), optimize=True)

            obj.output_image = f"outputs/{filename}"
            obj.save()

            output_path = obj.output_image

    else:
        form = DesignForm()

    return render(request, "redesign/form.html", {
        "form": form,
        "output_path": output_path
    })