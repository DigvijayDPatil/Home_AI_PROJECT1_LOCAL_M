from django.shortcuts import render
from .forms import DesignForm
from .models import DesignRequest
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os, uuid
from diffusers import DPMSolverMultistepScheduler


# Load model once (CPU)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # CPU inference
pipe.enable_attention_slicing()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None

def redesign_home(request):
    output_path = None
    if request.method == 'POST':
        form = DesignForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save()
            input_image = Image.open(obj.original_image.path)
            prompt = obj.prompt

            # Resize to reduce CPU load
            input_image = input_image.convert("RGB")
            input_image.thumbnail((512,512))
            # Generate redesigned image (CPU is slow)
            with torch.no_grad():
                output_image = pipe(
                prompt=prompt,
                image=input_image,
                strength=0.5,
                guidance_scale=8.5,
                num_inference_steps=30  # fewer steps = faster
            ).images[0]

            # Save output
            filename = f"{uuid.uuid4()}.png"
            output_dir = "media/outputs/"
            os.makedirs(output_dir, exist_ok=True)
            output_image.save(os.path.join(output_dir, filename),quality=95)
            obj.output_image = f"outputs/{filename}"
            obj.save()
            output_path = obj.output_image
    else:
        form = DesignForm()
    return render(request, "redesign/form.html", {"form": form, "output_path": output_path})