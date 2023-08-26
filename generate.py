from datetime import datetime
import sys
import os
import torch
torch.set_num_threads(os.cpu_count()*2)
torch.set_num_interop_threads(os.cpu_count()*2)
from diffusers import DiffusionPipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

base = DiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
)
base.enable_attention_slicing(slice_size=1)
base.set_progress_bar_config(disable=True)
refiner = DiffusionPipeline.from_pretrained(
    refiner_id,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    use_safetensors=True,
)
refiner.enable_attention_slicing(slice_size=1)
refiner.set_progress_bar_config(disable=True)
if torch.backends.mps.is_available():
    print("🚀 mps 🚀 ")
    base = base.to("mps")
    refiner = refiner.to("mps")


def t2i(prompt="a roman woman at work on her laptop, fresco, from Pompeii", seed=42, n_steps=20, high_noise_frac=0.8, negative_prompt="wrong, ugly", guidance_scale=7.5):
    gen = torch.Generator().manual_seed(seed)
    image = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=gen,
        num_inference_steps=n_steps,
        height=512,
        width=512,
        denoising_end=high_noise_frac,
        output_type="latent",
        guidance_scale=guidance_scale,
    ).images
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=gen,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        guidance_scale=guidance_scale,
    ).images[0]
    return image


vegetables = [
    "avocado with the pit",
    "broccoli floret",
    "celery",
    "dandelion flower and greens",
    "eggplant",
    "sliced fig",
    "green beans",
    "hazelnut nutella on bread",
]


def clock_time_to_still_life_prompt(now):
    # day is 12 hours 0600-1800, night is 1800-0600+1, split into quarters, like the roman night watch, eight prompts total
    hour = int(now.strftime("%H"))
    is_daytime = hour >= 6 and hour < 18
    quarter = ((hour + 24 - 6) % 24) // 3
    veg = vegetables[quarter]
    return f"pen and watercolor drawing of {veg}"


if __name__ == "__main__":
    archive = "260106"
    os.makedirs(archive, exist_ok=True)
    hour = int(sys.argv[1])
    minute = int(sys.argv[2])
    seed = int(datetime.now().timestamp())
    prompt = clock_time_to_still_life_prompt(datetime(2023,1,1,hour,minute))
    image = t2i(
        prompt=prompt,
        seed=seed,
        negative_prompt="wrong, ugly, abstract, geometric, tiled, wallpaper"
    )
    image.save(f"{archive}/{prompt}.{hour:02d}{minute:02d}.{seed}.png")
    image.save("/tmp/beauty.png")
    with open("/tmp/beauty.txt", "w") as f: f.write(prompt)