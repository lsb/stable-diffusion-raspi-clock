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


def t2i(prompt="a roman woman at work on her laptop, fresco, from Pompeii", seed=42, n_steps=20, high_noise_frac=0.8, negative_prompt="wrong, ugly"):
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
    ).images
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=gen,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    return image


animals = [
    "zebra",
    "armadillo",
    "bison",
    "capybara",
    "dragonfly",
    "elephant",
    "flamingo",
    "giraffe",
    "hippopotamus",
    "iguana",
    "jaguar",
    "kangaroo",
    "llama",
    "monkey",
    "narwhal",
    "octopus",
    "panda",
    "quail",
    "raccoon",
    "snakes",
    "turtle",
    "unicorn",
    "vulture",
    "worm",
]

vehicles = [
    "in a zeppelin",
    "in an ambulance",
    "on the bus",
    "in a chariot",
    "on a dump truck",
    "in an elevator",
    "in a fire truck",
    "in a golf cart",
    "in a helicopter",
    "at an ice cream truck",
    "in a jeep",
    "in a kayak",
    "in a limousine",
    "on a motorcycle",
    "in a navy submarine",
    "on an ocean liner",
    "on a plane",
    "on a quad bike",
    "in a rocket",
    "in a stroller",
]

media = [
    "etching by durer",
    "award winning photograph",
    "ukiyo-e",
]


def clock_time_to_animal_prompt(now):
    hour = int(now.strftime("%H"))
    minute = int(now.strftime("%M"))
    twelve_hour_time = now.strftime("%I %M %p")
    return f"{animals[hour]}, {vehicles[minute//3]}, at {twelve_hour_time}, {media[minute%3]}"


vegetables = [
    "avocado with the pit",
    "broccoli floret",
    "celery",
    "dandelion flower and greens",
]


daynight = [
    "in the sun",
    "under moonlight",
]


vegetable_media = [
    "watercolor"
]

def clock_time_to_still_life_prompt(now):
    # day is 12 hours 0600-1800, night is 1800-0600+1, split into quarters, like the roman night watch, eight prompts total
    hour = int(now.strftime("%H"))
    is_daytime = hour >= 6 and hour < 18
    quarter = ((hour + 24 - 6) % 12) // 3
    veg = vegetables[quarter]
    tod = daynight[0 if is_daytime else 1]
    return f"{veg} {tod}, {vegetable_media[0]}"



def save_image_for_time(hour, minutes, now=None):
    if now == None:
        now = datetime.now().timestamp()
    prompt = time_to_prompt(hour, minutes)
    print(prompt)
    img = render_prompt(prompt)
    filename = sanitize_alnum(f"sd15 {hour:02d} {minutes:02d} {prompt} {now}.jpg")
    img.save(filename)


if __name__ == "__main__":
    archive = "tempsperdu"
    os.makedirs(archive, exist_ok=True)
    hour = int(sys.argv[1])
    minute = int(sys.argv[2])
    seed = int(datetime.now().timestamp())
    prompt = clock_time_to_still_life_prompt(datetime(2023,1,1,hour,minute))
    image = t2i(
        prompt=prompt,
        seed=seed,
    )
    image.save(f"{archive}/{prompt}.{hour:02d}{minute:02d}.{seed}.png")
    image.save("/tmp/beauty.png")