from datetime import datetime, timedelta
import sys
import os
from PIL import Image
import io
import base64
import torch
torch.set_num_threads(os.cpu_count()*2)
torch.set_num_interop_threads(os.cpu_count()*2)
from diffusers import DiffusionPipeline, AutoencoderTiny, UniPCMultistepScheduler
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
vae_id = "sayakpaul/taesdxl-diffusers"

vae = AutoencoderTiny.from_pretrained(
    vae_id,
    use_safetensors=True,
    local_files_only=True,
)
vae.config.block_out_channels = vae.config.decoder_block_out_channels
base = DiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
    vae=vae,
    local_files_only=True,
)
base.enable_attention_slicing(slice_size=1)
base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
base.set_progress_bar_config(disable=True)
refiner = DiffusionPipeline.from_pretrained(
    refiner_id,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    use_safetensors=True,
    local_files_only=True,
)
refiner.enable_attention_slicing(slice_size=1)
refiner.set_progress_bar_config(disable=True)
#base.vae.enable_tiling()
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
    "honeycomb with bees",
]


def clock_time_to_still_life_prompt(now):
    # day is 12 hours 0600-1800, night is 1800-0600+1, split into quarters, like the roman night watch, eight prompts total
    hour = int(now.strftime("%H"))
    is_daytime = hour >= 6 and hour < 18
    quarter = ((hour + 24 - 6) % 24) // 3
    veg = vegetables[quarter]
    return f"pen and watercolor drawing of {veg}"


loading_image = Image.open(io.BytesIO(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYAQAAAAANfprEAAAAQUlEQVQI12P4/4+hgYlh/icQef8bCN3axnC3jOHcM4bn5xg+zWP4aQdCn+cxPDvHcP4Zw71vIFmgmqthCI3//wEA2oIh9bx+K3sAAAAASUVORK5CYII="
)))

if __name__ == "__main__":
    archive = "archive"
    os.makedirs(archive, exist_ok=True)
    if sys.argv[1] == "infinite":
        loading_image.save("/tmp/beauty.png")
        with open("/tmp/beauty.txt", "w") as f:
            f.write("believe us, they say, / it is a serious thing / just to be alive / on this fresh morning / in the broken world. / I beg of you, / do not walk by / without pausing / to attend to this / rather ridiculous performance.")
            # Oh do you have time
            #   to linger
            #     for just a little while
            #       out of your busy
            #
            # and very important day
            #   for the goldfinches
            #     that have gathered
            #       in a field of thistles
            #
            # for a musical battle,
            #   to see who can sing
            #     the highest note,
            #       or the lowest,
            #
            # or the most expressive of mirth,
            #   or the most tender?
            #     Their strong, blunt beaks
            #       drink the air
            #
            # as they strive
            #   melodiously
            #     not for your sake
            #       and not for mine
            #
            # and not for the sake of winning
            #   but for sheer delight and gratitude—
            #     believe us, they say,
            #       it is a serious thing
            #
            # just to be alive
            #   on this fresh morning
            #     in this broken world.
            #       I beg of you,
            #
            # do not walk by
            #   without pausing
            #     to attend to this
            #       rather ridiculous performance.
            #
            # It could mean something.
            #   It could mean everything.
            #     It could be what Rilke meant, when he wrote:
            #       𝘠𝘰𝘶 𝘮𝘶𝘴𝘵 𝘤𝘩𝘢𝘯𝘨𝘦 𝘺𝘰𝘶𝘳 𝘭𝘪𝘧𝘦.
            #
            # —Mary Oliver, Invitation
        target_latency = 2 * 3600
        current_steps = 5
        current_latency = 0
        for i in range(1000000000): # average number of total human heartbeats, ymmv
            prerender_time = datetime.now()
            hour = prerender_time.hour
            minute = 0
            seed = int(prerender_time.timestamp())
            prompt = clock_time_to_still_life_prompt(prerender_time + timedelta(seconds=current_latency))
            image = t2i(
                prompt=prompt,
                seed=seed,
                n_steps=current_steps,
                negative_prompt="wrong, ugly, abstract, geometric, tiled",
            )
            postrender_time = datetime.now()
            current_latency = postrender_time.timestamp() - prerender_time.timestamp()
            image.save(f"{archive}/{prompt}.{seed}.{prerender_time.strftime('%Y%m%dT%H%M%SZ')}.png")
            image.save("/tmp/beauty.png")
            caption = f"{prompt} - {i}, {current_steps} steps @ {int(current_latency)}s"
            with open("/tmp/beauty.txt", "w") as f: f.write(caption)
            steps_increment = 1 if current_latency < target_latency else -1
            current_steps = current_steps + steps_increment
    else:
        hour = int(sys.argv[1])
        minute = int(sys.argv[2])
        seed = int(datetime.now().timestamp())
        prompt = clock_time_to_still_life_prompt(datetime(2023,1,1,hour,minute))
        image = t2i(
            prompt=prompt,
            seed=seed,
            n_steps=100,
            negative_prompt="wrong, ugly, abstract, geometric, tiled, wallpaper"
        )
        image.save(f"{archive}/{prompt}.{hour:02d}{minute:02d}.{seed}.png")
        image.save("/tmp/beauty.png")
        with open("/tmp/beauty.txt", "w") as f: f.write(prompt)

