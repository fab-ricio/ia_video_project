# Ce script génère une vidéo à partir d'un texte, sans voix.
# Prérequis :
# pip install diffusers transformers accelerate moviepy torch

from moviepy.editor import ImageSequenceClip
import os
from diffusers import StableDiffusionPipeline
import torch
import time

# Texte à transformer en images
texte = """
un petit garçon en chemise bleu parle à une fille avec chemise rouge. le petit garçon parle avec des signes de mains pendant 10 secondes
"""

scenes = [ligne.strip() for ligne in texte.strip().split('\n') if ligne.strip()]

# Créer un dossier pour les images
os.makedirs("scenes", exist_ok=True)
# Créer un dossier pour la vidéo
os.makedirs("video", exist_ok=True)

# Charger le pipeline Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
).to("cpu")

# Paramètres d'animation
frames_per_scene = 6  # moins d'images par scène pour accélérer
fps = 8  # images par seconde pour la vidéo

# Choix du mode : 'local' (images fixes) ou 'colab' (vraie vidéo IA)
mode = 'modelscope'  # 'local' pour images fixes, 'modelscope' pour vraie vidéo IA

if mode == 'modelscope':
    print("\n=== Pour générer une vraie vidéo animée IA avec ModelScope, copie ce code dans Google Colab : ===\n")
    print('''\
!pip install modelscope
!pip install imageio[ffmpeg]
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
pipe = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
prompt = "un petit garçon en chemise bleu parle à une fille avec chemise rouge, style dessin animé mignon"
result = pipe({'text': prompt})
video_path = result[OutputKeys.OUTPUT_VIDEO]
from IPython.display import Video
Video(video_path, embed=True)
''')
    exit()

# Générer les images animées
images = []
total_frames = len(scenes) * frames_per_scene
start_time = time.time()

for idx, scene in enumerate(scenes):
    for frame in range(frames_per_scene):
        prompt = (
            scene +
            f", cute cartoon animation, frame {frame+1}, mouvement, style dessin animé, couleurs vives"
        )
        frame_start = time.time()
        image = pipe(prompt, height=384, width=384, num_inference_steps=20).images[0]
        path = f"scenes/scene_{idx}_frame_{frame}.png"
        image.save(path)
        images.append(path)
        # Estimation du temps restant
        frames_done = idx * frames_per_scene + frame + 1
        elapsed = time.time() - start_time
        avg_time = elapsed / frames_done
        frames_left = total_frames - frames_done
        eta = int(avg_time * frames_left)
        mins, secs = divmod(eta, 60)
        print(f"Image {frames_done}/{total_frames} générée. Temps restant estimé : {mins:02d}:{secs:02d}")

# Assembler les images en vidéo animée
clip = ImageSequenceClip(images, fps=fps)
clip.write_videofile("video/video_generee.mp4", codec="libx264")

print("Vidéo animée générée : video/video_generee.mp4")