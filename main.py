# Ce script génère une vidéo à partir d'un texte, sans voix.
# Prérequis :
# pip install diffusers transformers accelerate moviepy torch

from moviepy.editor import ImageSequenceClip
import os
from diffusers import StableDiffusionPipeline
import torch

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

# Générer les images animées
images = []
for idx, scene in enumerate(scenes):
    for frame in range(frames_per_scene):
        # Ajout d'une variation simple pour simuler le mouvement
        prompt = (
            scene +
            f", cute cartoon animation, frame {frame+1}, mouvement, style dessin animé, couleurs vives"
        )
        image = pipe(prompt, height=384, width=384, num_inference_steps=20).images[0]
        path = f"scenes/scene_{idx}_frame_{frame}.png"
        image.save(path)
        images.append(path)

# Assembler les images en vidéo animée
clip = ImageSequenceClip(images, fps=fps)
clip.write_videofile("video/video_generee.mp4", codec="libx264")

print("Vidéo animée générée : video/video_generee.mp4")