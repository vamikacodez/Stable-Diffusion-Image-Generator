#ai gpt for images workshop codes 

 
from diffusers import StableDiffusionPipeline
import torch

#loding the model
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype = torch.float16)
pipe = pipe.to("cuda")

#define any prompt
prompts = ["evening scene with waterfall with clear water which is flowing in a stream with small fishes and mini pebbles   ",]
images = []

#generating the two images 
for i,prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f'picture_{i}.jpg')
    images.append(image)

for i,prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f'picture_{i}.jpg')
    images.append(image)

import matplotlib.pyplot as plt
import matplotlib.image as mping

# assuming 'images' is the last containing image objects 
for i, image in enumerate(images):
    #save image to a file 
    image_path = f'picture_{i}.jpg'
    image.save(image_path)

    #load and display the image using matplotlib 
    img = mping.imread(image_path)
    plt.imshow(img)
    plt.title(f'image{i}')
    plt.show ()
