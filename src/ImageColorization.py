from matplotlib import pyplot as plt
from PIL import Image
from skimage import color
import numpy as np
import torch
from src.Models import get_model

model = get_model("../weights/res18_unet.pt", "../weights/final_initial_weights.pt")

def colorize(gray_image2):
    # Load image using Pillow and convert to RGB and then to numpy array
    gray_image = gray_image2.convert("RGB")
    gray_image = np.array(gray_image)
    lab_image = color.rgb2lab(gray_image)
    L = lab_image[:, :, 0]
    L = (L - 50.) / 100.
    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float()
    model.eval()
    with torch.no_grad():
        L = L.to(model.device)
        ab = model.net_G(L).cpu().numpy()
    ab = ab.transpose((0, 2, 3, 1))
    ab = ab[0] * 128.
    L = (L.cpu().numpy()[0] * 100.) + 50.
    L = L[0]
    lab_image = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized_image = color.lab2rgb(lab_image)

    colorized_image_save = Image.fromarray((colorized_image * 255).astype(np.uint8))
    colorized_image_save.save(f'../images/color.jpg')

    return colorized_image_save

if __name__ =='__main__' :
    colorized_image = colorize("../Images/lion.jpg", model)
    plt.imshow(colorized_image)
    plt.axis('off')  # to hide the axis
    plt.show()