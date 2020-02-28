
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

def save_plot(name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))


# Task 4b
indices = [14, 26, 32, 49, 52]

fig = plt.figure(figsize=(20, 4))

for i in range(len(indices)):
    plt.subplot(2,5,i+1)
    plt.title('Filter ' + str(indices[i]))
    plt.imshow(torch_image_to_numpy(first_conv_layer.weight[indices[i]]))

    plt.subplot(2, 5, i+6)
    plt.imshow(activation[0, indices[i]].data, cmap="gray")

save_plot('task4b')
plt.show()


# Task 4c
fig = plt.figure(figsize=(20, 4))

for i in range(10):
    image = Image.open("images/zebra.jpg")
    image = image_transform(image)[None]

    for child in model.children():
        if isinstance(child, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            # Break when we come to the second last layer
            break
        image = child(image)

    plt.subplot(2,5,i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(image[0,i].data, cmap="gray")

fig.subplots_adjust(hspace=0.5)
save_plot('task4c')
plt.show()