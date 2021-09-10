import torch 
import torchvision
import torchvision.transforms as transforms
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_CLASSES = json.load(open('xai_isact_2021/imagenet-labels.json'))

read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN,
                          std=IMAGENET_STD),
    lambda x: torch.unsqueeze(x, 0)
])

def tensor_to_numpy(image_tensor):
    inp = image_tensor[0].clone()
    for channel in range(3):
        inp[channel] = inp[channel]*IMAGENET_STD[channel] + IMAGENET_MEAN[channel]
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp

def tensor_imshow(image_tensor, logits):
    inp = tensor_to_numpy(image_tensor)
    probs = torch.nn.functional.softmax(logits, 1)[0].cpu().detach().numpy()
    sorted_classes = np.argsort(-1 * logits.cpu().detach().numpy(), axis=1)[0]
    top_3_classes = [sorted_classes[0], sorted_classes[1], sorted_classes[2]]
    title_string = IMAGENET_CLASSES[top_3_classes[0]] + ': ' + str(probs[top_3_classes[0]]) + ', ' \
        + IMAGENET_CLASSES[top_3_classes[1]] + ': ' + str(probs[top_3_classes[1]])  + ', ' \
        + IMAGENET_CLASSES[top_3_classes[2]] + ': ' + str(probs[top_3_classes[2]])

    plt.imshow(inp)
    plt.title(title_string)
    plt.axis('off')
    plt.show()

def show_saliency(image_tensor, raw_saliency):
    cm = mpl.cm.get_cmap('jet')
    raw_saliency = raw_saliency - np.min(raw_saliency)
    raw_saliency = raw_saliency/np.max(raw_saliency)
    explanation_ = cm(raw_saliency)
    explanation_ = explanation_[:, :, :3]

    img = tensor_to_numpy(image_tensor)

    plt.imshow(img)
    plt.imshow(explanation_, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()


def sample_random_image(base_path='xai_isact_2021/imagenet_random'):
    image_list = os.listdir(base_path)
    random_image = np.random.randint(low=0, high=len(image_list))

    image_path = os.path.join(base_path, image_list[random_image])

    return image_path

if __name__=='__main__':
    random_image_path = sample_random_image('imagenet_random')

    model = torchvision.models.resnet50(True)
    model = model.eval().cuda()
    image_tensor = read_tensor(random_image_path).float().cuda()

    logits = model(image_tensor)

    tensor_imshow(image_tensor, logits)
    plt.show()



