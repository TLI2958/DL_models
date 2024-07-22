import os
import random
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import rotate, shift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 15
input_size = (20, 20)
hidden_size = 200
memory_shape = (128, 40)
nb_read = 4
nb_classes = 5
output_size = nb_classes
samples_per_class = 10
gamma = 0.95

# checkpoint
def save_checkpoint(model, optimizer, episode, usage_weight, read_weight, loader_name = None):
    checkpoint_path = f'trained_{episode}.pth'
    torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'usage_weight': usage_weight,
            'read_weight': read_weight,
        }, checkpoint_path)

    
def load_checkpoint(model, optimizer, episode = 1):
    checkpoint_path = f'trained_{episode}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'], )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], )
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler.state.dict'],)
        episode = checkpoint['episode']
        print(f"Checkpoint found. Resuming training from episode {episode}.")
        return model, optimizer, episode, checkpoint['usage_weight'], checkpoint['read_weight']
    else:
        return model, optimizer, 1, torch.randn((batch_size, memory_shape[0]), requires_grad = False), torch.randn((batch_size, nb_read, memory_shape[0]), requires_grad = False)

def load_transform(image_path, angle=0., s=(0, 0), size=(20, 20)):
    # Load the image
    original = Image.open(image_path).convert('L')  # Convert to grayscale

    # Rotate the image
    rotated = original.rotate(angle, fillcolor=255)  # fillcolor=255 for white background

    # Convert to numpy array for shift operation
    rotated_array = np.array(rotated)
    shifted_array = shift(rotated_array, shift=s, cval=255)

    # Convert back to PIL Image
    shifted = Image.fromarray(shifted_array)

    # Resize the image
    resized = shifted.resize(size, Image.LANCZOS)

    # Invert the image
    inverted = ImageOps.invert(resized)

    # Normalize the image
    inverted_array = np.array(inverted, dtype=np.float32) / 255.0
    max_value = np.max(inverted_array)
    if max_value > 0.0:
        inverted_array /= max_value
    inverted_array = torch.tensor(inverted_array)
    return inverted_array


def get_shuffled_images(paths, labels, nb_samples=None):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    random.shuffle(images)
    return images


def time_offset_input(labels_and_images):
    labels, images = zip(*labels_and_images)
    time_offset_labels = (None,) + labels[:-1]
    return zip(images, time_offset_labels)



class OmniglotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.character_folders = [os.path.join(root_dir, family, character)
                                  for family in os.listdir(root_dir)
                                  if os.path.isdir(os.path.join(root_dir, family))
                                  for character in os.listdir(os.path.join(root_dir, family))]
        self.images = [(os.path.join(character, img), character)
                       for character in self.character_folders
                       for img in os.listdir(character) if img.endswith(".png")]
        self.labels = {character: idx for idx, character in enumerate(self.character_folders)}
        self.classes = list(self.labels.keys())


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, character = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[character]

        if self.transform:
            image = self.transform(image)

        return image, label

    def sample_classes(self, num_samples=5, samples_per_class=10, max_rotation= 0,
                       max_shift=0, batch_size = 15, image_size = (20, 20)):
        sampled_character_folders = random.sample(self.character_folders, num_samples)
        random.shuffle(sampled_character_folders)
        example_inputs = torch.zeros((batch_size, num_samples * samples_per_class, np.prod(image_size)))
        example_outputs = torch.zeros((batch_size, num_samples * samples_per_class))


        for i in range(batch_size):
            labels_and_images = get_shuffled_images(sampled_character_folders,
                                                    range(num_samples),
                                                    nb_samples= samples_per_class)
            sequence_length = len(labels_and_images)
            labels, image_files = zip(*labels_and_images)
            labels = torch.tensor(labels)


            angles = np.random.uniform(-max_rotation, max_rotation, size=sequence_length)
            shifts = np.random.randint(-max_shift, max_shift + 1, size=(sequence_length, 2))

            images = torch.stack([load_transform(filename, angle=angle, s=shift, \
                size= image_size).flatten() for (filename, angle, shift) in \
                zip(image_files, angles, shifts)], dim = 0)

            example_inputs[i] = images
            example_outputs[i] = labels

        return example_inputs, example_outputs

class SampledDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# Usage example
transform = transforms.Compose([
    transforms.Resize((20, 20)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])


    
def eval_acc(i, p, t, acc, idx):
    for j in range(p.shape[-1]):
        t_ = int(t[i][j].item())
        p_ = int(p[i][j].item())
        ind = (i, t_)
        acc[i, int(idx[ind].item())] += p_ == t_
        idx[ind] += 1





