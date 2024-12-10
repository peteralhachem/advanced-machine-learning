from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)

        return img.convert('RGB')


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None, split='train', transform='None') : 
    # dir = 'Caltech101'
    images = []
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return x.lower().endswith(extensions)

    inputFile = os.path.join(dir, split + '.txt') # 'Caltech101/{split}.txt'
    with open(inputFile, 'r') as f:      
      input_images = f.read().splitlines()

    root = os.path.join(dir, '101_ObjectCategories/') # 'Caltech101/101_ObjectCategories/'

    for fname in input_images:
      fpath = os.path.split(fname)
      # print(fpath) # 'accordion' 'image_0002.jpg'
      target = fpath[0] # 'accordion'
      path = os.path.join(root, fname) # 'Caltech101/101_ObjectCategories/accordion/image_0002.jpg'
      if is_valid_file(path) and target != 'BACKGROUND_Google':
        item = (path, class_to_idx[target])
        images.append(item)

    return images # paths

class Caltech(VisionDataset):
    ''' Caltech 101 Dataset '''

    def __init__(self, root, split='train', transform=None, target_transform=None):
        '''
          Args:
            root (string): Directory with all the images.
            split (string): Defines the split you are going to use (split files are called 'train.txt' and 'test.txt').
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional trasform to be applied on the target labels (not used).
        '''
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, IMG_EXTENSIONS, split=self.split, transform=transform)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = pil_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):        
        root = os.path.join(dir, '101_ObjectCategories/')
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.remove('BACKGROUND_Google')
        classes.sort()                
        #print(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        #print(class_to_idx)
        return classes, class_to_idx

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            image = self.transform(sample)

        return image, label

    def __len__(self):   
        '''
        The __len__ method returns the length of the dataset
        '''     
        return len(self.samples)
