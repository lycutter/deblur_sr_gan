from os import listdir
from os.path import join
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        # Resize((128 // upscale_factor, 128 // upscale_factor), interpolation=Image.BICUBIC),
        ToTensor()
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def train_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        # Resize((128,128), interpolation=Image.BICUBIC),
        ToTensor()

    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

        self.hr_transform = train_hr_transform(crop_size)
        self.image_names = []
        for i in range(len(self.image_filenames)):
            image_name = self.image_filenames[i].split('\\')[-1]
            self.image_names.append(image_name)


    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        image_name = self.image_names[index]
        return [hr_image, image_name]

    def __len__(self):
        return len(self.image_filenames)

train_set = TrainDatasetFromFolder('D:/pythonWorkplace/Dataset/CelebA_train/no_crop', crop_size=128)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=1, shuffle=False)

for i, [data, image_name] in enumerate(train_loader):
    face= transforms.ToPILImage()(data.cpu()[0])
    image_name = image_name[0]
    face.save('D:/pythonWorkplace/Dataset/CelebA_train/crop_X128/' + image_name)