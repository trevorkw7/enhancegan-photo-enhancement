import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PhotoDataset(Dataset):
    def __init__(self, root_dir, split='train', size=224):
        """
        Expects folder structure:
          root_dir/
            train/raw, train/edited
            val/raw,   val/edited
            test/raw,  test/edited
        """
        self.raw_dir    = os.path.join(root_dir, split, 'raw')
        self.edit_dir   = os.path.join(root_dir, split, 'edited')
        self.fns        = sorted(os.listdir(self.raw_dir))
        self.transform  = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        raw_path  = os.path.join(self.raw_dir, fn)
        edit_path = os.path.join(self.edit_dir, fn)
        raw  = Image.open(raw_path).convert('RGB')
        edit = Image.open(edit_path).convert('RGB')
        return self.transform(raw), self.transform(edit)
