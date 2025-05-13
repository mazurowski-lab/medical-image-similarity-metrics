import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# constants
data_dirs = {
    'dbc_by_scanner' : 'data/dbc/png_subset/sorted_by_scanner',
}

cur_dir = os.getcwd()

class MedicalDataset(Dataset):
    def __init__(self, label_csv, data_dir, img_size, transform, make_3_channel=False):
        self.label_csv = label_csv
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        
        self.make_3_channel = make_3_channel
        
        # to be initialized by child class
        self.labels = None
                 
    def normalize(self, img):
        # normalize to range [0, 255]
        # img expected to be array
                 
        # uint16 -> float
        img = img.astype(np.float) * 255. / img.max()
        # float -> unit8
        img = img.astype(np.uint8)
        
        return img
    
    def __getitem__(self, idx):
        
        fpath, target  = self.labels[idx]
        
        # print(fpath)
        
        # load img from file (png or jpg)
        img_arr = io.imread(fpath, as_gray=True)
        
        # print(img_arr.shape)
        
        # normalize
        img_arr = self.normalize(img_arr)
        
        # print(img_arr)
        
        # convert to tensor
        data = torch.from_numpy(img_arr)
        # print(data, type(data))
        data = data.type(torch.FloatTensor) 
       
        # add channel dim
        data = torch.unsqueeze(data, 0)
        
        # resize to standard dimensionality
        data = transforms.Resize((self.img_size, self.img_size))(data)
        # bilinear by default
        
        # make 3-channel (testing only)
        if self.make_3_channel:
            data = torch.cat([data, data, data], dim=0)
        
        # do any data augmentation/training transformations
        if self.transform:
            data = self.transform(data)
        
        return data, target
    
    def __len__(self):
        return len(self.labels)
    
class DBCDataset(MedicalDataset):
    def __init__(self, img_size, labeling='feature: TE', train_transform=None, make_3_channel=False, unique_patients=False):
        super(DBCDataset, self).__init__(None, data_dirs['dbc_by_scanner'], img_size, train_transform, make_3_channel=make_3_channel)
        # constants
        clinical_features_path = 'data/dbc/maps/Clinical_and_Other_Features.csv'


        labels = []
        patient_IDs_used = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building DBC dataset.')
        if labeling == 'default':
            for target, target_label in enumerate(['neg', 'pos']):
                case_dir = os.path.join(self.data_dir, target_label)
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        
                        patient_ID = fname.split('-')[2].replace('.png', '')
                        if unique_patients:
                            # if we only one one datapoint per patient
                            if patient_ID in patient_IDs_used:
                                continue
                            else:
                                patient_IDs_used.append(patient_ID)
                        
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        if 'feature' in labeling:
            # load features
            clinical_features = pd.read_csv(clinical_features_path)


            feature_name = labeling.split(':')[1].strip()
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(".png"):
                        fpath = os.path.join(root, file)
                        patient_ID = file.split('-')[2].replace('.png', '')
                        if unique_patients:
                            # if we only one one datapoint per patient
                            if patient_ID in patient_IDs_used:
                                continue
                            else:
                                patient_IDs_used.append(patient_ID)

                        # get data for named feature
                        patient_ID_long = 'Breast_MRI_{}'.format(patient_ID.zfill(3))
                        print(patient_ID_long)
                        feature_val = clinical_features[clinical_features['Patient ID'] == patient_ID_long][feature_name].values[0]
                        print(feature_val)

        else:
            raise NotImplementedError
            
        self.labels = labels
         
# utils
def get_datasets(dataset_name, labeling='default', train_size=None, 
                 test_size=None, val_size=None, img_size=224, make_3_channel=False,
                 unique_DBC_patients=False):
    # either (1) specify train_frac, which split of subset to create train and test sets, or
    # (2) specify test_size
    
    if labeling != 'default':
        print('using non-default {} labeling.'.format(labeling))

    # first, option of getting subset of full dataset stored
    # then, option of splitting what's left into train and test
    # create dataset
    if dataset_name == 'dbc_by_scanner':
        dataset = DBCDataset(img_size, labeling, make_3_channel=make_3_channel, unique_patients=unique_DBC_patients)
    else:
        raise NotImplementedError
        
    # split into subsets if chosen
    if train_size and val_size and test_size:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(1337))
        
        return train_dataset, val_dataset, test_dataset
    else:
        return dataset