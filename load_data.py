import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib


def get_all_nii_paths(root_dir, m_list):
    all_cases = []

    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        case_paths = []
        for modality in m_list:
            modality_path = os.path.join(subdir_path, modality)
            if os.path.exists(modality_path):
                case_paths.append(modality_path)
            else:
                break
        if len(case_paths) == len(m_list):
            all_cases.append(case_paths)

    return all_cases

class NiiDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        modalities_list = self.file_list[idx]
        m_list_tensor = []
        for i in range(len(modalities_list)):
            labels_nii = nib.load(modalities_list[i])
            m_numpy = self.min_max_normalize(labels_nii.get_fdata())
            m_list_tensor.append(m_numpy)

        m_list_tensor = torch.cat(m_list_tensor, dim=0)
        # shape: (num_modality, H, W, Z）
        return m_list_tensor

    def min_max_normalize(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)

        normalized_arr = (arr - min_val) / (max_val - min_val)

        if max_val == 0:
          return np.zeros_like(arr)

        normalized_arr = np.transpose(normalized_arr, (2, 0, 1))
        tensor_nii = torch.from_numpy(normalized_arr).float().unsqueeze(1)  # shape: (Z, 1, W, H)

        tensor_nii = F.interpolate(tensor_nii, size=(224, 224), mode='bilinear', align_corners=False)

        return tensor_nii.permute(1,0,2,3) # shape: (1, Z, W, H)

def build_loader_for_FMM(data_path,m_list):
    nii_files = get_all_nii_paths(data_path,m_list)


    dataset = NiiDataset(nii_files)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12,
                            pin_memory=True, drop_last=False)


    return dataloader
