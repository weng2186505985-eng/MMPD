import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

class MMPDDataset(Dataset):
    def __init__(self, tm_path, tc_path, label_path, indices, tm_shape, tc_shape, tm_indices, seq_len, pred_len, stride=1, mode='train'):
        self.tm_path = tm_path
        self.tc_path = tc_path
        self.label_path = label_path
        self.indices = indices
        self.tm_shape = tm_shape
        self.tc_shape = tc_shape
        self.tm_indices = tm_indices
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
        self.tm_mmap = None
        self.tc_mmap = None
        self.label_mmap = None
        
        self.valid_starts = []
        min_idx, max_idx = min(indices), max(indices)
        for i in range(min_idx, max_idx - seq_len - pred_len + 1, stride):
            self.valid_starts.append(i)
                
    def _open_mmap(self):
        if self.tm_mmap is None:
            self.tm_mmap = np.memmap(self.tm_path, dtype='float32', mode='r', shape=self.tm_shape)
            if self.tc_shape[1] > 0:
                self.tc_mmap = np.memmap(self.tc_path, dtype='int8', mode='r', shape=self.tc_shape)
            self.label_mmap = np.memmap(self.label_path, dtype='int8', mode='r', shape=(self.tm_shape[0],))

    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        self._open_mmap()
        start = self.valid_starts[idx]
        
        # TM features
        tm_cond = self.tm_mmap[start : start + self.seq_len, self.tm_indices]
        tm_0 = self.tm_mmap[start + self.seq_len : start + self.seq_len + self.pred_len, self.tm_indices]
        
        # TC features
        if self.tc_mmap is not None:
            tc_cond = self.tc_mmap[start : start + self.seq_len, :].astype(np.float32)
            tc_0 = self.tc_mmap[start + self.seq_len : start + self.seq_len + self.pred_len, :].astype(np.float32)
            x_cond = np.concatenate([tm_cond, tc_cond], axis=-1)
            x_0 = np.concatenate([tm_0, tc_0], axis=-1)
        else:
            x_cond, x_0 = tm_cond, tm_0
        
        # Labels
        if self.label_mmap is not None:
            label = self.label_mmap[start + self.seq_len : start + self.seq_len + self.pred_len]
        else:
            label = np.zeros(self.pred_len, dtype=np.int8)
            
        return torch.from_numpy(x_cond), torch.from_numpy(x_0), torch.from_numpy(label)

def get_dataloaders(metadata_file, config):
    with open(metadata_file, 'rb') as f:
        meta = pickle.load(f)
    
    base = os.path.dirname(metadata_file)
    tm_path = os.path.join(base, 'mission1_tm.mmap')
    tc_path = os.path.join(base, 'mission1_tc.mmap')
    label_path = os.path.join(base, 'mission1_labels.mmap')
    
    time_index = meta['time_index']
    all_indices = np.arange(len(time_index))
    test_mask = time_index >= meta['test_start_date']
    train_val_indices = all_indices[~test_mask]
    
    split_idx = int(len(train_val_indices) * 0.85)
    train_indices = train_val_indices[:split_idx]
    val_indices = train_val_indices[split_idx:]
    
    def create_ds(indices, mode):
        return MMPDDataset(tm_path, tc_path, label_path, indices, 
                          meta['tm_shape'], meta['tc_shape'], meta['tm_indices'],
                          config['seq_len'], config['pred_len'], config['data_stride'], mode)

    num_workers = config.get('num_workers', 0)
    train_loader = DataLoader(create_ds(train_indices, 'train'), 
                              batch_size=config['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=2 if num_workers > 0 else None)
    val_loader = DataLoader(create_ds(val_indices, 'val'), 
                            batch_size=config['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=(num_workers > 0),
                            prefetch_factor=2 if num_workers > 0 else None)
    
    return train_loader, val_loader, {
        'enc_in': len(meta['features']),
        'num_tm': len(meta['tm_features']),
        'num_tc': len(meta['tc_features'])
    }
