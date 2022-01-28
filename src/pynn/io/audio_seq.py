# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import struct
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from . import smart_open
 
class SpectroDataset(Dataset):
    def __init__(self, scp_paths, label_paths=None, paired_label=False,
                 verbose=True, sek=True, sort_src=False, pack_src=False,
                 downsample=1, preload=False, threads=4, fp16=False, 
                 spec_drop=False, spec_bar=2, spec_ratio=0.4,
                 time_stretch=False, time_win=10000, mean_sub=False, var_norm=False):
        self.scp_paths = scp_paths.split(',')     # path to the .scp file
        self.label_paths = label_paths.split(',') # path to the label file
        self.paired_label = paired_label
        
        ## same cv set for all trainings
        if not time_stretch:
            random.seed(42)

        self.downsample = downsample
        self.sort_src = sort_src
        self.pack_src = pack_src
        self.sek = sek

        self.mean_sub = mean_sub
        self.var_norm = var_norm
        self.spec_drop = spec_drop
        self.spec_bar = spec_bar
        self.spec_ratio = spec_ratio
        self.time_stretch = time_stretch
        self.time_win = time_win
        self.fp16 = fp16

        self.threads = threads
        self.preload = preload
        self.utt_lbl = None
        self.ark_cache = None
        self.ark_files = {}

        self.scp_file = None
        self.lbl_dic = None

        self.verbose = verbose
        self.rank = 0
        self.parts = 1
        self.epoch = -1

        self.total_utts = 0
        self.utt_stats = {}
        self.scp_sets = {}
        self.scp_set_length ={}
        self.total_labels = 0
        for scp_set in self.scp_paths:
            self.scp_sets[scp_set.rsplit('/',1)[-1]] = {}
        self.__load_utts()
                   
        self.cs_scp_number = {}
        for scp_name, _ in self.scp_sets.items():
            self.cs_scp_number[scp_name] = 0
      


    def __utt_id_exists(self, utt_id):
        for scp_set in self.scp_sets: 
           if utt_id in scp_set:
               return True
        return False

    def __load_utts(self):
        for scp_path in self.scp_paths:
            path = os.path.dirname(scp_path)
            scp_dir = path + '/' if path != '' else ''
            scp_set = scp_path.rsplit('/',1)[-1]
            utts = list()
            for line in smart_open(scp_path, 'r'):
                if line.startswith('#'): continue
                tokens = line.replace('\n','').split(' ')
                utt_id, path_pos = tokens[0:2]
                utt_len = -1 if len(tokens)<=2 else int(tokens[2])
                path, pos = path_pos.split(':')
                if utt_len < 0: utt_len = self._read_length(path, pos, cache=True)
                path = path if path.startswith('/') else scp_dir + path
                #utts[utt_id] = (utt_id, path, pos, utt_len)
                utts.append((utt_id, path, pos, utt_len))
                self.utt_stats[utt_id] = scp_set
                self.total_utts += 1
            #utts =  OrderedDict(sorted(utts.items(), key=lambda item: item[1][3]))
            #utts = {utt_id: (utt_id, path, pos, utt_len) for utt_id, (utt_id, path, pos, utt_len) in sorted(utts.items(), key=lambda item: item[1][3])}
            utts = sorted(utts, key=lambda item: item[3])
            self.scp_sets[scp_set] = utts
            self.scp_set_length[scp_set] = len(utts)-1

    def __create_cs(self, cs_ratio, cs_noswitch):       
        no_cs = {}
        for scp_set, _ in self.scp_sets.items():
            no_cs[scp_set] = list()
 
        if cs_noswitch:
            no_cs_pairs = cs_noswitch.split(',') #format "0:1,0:2" means no cs from scp_set index 0 to (1 and 2)
            for cs_pairs in no_cs_pairs:
                s_scp, to_scp = cs_pairs.split(':')
                if s_scp in no_cs:
                    no_cs[s_scp].append(to_scp)
                else:
                    no_cs[s_scp] = [to_scp]


        five_sec_frames = (((5000-25)//10)+1)
        ten_sec_frames = (((10000-25)//10)+1)
        fiveteen_sec_frames = (((15000-25)//10)+1)
        twenty_sec_frames = (((20000-25)//10)+1)
        twentyfive_sec_frames = (((25000-25)//10)+1)
        offset = (((2000-25)//10)+1)  #anzahl frames in 2.05 sec or 2005ms (frames:25ms stride:10ms)
        length_dict = {0: five_sec_frames, 1: ten_sec_frames, 2: fiveteen_sec_frames, 3: twenty_sec_frames, 4: twentyfive_sec_frames}
        cs_part = self.total_utts * cs_ratio
        mono_part = self.total_utts * (1-cs_ratio)
        mono_count = 0
        cs_parts = [cs_part*0.25 if key < 3 else cs_part*0.125 for key, time in length_dict.items()]
        
        cs_utts = {}
        self.used_ids = {}

        for key, time in length_dict.items():
            time_cs_elements = 0
            scp_set_cut_length = self.scp_set_length.copy()
            while time_cs_elements < cs_parts[key] and cs_ratio > 0.0:                
                size = 0
                scp_set = random.choice(list(self.scp_sets)) #select random scp_set
                #utt_id = random.choice(list(self.scp_sets[scp_set])) #select random utterance from scp_set
                utt_idx = random.randint(0, scp_set_cut_length[scp_set])
                #utt_id = list(self.scp_sets[scp_set].keys())[utt_idx]
                #utt_id, path, pos, utt_len = self.scp_sets[scp_set][utt_id]
                utt_id, path, pos, utt_len = self.scp_sets[scp_set][utt_idx]
                if utt_len < 0: utt_len = self._read_length(path, pos, cache=True)
                size = utt_len

                if size > time-offset:                   
		    ## scp_set_length is sorted according to length if idx is too long >idx will be too long as well
                    scp_set_cut_length[scp_set] = utt_idx -1 
                    continue
               
                ids, utt_infos = list(), list()
                ids.append(utt_id)
                utt_infos.append((utt_id, path, pos, utt_len))
                self.cs_scp_number[scp_set] += 1
                self.used_ids[utt_id] = 1
                while size < time-offset:
                    n_scp_set = random.choice(list(self.scp_sets))
                    while scp_set==n_scp_set or n_scp_set in no_cs[scp_set]:
                        n_scp_set = random.choice(list(self.scp_sets))
                    
                    #utt_id = random.choice(list(self.scp_sets[n_scp_set]))
                    #utt_id, path, pos, utt_len = self.scp_sets[n_scp_set][utt_id]
                    utt_idx = random.randint(0, scp_set_cut_length[scp_set])
                    utt_id, path, pos, utt_len = self.scp_sets[scp_set][utt_idx]
                    if utt_len < 0: utt_len = self._read_length(path, pos, cache=True)

                    ids.append(utt_id)
                    utt_infos.append((utt_id, path, pos, utt_len))
                    self.cs_scp_number[scp_set] += 1
                    self.used_ids[utt_id] = 1
                    scp_set = n_scp_set
                    size += utt_len

                time_cs_elements += 1
                cs_utts[tuple(ids)] = utt_infos
       
        mono_count = 0
        for scp_set in self.scp_sets:
            for utt_infos in self.scp_sets[scp_set]:
                if utt_infos[0] in self.used_ids: continue 
                cs_utts[(utt_infos[0],)] = [utt_infos]
                self.cs_scp_number[scp_set] += 1
                self.used_ids[utt_infos[0]] = 1             
                mono_count += 1       
        
        while mono_count < mono_part and cs_ratio > 0.0:
            scp_set = random.choice(list(self.scp_sets))               
            utt_idx = random.randint(0, len(self.scp_sets[scp_set])-1)
            utt_id, path, pos, utt_len = self.scp_sets[scp_set][utt_idx]
            if utt_len < 0: utt_len = self._read_length(path, pos, cache=True)
 
            cs_utts[(utt_id,)] = [(utt_id, path, pos, utt_len)]
            self.cs_scp_number[scp_set] += 1
            self.used_ids[utt_id] = 1
            mono_count += 1      

        return cs_utts

    def __make_stats(self, cs_utts):
        print("Debug: Totel number of utterances befor and after codeswitch: {} -> {} ".format(
                                                                                        sum([len(scp_set) for _, scp_set in self.scp_sets.items()]) , len(cs_utts)))
        for scp_name, scp_set in self.scp_sets.items():
            print("Debug: Number of utts in {}(dataset): {} in percentage: {}".format(scp_name, len(scp_set), (self.cs_scp_number[scp_name]/len(cs_utts)*100)))




    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts
   
    def set_epoch(self, epoch):
        self.epoch = epoch

    def print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def get_total_labels(self):
        return self.total_labels

    def initialize(self, b_input=20000, b_sample=64, cs_ratio=0.0, cs_noswitch=''):
        #if self.utt_lbl is not None:
        #    return
        self.total_labels = 0
        cs_utts = self.__create_cs(cs_ratio, cs_noswitch)
        self.__make_stats(cs_utts)

        labels = {}
        used_label_paths= list()
        for label_path in self.label_paths:
         if label_path in used_label_paths: continue
         used_label_paths.append(label_path)
         for line in smart_open(label_path, 'r'):
             tokens = line.split()
             utt_id = tokens[0]
          
             if utt_id == '' or utt_id not in self.used_ids: continue

             if self.paired_label:
                 sp = tokens.index('|', 1)
                 lb1 = [int(token) for token in tokens[1:sp]]
                 lb1 = [1] + [el+2 for el in lb1] + [2] if self.sek else lb1
                 lb2 = [int(token) for token in tokens[sp+1:]]
                 lb2 = [1] + [el+2 for el in lb2] + [2] if self.sek else lb2
                 lbl = (lb1, lb2)
             else:
                 lbl = [int(token) for token in tokens[1:]]
                 lbl = [el+2 for el in lbl] if self.sek else lbl
                 self.total_labels += len(lbl) 

             labels[utt_id] = lbl

        utt_lbl = []
        not_loaded_labels = 0
        for utt_ids, utt_infos in cs_utts.items():
            if any(utt_id not in labels for utt_id in utt_ids):
                not_loaded_labels += 1
                continue

            lbls = [1] + [lbls for utt_id in utt_ids for lbls in labels[utt_id]] + [2]
            utt_lbl.append([[*utt_infos], lbls])

        print("Labels not found #{}".format(not_loaded_labels))
        self.utt_lbl = utt_lbl
        self.print('%d label sequences loaded.' % len(self.utt_lbl))
        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')
        
        if self.preload:
            self.print('Loading ark files.. ', end='')
            self.preload_feats()
            self.print('Done.')

    def create_loader(self):
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]
        
        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        complex_utts = self.utt_lbl
              
        self._close_ark_files()
     
        lst = list()
        for j, (utt_infos, _) in enumerate(complex_utts):
            complex_length = sum([utt[3] for utt in utt_infos])
            lst.append((j, complex_length))
        lst = sorted(lst, key=lambda e : e[1])

        s, j, step = 0, 4, 4
        batches = []
        while j <= len(lst):
            bs = j - s
            if lst[j-1][1]*bs < b_input and bs < b_sample:
                j += step
                continue
            if bs > 8: j = s + (bs // 8) * 8
            batches.append([idx for idx, _ in lst[s:j]])
            s = j
            j += step
        if s < len(lst): batches.append([idx for idx, _ in lst[s:]])
        return batches

    def preload_feats(self):
        mats = {}
        for utt in self.utt_lbl:
            utt_id, path, pos = utt[0:3]
            mat = self._read_mat(path, pos, cache=True)
            mats[utt_id] = mat
        self.ark_cache = mats
        self._close_ark_files()
 
    def _read_string(self, ark_file):
        s = ''
        while True:
            c = ark_file.read(1).decode('utf-8')
            if c == ' ' or c == '': return s
            s += c

    def _read_integer(self, ark_file):
        n = ord(ark_file.read(1))
        return struct.unpack('>i', ark_file.read(n)[::-1])[0]

    def _read_length(self, path, pos, cache=False):
        if cache and path in self.ark_files:
            ark_file = self.ark_files[path]
        else:
            ark_file = smart_open(path, 'rb')
            if cache: self.ark_files[path] = ark_file

        ark_file.seek(int(pos))
        header = ark_file.read(2).decode('utf-8')
        if header != "\0B":
            raise Exception("Input .ark file is not binary")
        format = self._read_string(ark_file)
        utt_len = self._read_integer(ark_file) if format in ('FM', 'HM') else -1

        if not cache: ark_file.close()
        return utt_len

    def _close_ark_files(self):
        for fi in self.ark_files.values(): fi.close()
        self.ark_files = {}

    def _read_mat(self, path, pos, cache=False):
        if cache and path in self.ark_files:
            ark_file = self.ark_files[path]
        else:
            ark_file = smart_open(path, 'rb')
            if cache: self.ark_files[path] = ark_file

        ark_file = smart_open(path, 'rb')

        ark_file.seek(int(pos))
        header = ark_file.read(2).decode('utf-8')
        if header != "\0B": return None

        format = self._read_string(ark_file)
        if format == "FM" or format == "HM":
            rows = self._read_integer(ark_file)
            cols = self._read_integer(ark_file)
            fm, dt, sz = ("<%df", np.float32, 4) if format == "FM" else ("<%de", np.float16, 2)
            utt_mat = struct.unpack(fm % (rows * cols), ark_file.read(rows*cols*sz))
            utt_mat = np.array(utt_mat, dtype=dt)
            if self.fp16 and dt == np.float32:
                utt_mat = utt_mat.astype(np.float16)
            utt_mat = np.reshape(utt_mat, (rows, cols))
        else:
            utt_mat = None

        if not cache: ark_file.close()
        return utt_mat

    def __len__(self):
        return len(self.utt_lbl)

    def __getitem__(self, index):
        utt_infos, lbl = self.utt_lbl[index]
        utt_mat = np.array([])
        for utt_info in utt_infos:
            if utt_mat.size == 0:
                utt_mat = self.read_mat_cache(*utt_info[0:3])
            else:
                new_utt_mat = self.read_mat_cache(*utt_info[0:3])
                utt_mat = np.concatenate((utt_mat, new_utt_mat), axis=0)

        return (utt_mat, lbl)

    def read_mat_cache(self, utt_id, path, pos):
        cache = self.ark_cache
        if cache is not None and utt_id in cache:
            return cache[utt_id]
        return self._read_mat(path, pos)

    def read_utt(self):
        if self.scp_file is None:
            self.scp_file = smart_open(self.scp_path, 'r')
            path = os.path.dirname(self.scp_path)
            self.scp_dir = path + '/' if path != '' else ''

        line = self.scp_file.readline()
        if not line: return None, None
        utt_id, path_pos = line.replace('\n','').split(' ')[0:2]
        path, pos = path_pos.split(':')
        path = path if path.startswith('/') else self.scp_dir + path
        utt_mat = self._read_mat(path, pos)
        return utt_id, utt_mat

    def read_batch_utt(self, batch_size=10):
        mats, ids = [], []
        for i in range(batch_size):
            utt_id, utt_mat = self.read_utt()
            if utt_id is None or utt_id == '': break
            mats.append(utt_mat)
            ids.append(utt_id)
        if len(mats) == 0: return (None, None, None)

        lst = sorted(zip(mats, ids), key=lambda e : -e[0].shape[0])
        src, ids = zip(*lst)
        src = self.augment_src(src)
        src = self.collate_src(src)
        return (*src, ids)

    #TODO self.label_path determine when used and how to adapt for code-switching
    def read_label(self, utt_id):
        if self.lbl_dic is None:
            lbl_dic = {}
            for line in smart_open(self.label_path, 'r'):
                tokens = line.split()
                utt_id = tokens[0]
                if utt_id == '': continue
                lbl_dic[utt_id] = [int(token) for token in tokens[1:]]
            self.lbl_dic = lbl_dic

        if utt_id not in self.label_dic:
            return None

        utt_lbl = self.label_dic[utt_id]
        if self.sek:
            utt_lbl = [1] + [el+2 for el in utt_lbl] + [2]
        return utt_lbl

    def timefreq_drop_inst(self, inst, num=2, time_drop=0.4, freq_drop=0.4):
        time_num, freq_num = inst.shape
        freq_num = freq_num

        n = random.randint(0, int(freq_drop*freq_num))
        f0 = random.randint(0, freq_num-n)
        inst[:, f0:f0+n] = 0

        max_time = int(time_drop * time_num)
        num = random.randint(1, num)
        time_len = max_time // num
        for i in range(num):
            n = min(max_time, random.randint(0, time_len))
            t0 = random.randint(0, time_num-n)
            inst[t0:t0+n, :] = 0    

        return inst

    def time_stretch_inst(self, inst, low=0.85, high=1.2, win=10000):
        time_len = inst.shape[0]
        ids = None
        for i in range((time_len // win) + 1):
            s = random.uniform(low, high)
            e = min(time_len, win*(i+1))          
            r = np.arange(win*i, e-1, s, dtype=np.float32)
            r = np.round(r).astype(np.int32)
            ids = r if ids is None else np.concatenate((ids, r))
        return inst[ids]

    def mean_sub_inst(self, inst):
        return inst - inst.mean(axis=0, keepdims=True)

    def std_norm_inst(self, inst):
        return (inst - inst.mean(axis=0, keepdims=True)) / inst.std(axis=0, keepdims=True)

    def down_sample_inst(self, feats, cf=4):
        feats = feats[:(feats.shape[0]//cf)*cf,:]
        return feats.reshape(feats.shape[0]//cf, feats.shape[1]*cf)
     
    def augment_src(self, src):
        insts = []
        bar, ratio = self.spec_bar, self.spec_ratio
        for inst in src:
            inst = self.mean_sub_inst(inst) if self.mean_sub and not self.var_norm else inst
            inst = self.std_norm_inst(inst) if self.var_norm else inst
            inst = self.time_stretch_inst(inst, win=self.time_win) if self.time_stretch else inst
            inst = self.timefreq_drop_inst(inst, num=bar, time_drop=ratio) if self.spec_drop else inst            
            inst = self.down_sample_inst(inst, self.downsample) if self.downsample > 1 else inst
            insts.append(inst)
        return insts

    def collate_src(self, insts):
        max_len = max(inst.shape[0] for inst in insts)
        inputs = np.zeros((len(insts), max_len, insts[0].shape[1]))
        masks = torch.zeros((len(insts), max_len), dtype=torch.uint8)
        
        for idx, inst in enumerate(insts):
            inputs[idx, :inst.shape[0], :] = inst
            masks[idx, :inst.shape[0]] = 1
        inputs = torch.HalfTensor(inputs) if self.fp16 else torch.FloatTensor(inputs)

        return inputs, masks

    def collate_src_pack(self, insts):
        max_len = max(inst.shape[0] for inst in insts)
        masks = torch.zeros((len(insts), max_len), dtype=torch.uint8)
        inputs = []
        
        for idx, inst in enumerate(insts):
            inputs.append(torch.HalfTensor(inst) if self.fp16 else torch.FloatTensor(inst))
            masks[idx, 0:inst.shape[0]] = 1
        inputs = pack_sequence(inputs)
                    
        return inputs, masks

    def collate_tgt(self, tgt):
        if self.paired_label:
            lb1, lb2 = zip(*tgt)
            max_len = max(len(inst) for inst in lb1)
            lb1 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb1])
            max_len = max(len(inst) for inst in lb2)
            lb2 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb2])
            labels = (torch.LongTensor(lb1), torch.LongTensor(lb2))
        else:
            max_len = max(len(inst) for inst in tgt)
            labels = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
            labels = (torch.LongTensor(labels),)
 
        return (*labels,) 

    def collate_fn(self, batch):
        src, tgt = zip(*batch)
        src = self.augment_src(src)

        if self.sort_src or self.pack_src:
            lst = sorted(zip(src, tgt), key=lambda e : -e[0].shape[0])
            src, tgt = zip(*lst)

        src = self.collate_src(src) if not self.pack_src else self.collate_src_pack(src)
        tgt = self.collate_tgt(tgt)
            
        return (*src, *tgt)
