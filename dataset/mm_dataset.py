import os
import json
import numpy as np
import math
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from config.config_mm import Config, parse_args, class_dict
from decord import VideoReader
from decord import cpu
import concurrent.futures
DEBUG_MODE = False


class MMDataset(Dataset):
    def __init__(self, data_path, mode, modal, fps, num_frames, len_feature, sampling, seed=-1, supervision='weak'):
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.deterministic = True
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.benchmark = False

        self.mode = mode
        self.fps = fps
        self.num_frames = num_frames
        self.len_feature = len_feature

        self.local_data_path = os.path.join(data_path, self.mode)

        # For video processing
        self.transform = self.get_transform(mode, 224)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = dict((v, k) for k, v in class_dict.items())
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_frame, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_frame, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_frame
    

    def process_video(self, video_path, desired_fps):
        """Load video, adjust to desired FPS, resize frames, and return as numpy array."""
        vr = VideoReader(video_path, ctx=cpu(0))
        original_fps = vr.get_avg_fps()
        frame_interval = round(original_fps / desired_fps)
        usable_frame_count = math.ceil(len(vr) / frame_interval)

        frame_indices = range(0, len(vr), frame_interval)
        frames = vr.get_batch(frame_indices).asnumpy()
        transformed_frames = [self.transform(frame) for frame in frames]

        if DEBUG_MODE:
            print(f"Total frames: {len(vr)}")
            print(f"Original FPS: {original_fps}")
            print(f"Frame interval: {frame_interval}")
            print(f"Usable frame count: {usable_frame_count}")

        return torch.stack(transformed_frames), usable_frame_count


    def get_data(self, index):
        vid_name = self.vid_list[index]
        vid_num_frame = 0
        # Get all filename have vid_name in self.feature_path
        vid_name_all = sorted([f for f in os.listdir(self.local_data_path) if vid_name == '_'.join(f.split('.')[0].split('_')[-2:])])

        # For video processing
        video_paths = [os.path.join(self.local_data_path, vid_n) for vid_n in vid_name_all]
        # Using ThreadPoolExecutor to process videos
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
            args = ((path, self.fps) for path in video_paths)
            future_to_video = {executor.submit(self.process_video, *arg): arg[0] for arg in args}
            results = []
            frame_counts = []
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_data, frame_count = future.result()
                    results.append(video_data)
                    frame_counts.append(frame_count)
                    if DEBUG_MODE:
                        print(f"Processed and resized {video_path}")
                except Exception as exc:
                    print(f'{video_path} generated an exception: {exc}')

        vid_num_frame = min(frame_counts)
        if self.sampling == 'random':
            sample_idx = self.random_perturb(vid_num_frame)
        elif self.sampling == 'uniform':
            sample_idx = self.uniform_sampling(vid_num_frame)

        results = [result[sample_idx] for result in results]

        combined_video_data = torch.stack(results)
        return combined_video_data, vid_num_frame, sample_idx
    

    def get_label(self, index, vid_num_frame, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)
        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_frame, self.num_classes])
            t_factor = self.fps 

            for class_idx in range(self.num_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]
            return label, torch.from_numpy(temp_anno)


    def random_perturb(self, length):
        if self.num_frames == length:
            return np.arange(self.num_frames).astype(int)
        samples = np.arange(self.num_frames) * length / self.num_frames
        for i in range(self.num_frames):
            if i < self.num_frames - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_frames == length:
            return np.arange(self.num_frames).astype(int)
        samples = np.arange(self.num_frames) * length / self.num_frames
        samples = np.floor(samples)
        return samples.astype(int)
    
    def get_transform(self, mode, input_size):
        if mode == "train":
            return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        else:
            return transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
