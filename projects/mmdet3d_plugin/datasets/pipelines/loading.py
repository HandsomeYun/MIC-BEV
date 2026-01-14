import mmcv
import numpy as np
from torchvision.transforms import GaussianBlur
import random
import torch
import hashlib
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, max_views , to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.max_views = max_views

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        # ─── pad dummy cameras up to 4 ───────────────────────────────────────
        real_v   = len(results['img'])
        pad_n = self.max_views - real_v
        if pad_n > 0:
            # create a “black” image matching one real view
            pad_img = np.zeros_like(results['img'][0]) # create a “black” image matching one real view
            results['img'] += [pad_img] * pad_n
            
            # Pad lidar2img
            if 'lidar2img' in results:
                pad_l2i  = np.eye(4, dtype=np.float32)
                max_z = 1.0 #cfg.point_cloud_range[5]
                pad_l2i[2,3] = -(max_z + 0.1)  # e.g. –1.1
                results['lidar2img'] += [pad_l2i] * pad_n
            # Pad cam2img
            if 'cam2img' in results:
                pad_K = np.eye(4, dtype=np.float32)
                pad_K[2, 3] = -1.0
                results['cam2img'] += [pad_K] * pad_n
            # Pad camera intrinsics
            if 'cam_intrinsic' in results:
                pad_3x3 = np.eye(3, dtype=np.float32)
                results['cam_intrinsic'] += [pad_3x3] * pad_n
            if 'img2lidars' in results:
                pad_3x3 = np.eye(4, dtype=np.float32)
                results['img2lidars'] += [pad_3x3] * pad_n
        # ─────────────────────────────────────────────────────────────────────

        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        # print(f"results['img_shape']: {results['img_shape']}")
        # print(f"real_v: {real_v}")
        # print(f"pad_n: {pad_n}")
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class FilterEmptyGT(object):
    """Drop samples with zero GT after range/name filtering."""
    def __call__(self, results):
        if len(results.get('gt_labels_3d', [])) == 0:
            return None
        return results
    

@PIPELINES.register_module()
class RandomMaskMultiView(object):
    """Randomly mask out exactly 1 camera view per sample.

    With probability `mask_prob`, picks exactly one *real* view
    and replaces it either with a blank image or a Gaussian-blurred version.
    Ensures at least one real view remains unmasked.
    """
    def __init__(self, mask_prob: float = 0.25, blur_kernel_size: int = 11, blur_sigma=(3.0, 10.0), deterministic: bool = False):
        assert 0.0 <= mask_prob <= 1.0
        self.mask_prob = mask_prob
        # Ensure kernel size is odd for Gaussian blur
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.deterministic = deterministic
        # Use fixed sigma
        if isinstance(blur_sigma, (tuple, list)) and len(blur_sigma) == 2:
            self.blur_sigma_range = blur_sigma
        elif isinstance(blur_sigma, (float, int)):
            self.blur_sigma_range = (blur_sigma, blur_sigma)
        else:
            raise ValueError("blur_sigma must be a float or a tuple/list of two floats")

    def __call__(self, results):
        if self.deterministic:
            sample_id = results.get('img_info', {}).get('token', None) \
                        or results.get('frame_id', 0)
            seed = int(hashlib.sha1(str(sample_id).encode()).hexdigest(), 16) % (2**32)
            random.seed(seed)
            torch.manual_seed(seed)
        # 1) Random chance
        if random.random() >= self.mask_prob:
            return results

        num_views = len(results['img'])
        # 2) Identify real (non-padded) view indices
        if 'lidar2img' in results:
            real_idx = [
                i for i, E in enumerate(results['lidar2img'])
                if not np.allclose(E, np.eye(E.shape[0]), atol=1e-6)
            ]
        else:
            real_idx = list(range(num_views))

        # 3) If there are no real views, skip masking
        if not real_idx:
            return results
        
        # If only one real view, skip masking
        if len(real_idx) <= 1:
            return results

        # 4) Always mask exactly one view
        mask_ind = random.choice(real_idx)

        # 5) Replace the chosen view
        if random.random() < 0.5:
            # total blank
            results['img'][mask_ind] = np.zeros_like(results['img'][mask_ind])
        else:
            # Randomly sample sigma within range
            sigma = random.uniform(*self.blur_sigma_range)
            blur = GaussianBlur(kernel_size=self.blur_kernel_size, sigma=sigma)
            # apply Gaussian blur to the original image
            img = results['img'][mask_ind]
            img_t = torch.from_numpy(img).float()
            # If HxWxC, permute to CxHxW
            if img_t.ndim == 3 and img_t.shape[-1] not in img_t.shape[:2]:
                img_t = img_t.permute(2, 0, 1)
            blurred = blur(img_t.unsqueeze(0)).squeeze(0)
            # Restore original shape if permuted
            if img.ndim == 3 and blurred.shape[0] == img.shape[-1]:
                blurred = blurred.permute(1, 2, 0)
            results['img'][mask_ind] = blurred.numpy()

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mask_prob={self.mask_prob}, '
                f'blur_kernel_size={self.blur_kernel_size}, '
                f'blur_sigma_range={self.blur_sigma_range})')

@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list  

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
