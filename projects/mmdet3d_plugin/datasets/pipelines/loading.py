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

    def __init__(self, max_views, to_float32=False, color_type='unchanged',
                 sample_views=False, deterministic=False,
                 selected_view_indices=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.max_views = max_views
        self.sample_views = sample_views
        self.deterministic = deterministic
        self.selected_view_indices = selected_view_indices

    def _select_view_indices(self, results, num_views):
        if num_views <= self.max_views:
            return list(range(num_views))
        if not self.sample_views:
            return list(range(num_views))
        if self.selected_view_indices is not None:
            valid_indices = [
                int(idx) for idx in self.selected_view_indices
                if 0 <= int(idx) < num_views
            ]
            if len(valid_indices) >= self.max_views:
                return valid_indices[:self.max_views]

        sample_id = results.get('img_info', {}).get('token', None) \
                    or results.get('frame_id', None) \
                    or results.get('sample_idx', 0)
        seed = int(hashlib.sha1(str(sample_id).encode()).hexdigest(), 16) % (2**32)
        if self.deterministic:
            rng = random.Random(seed)
            return sorted(rng.sample(range(num_views), self.max_views))

        return sorted(random.sample(range(num_views), self.max_views))

    @staticmethod
    def _slice_camera_field(results, key, indices):
        value = results.get(key)
        if isinstance(value, list) and len(value) >= max(indices) + 1:
            results[key] = [value[i] for i in indices]
        elif isinstance(value, tuple) and len(value) >= max(indices) + 1:
            results[key] = tuple(value[i] for i in indices)

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
        selected_view_indices = self._select_view_indices(results, len(filename))
        if selected_view_indices != list(range(len(filename))):
            filename = [filename[i] for i in selected_view_indices]
            results['img_filename'] = filename
            for key in ('lidar2img', 'cam2img', 'cam_intrinsic',
                        'img2lidars', 'lidar2cam', 'img_timestamp',
                        'camera_names'):
                self._slice_camera_field(results, key, selected_view_indices)
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
        results['num_real_views'] = real_v
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
                pad_intrinsic = np.eye(4, dtype=np.float32)
                results['cam_intrinsic'] += [pad_intrinsic] * pad_n
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
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f"max_views={self.max_views}, "
        repr_str += f"sample_views={self.sample_views}, "
        repr_str += f"deterministic={self.deterministic}, "
        repr_str += f"selected_view_indices={self.selected_view_indices})"
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
    def __init__(self,
                 mask_prob: float = 0.25,
                 blur_kernel_size: int = 11,
                 blur_kernel_size_range=None,
                 blur_sigma=(3.0, 10.0),
                 blur_levels=None,
                 blur_level_probs=None,
                 deterministic: bool = False,
                 mask_ratio: float = 0.5,
                 blur_ratio: float = 0.5):
        assert 0.0 <= mask_prob <= 1.0
        assert mask_ratio >= 0.0 and blur_ratio >= 0.0
        assert mask_ratio + blur_ratio > 0.0
        self.mask_prob = mask_prob
        total_ratio = mask_ratio + blur_ratio
        self.mask_ratio = mask_ratio / total_ratio
        self.blur_ratio = blur_ratio / total_ratio
        # Ensure kernel size is odd for Gaussian blur
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_kernel_size_range = self._parse_kernel_size_range(blur_kernel_size_range)
        self.deterministic = deterministic
        self.blur_levels = self._parse_blur_levels(blur_levels)
        self.blur_level_probs = None
        if blur_level_probs is not None:
            if self.blur_levels is None:
                raise ValueError("blur_level_probs requires blur_levels")
            if len(blur_level_probs) != len(self.blur_levels):
                raise ValueError("blur_level_probs must match blur_levels length")
            prob_sum = float(sum(blur_level_probs))
            if prob_sum <= 0:
                raise ValueError("blur_level_probs must sum to a positive value")
            self.blur_level_probs = [float(p) / prob_sum for p in blur_level_probs]
        # Use fixed sigma
        if isinstance(blur_sigma, (tuple, list)) and len(blur_sigma) == 2:
            self.blur_sigma_range = blur_sigma
        elif isinstance(blur_sigma, (float, int)):
            self.blur_sigma_range = (blur_sigma, blur_sigma)
        else:
            raise ValueError("blur_sigma must be a float or a tuple/list of two floats")

    @staticmethod
    def _to_odd_kernel_size(kernel_size):
        kernel_size = int(kernel_size)
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    @classmethod
    def _parse_kernel_size_range(cls, kernel_size_range):
        if kernel_size_range is None:
            return None
        if not isinstance(kernel_size_range, (tuple, list)) or len(kernel_size_range) != 2:
            raise ValueError("blur_kernel_size_range must be a tuple/list of two ints")
        low = cls._to_odd_kernel_size(kernel_size_range[0])
        high = cls._to_odd_kernel_size(kernel_size_range[1])
        if low > high:
            low, high = high, low
        return (low, high)

    @staticmethod
    def _parse_blur_levels(blur_levels):
        if blur_levels is None:
            return None
        parsed = []
        for level in blur_levels:
            if isinstance(level, str):
                sigma_text, kernel_text = level.split(":", 1)
                sigma = float(sigma_text)
                kernel_size = int(kernel_text)
            elif isinstance(level, (tuple, list)) and len(level) == 2:
                sigma = float(level[0])
                kernel_size = int(level[1])
            else:
                raise ValueError(
                    "Each blur level must be 'sigma:kernel' or (sigma, kernel)"
                )
            kernel_size = RandomMaskMultiView._to_odd_kernel_size(kernel_size)
            parsed.append((sigma, kernel_size))
        if not parsed:
            raise ValueError("blur_levels cannot be empty")
        return parsed

    def _sample_blur_level(self, rng):
        if self.blur_levels is None:
            sigma = rng.uniform(*self.blur_sigma_range)
            if self.blur_kernel_size_range is None:
                return sigma, self.blur_kernel_size
            low, high = self.blur_kernel_size_range
            kernel_size = rng.randrange(low, high + 1, 2)
            return sigma, kernel_size
        if self.blur_level_probs is None:
            return rng.choice(self.blur_levels)
        idx = rng.choices(
            range(len(self.blur_levels)),
            weights=self.blur_level_probs,
            k=1,
        )[0]
        return self.blur_levels[idx]

    def __call__(self, results):
        results.pop('mask_multiview_info', None)
        if self.deterministic:
            sample_id = results.get('img_info', {}).get('token', None) \
                        or results.get('frame_id', None) \
                        or results.get('sample_idx', 0)
            seed = int(hashlib.sha1(str(sample_id).encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            torch.manual_seed(seed)
        else:
            seed = None
            rng = random
        # 1) Random chance
        if rng.random() >= self.mask_prob:
            return results

        num_views = len(results['img'])
        # 2) Identify real (non-padded) view indices
        if 'num_real_views' in results:
            real_idx = list(range(min(int(results['num_real_views']), num_views)))
        elif 'lidar2img' in results:
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
        mask_ind = rng.choice(real_idx)

        # 5) Replace the chosen view
        if rng.random() < self.mask_ratio:
            # total blank
            results['img'][mask_ind] = np.zeros_like(results['img'][mask_ind])
            results['mask_multiview_info'] = dict(
                view_idx=int(mask_ind),
                action='mask',
                alpha=1.0,
                sigma=None,
                seed=seed,
            )
        else:
            sigma, kernel_size = self._sample_blur_level(rng)
            blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
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
            results['mask_multiview_info'] = dict(
                view_idx=int(mask_ind),
                action='blur',
                alpha=1.0,
                sigma=float(sigma),
                kernel_size=int(kernel_size),
                seed=seed,
            )

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mask_prob={self.mask_prob}, '
                f'blur_kernel_size={self.blur_kernel_size}, '
                f'blur_kernel_size_range={self.blur_kernel_size_range}, '
                f'blur_sigma_range={self.blur_sigma_range}, '
                f'blur_levels={self.blur_levels}, '
                f'blur_level_probs={self.blur_level_probs}, '
                f'mask_ratio={self.mask_ratio}, '
                f'blur_ratio={self.blur_ratio})')

@PIPELINES.register_module()
class RandomCalibrationPerturbation(object):
    """Randomly perturb the calibration of one camera in selected samples."""

    def __init__(self,
                 perturb_prob=0.25,
                 rot_noise_deg=(0.0, 2.0),
                 trans_noise_m=(0.0, 0.2),
                 rot_prob=0.5,
                 trans_prob=0.5):
        assert 0.0 <= perturb_prob <= 1.0
        assert 0.0 <= rot_prob <= 1.0
        assert 0.0 <= trans_prob <= 1.0
        self.perturb_prob = perturb_prob
        self.rot_noise_deg = rot_noise_deg
        self.trans_noise_m = trans_noise_m
        self.rot_prob = rot_prob
        self.trans_prob = trans_prob

    @staticmethod
    def _random_unit_vector():
        vec = np.random.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _axis_angle_to_matrix(axis, angle):
        axis = axis.astype(np.float32)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ], dtype=np.float32)

    @staticmethod
    def _require_4x4(matrix, name):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape == (4, 4):
            return matrix
        raise ValueError(f"{name} must be a 4x4 matrix, got {matrix.shape}")

    def _sample_delta(self):
        apply_rot = random.random() < self.rot_prob
        apply_trans = random.random() < self.trans_prob
        if not apply_rot and not apply_trans:
            apply_rot = random.random() < 0.5
            apply_trans = not apply_rot

        delta = np.eye(4, dtype=np.float32)
        if apply_rot:
            angle_deg = random.uniform(*self.rot_noise_deg)
            angle = np.deg2rad(angle_deg)
            if random.random() < 0.5:
                angle = -angle
            delta[:3, :3] = self._axis_angle_to_matrix(
                self._random_unit_vector(), angle)

        if apply_trans:
            trans_mag = random.uniform(*self.trans_noise_m)
            delta[:3, 3] = self._random_unit_vector() * trans_mag

        return delta

    def __call__(self, results):
        if random.random() >= self.perturb_prob:
            return results
        if 'lidar2cam' not in results:
            return results

        num_real_views = len(results.get('filename', results['lidar2cam']))
        num_real_views = min(num_real_views, len(results['lidar2cam']))
        if num_real_views <= 0:
            return results

        view_idx = random.randrange(num_real_views)
        delta = self._sample_delta()
        lidar2cam = delta @ self._require_4x4(results['lidar2cam'][view_idx], 'lidar2cam')
        results['lidar2cam'][view_idx] = lidar2cam
        if 'cam_intrinsic' in results:
            results['lidar2img'][view_idx] = (
                self._require_4x4(results['cam_intrinsic'][view_idx], 'cam_intrinsic')
                @ lidar2cam)

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(perturb_prob={self.perturb_prob}, '
                f'rot_noise_deg={self.rot_noise_deg}, '
                f'trans_noise_m={self.trans_noise_m}, '
                f'rot_prob={self.rot_prob}, trans_prob={self.trans_prob})')

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
