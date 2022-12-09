import numpy as np
import torch
import os
from torchvision import transforms
import torch.utils.data
import PIL
import torchvision.transforms.functional as FT
#from .functional import _interpolation_modes_from_int, InterpolationMode
from PIL import Image


if 'DATA_ROOT' in os.environ:
    DATA_ROOT = os.environ['DATA_ROOT']
else:
    DATA_ROOT = './data'

IMAGENET_PATH = './data/imagenet/raw-data'


def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


mean = {
    'mnist': (0.1307,),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

std = {
    'mnist': (0.3081,),
    'cifar10': (0.2470, 0.2435, 0.2616)
}

# class ElasticTransform(torch.nn.Module):
#     """Transform a tensor image with elastic transformations.
#     Given alpha and sigma, it will generate displacement
#     vectors for all pixels based on random offsets. Alpha controls the strength
#     and sigma controls the smoothness of the displacements.
#     The displacements are added to an identity grid and the resulting grid is
#     used to grid_sample from the image.
#     Applications:
#         Randomly transforms the morphology of objects in images and produces a
#         see-through-water-like effect.
#     Args:
#         alpha (float or sequence of floats): Magnitude of displacements. Default is 50.0.
#         sigma (float or sequence of floats): Smoothness of displacements. Default is 5.0.
#         interpolation (InterpolationMode): Desired interpolation enum defined by
#             :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
#             If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
#             For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
#         fill (sequence or number): Pixel fill value for the area outside the transformed
#             image. Default is ``0``. If given a number, the value is used for all bands respectively.
#     """

#     def __init__(self, alpha=50.0, sigma=5.0, interpolation=InterpolationMode.BILINEAR, fill=0):
#         super().__init__()
#         _log_api_usage_once(self)
#         if not isinstance(alpha, (float, Sequence)):
#             raise TypeError(f"alpha should be float or a sequence of floats. Got {type(alpha)}")
#         if isinstance(alpha, Sequence) and len(alpha) != 2:
#             raise ValueError(f"If alpha is a sequence its length should be 2. Got {len(alpha)}")
#         if isinstance(alpha, Sequence):
#             for element in alpha:
#                 if not isinstance(element, float):
#                     raise TypeError(f"alpha should be a sequence of floats. Got {type(element)}")

#         if isinstance(alpha, float):
#             alpha = [float(alpha), float(alpha)]
#         if isinstance(alpha, (list, tuple)) and len(alpha) == 1:
#             alpha = [alpha[0], alpha[0]]

#         self.alpha = alpha

#         if not isinstance(sigma, (float, Sequence)):
#             raise TypeError(f"sigma should be float or a sequence of floats. Got {type(sigma)}")
#         if isinstance(sigma, Sequence) and len(sigma) != 2:
#             raise ValueError(f"If sigma is a sequence its length should be 2. Got {len(sigma)}")
#         if isinstance(sigma, Sequence):
#             for element in sigma:
#                 if not isinstance(element, float):
#                     raise TypeError(f"sigma should be a sequence of floats. Got {type(element)}")

#         if isinstance(sigma, float):
#             sigma = [float(sigma), float(sigma)]
#         if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
#             sigma = [sigma[0], sigma[0]]

#         self.sigma = sigma

#         # Backward compatibility with integer value
#         if isinstance(interpolation, int):
#             warnings.warn(
#                 "Argument interpolation should be of type InterpolationMode instead of int. "
#                 "Please, use InterpolationMode enum."
#             )
#             interpolation = _interpolation_modes_from_int(interpolation)
#         self.interpolation = interpolation

#         if not isinstance(fill, (int, float)):
#             raise TypeError(f"fill should be int or float. Got {type(fill)}")
#         self.fill = fill

#     @staticmethod
#     def get_params(alpha: List[float], sigma: List[float], size: List[int]) -> Tensor:
#         dx = torch.rand([1, 1] + size) * 2 - 1
#         if sigma[0] > 0.0:
#             kx = int(8 * sigma[0] + 1)
#             # if kernel size is even we have to make it odd
#             if kx % 2 == 0:
#                 kx += 1
#             dx = F.gaussian_blur(dx, [kx, kx], sigma)
#         dx = dx * alpha[0] / size[0]

#         dy = torch.rand([1, 1] + size) * 2 - 1
#         if sigma[1] > 0.0:
#             ky = int(8 * sigma[1] + 1)
#             # if kernel size is even we have to make it odd
#             if ky % 2 == 0:
#                 ky += 1
#             dy = F.gaussian_blur(dy, [ky, ky], sigma)
#         dy = dy * alpha[1] / size[1]
#         return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

#     def forward(self, tensor: Tensor) -> Tensor:
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be transformed.
#         Returns:
#             PIL Image or Tensor: Transformed image.
#         """
#         _, height, width = F.get_dimensions(tensor)
#         displacement = self.get_params(self.alpha, self.sigma, [height, width])
#         return F.elastic_transform(tensor, displacement, self.interpolation, self.fill)

#     def __repr__(self):
#         format_string = self.__class__.__name__
#         format_string += f"(alpha={self.alpha}"
#         format_string += f", sigma={self.sigma}"
#         format_string += f", interpolation={self.interpolation}"
#         format_string += f", fill={self.fill})"
#         return format_string


class GaussianBlur(object):
    """
        PyTorch version of
        https://github.com/google-research/simclr/blob/244e7128004c5fd3c7805cf3135c79baa6c3bb96/data_util.py#L311
    """
    def gaussian_blur(self, image, sigma):
        image = image.reshape(1, 3, 224, 224)
        radius = np.int(self.kernel_size/2)
        kernel_size = radius * 2 + 1
        x = np.arange(-radius, radius + 1)

        blur_filter = np.exp(
              -np.power(x, 2.0) / (2.0 * np.power(np.float(sigma), 2.0)))
        blur_filter /= np.sum(blur_filter)

        conv1 = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), groups=3, padding=[kernel_size//2, 0], bias=False)
        conv1.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 0, 1])))

        conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), groups=3, padding=[0, kernel_size//2], bias=False)
        conv2.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 1, 0])))

        res = conv2(conv1(image))
        assert res.shape == image.shape
        return res[0]

    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            if np.random.uniform() < self.p:
                return self.gaussian_blur(img, sigma=np.random.uniform(0.2, 2))
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, p={1})'.format(self.kernel_size, self.p)

class CenterCropAndResize(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, proportion, size):
        self.proportion = proportion
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped and image.
        """
        w, h = (np.array(img.size) * self.proportion).astype(int)
        img = FT.resize(
            FT.center_crop(img, (h, w)),
            (self.size, self.size),
            interpolation=PIL.Image.BICUBIC
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(proportion={0}, size={1})'.format(self.proportion, self.size)


class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)


class MultiplyBatchSampler(torch.utils.data.sampler.BatchSampler):
    MULTILPLIER = 2

    def __iter__(self):
        for batch in super().__iter__():
            yield batch * self.MULTILPLIER


class ContinousSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, sampler, n_iterations):
        self.base_sampler = sampler
        self.n_iterations = n_iterations

    def __iter__(self):
        cur_iter = 0
        while cur_iter < self.n_iterations:
            for batch in self.base_sampler:
                yield batch
                cur_iter += 1
                if cur_iter >= self.n_iterations: return

    def __len__(self):
        return self.n_iterations
    
    def set_epoch(self, epoch):
        self.base_sampler.set_epoch(epoch)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort


class DummyOutputWrapper(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, dummy):
        self.dummy = dummy
        self.dataset = dataset

    def __getitem__(self, index):
        return (*self.dataset[index], self.dummy)

    def __len__(self):
        return len(self.dataset)
