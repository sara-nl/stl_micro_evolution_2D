import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from scipy.ndimage import median_filter
from scipy import ndimage
import cv2

"""Utils for preprocessing the image data"""


def preprocess_data(
    images,
    target_size=(128, 128),
    n_frames_input=12,
    n_frames_output=12,
    denoise=False,
    augment=False,
    edge_enhance=False,
    contrast_factor=0.5,
):
    """
    Preprocesses the images for semantic segmentation training.

    Parameters:
    - images: A numpy array of images in the shape (T, H, W) where T is number of frames.
    - target_size: A tuple (height, width) for the output spatial dimensions.
    - n_frames_input: The number of frames to use as input for the model.
    - n_frames_output: The number of frames to use as labels for the model.
    - augment: Boolean flag to control whether to apply data augmentation.

    Returns:
    - A tuple of PyTorch tensors (input_tensor, label_tensor).
    """

    # Determine the actual number of input frames based on available data
    n_frames_input_actual = min(n_frames_input, (images.shape[0] + 1) // 2)
    n_frames_label_actual = images.shape[0] - n_frames_input_actual

    input_frames = images[:n_frames_input_actual]
    label_frames = images[
        n_frames_input_actual : n_frames_input_actual + n_frames_label_actual
    ]

    # Apply noise reduction
    if denoise:
        input_frames = remove_noise(input_frames, kernel_size=3)
        label_frames = remove_noise(label_frames, kernel_size=3)

    # Pad and then normalize
    processed_input = normalize_images(pad_images(input_frames, new_size=target_size))
    processed_label = normalize_images(pad_images(label_frames, new_size=target_size))

    # segmentation:
    # label_frames = images[
    #    n_frames_input_actual : n_frames_input_actual + n_frames_label_actual
    # ]
    processed_input = torch.from_numpy(processed_input).float().unsqueeze(1)
    processed_label = torch.from_numpy(processed_label).float().unsqueeze(1)

    if augment:
        angle = np.random.choice([90, 180, 270])
        # Apply the same rotation to all frames
        processed_input = torch.stack([rotate_tensor(frame, angle) for frame in processed_input])
        processed_label = torch.stack([rotate_tensor(frame, angle) for frame in processed_label])

        pass

    return (
        processed_input, processed_label
        # segmentation:
        # torch.from_numpy(padded_input).float().unsqueeze(1)
        # torch.from_numpy(padded_label).long() #label
    )


def rotate_tensor(tensor, angle):
    """
    Rotates a tensor image by one of the specified angles.

    Parameters:
    - tensor: A PyTorch tensor of the image.
    - angle: The angle to rotate by (90, 180, 270 degrees).

    Returns:
    - Rotated tensor.
    """
    if angle == 90:
        return tensor.permute(0, 2, 1).flip(1)
    elif angle == 180:
        return tensor.flip(1).flip(2)
    elif angle == 270:
        return tensor.permute(0, 2, 1).flip(2)
    else:
        return tensor

def adjust_contrast(images, factor):
    """
    Adjusts the contrast of an image array by scaling the range of intensity values.

    Parameters:
    - images: A numpy array of images with pixel values in range [0, 1].
    - factor: A factor by which to scale the contrast. A value of 1 gives the original image,
              less than 1 decreases contrast, and greater than 1 increases it.

    Returns:
    - The image array with adjusted contrast.
    """
    # Find the mean pixel value across the whole array
    mean = images.mean()

    # Scale pixel values away from or towards the mean based on the factor
    # Ensure that we clip the values to stay within [0, 1]
    return np.clip((1 + factor) * (images - mean) + mean, 0, 1)


def apply_filter(
    images,
    median_kernel_size=3,
):
    # median filter to reduce noise
    images_filtered = np.array(
        [median_filter(frame, size=median_kernel_size) for frame in input_frames]
    )

    return images_filtered


def enhance_edges(images):
    """
    Enhance the edges in each image using the Sobel filter.

    Parameters:
    - images: A numpy array of images in the shape (T, H, W)

    Returns:
    - A numpy array of images with enhanced edges.
    """
    # Sobel filter to find the horizontal and vertical edges
    sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_vertical = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    enhanced_images = np.zeros_like(images)
    for i, frame in enumerate(images):
        dx = ndimage.convolve(frame, sobel_horizontal)
        dy = ndimage.convolve(frame, sobel_vertical)
        # Combine the two directional edges
        enhanced_images[i] = np.hypot(dx, dy)
    return enhanced_images





def remove_noise(images, kernel_size=5):
    """
    Applies median filtering to each frame in a series of images to reduce noise.

    Args:
        images (numpy.ndarray): A numpy array of images in the shape (T, H, W),
                                where T is the number of frames.
        kernel_size (int): The size of the kernel. It must be an odd integer (3, 5, 7, ...).

    Returns:
        numpy.ndarray: The denoised series of images.
    """
    #print('noise is reduced')
    denoised_images = np.zeros_like(images)
    images = images.astype(np.uint8)

    for i in range(images.shape[0]):
        denoised_images[i] = cv2.medianBlur(images[i], kernel_size)
    return denoised_images


def normalize_images(images, rangemin=0.0, rangemax=199.0, mean = 0.5, std = 0.3):
    # Normalize images to [0, 1] range
    images_normalized = images / rangemax
    #return (images_normalized - mean) / std
    return images_normalized 


def pad_temporal_sequence(sequence, target_length=12, pad_side="both"):
    current_length = sequence.shape[0]
    padding_length = target_length - current_length
    if padding_length <= 0:
        return sequence

    padding = np.zeros(
        (padding_length, sequence.shape[-2], sequence.shape[-1]), dtype=np.float32
    )

    # Concatenate padding based on the side specified
    if pad_side == "start":
        padded_sequence = np.concatenate((padding, sequence), axis=0)
    elif pad_side == "end":
        padded_sequence = np.concatenate((sequence, padding), axis=0)
    else:  # 'both' or any other case, for future-proofing or error handling
        padded_sequence = sequence  # Default to no padding if 'both' or unrecognized

    return padded_sequence


def pad_images(images, new_size=(128, 128), image_mode="L"):
    """
    Pad a batch of images to a new size, centering the original image within the new dimensions.

    Parameters:
    - images (np.ndarray): A numpy array of images to be padded.
      Expected shape: (N, H, W) or (N, H, W, C) where
      N is the number of images, H is height, W is width, and C is channel.
    - new_size (tuple): The target size as a tuple (height, width), where the new size
      is expected to be larger than the original size of the images.
    - image_mode (str): The mode to be used for the new PIL images. Common modes include
      "L" (8-bit pixels, black and white), "RGB", and "RGBA". Default is "L".

    Returns:
    - np.ndarray: The padded images as a numpy array with the new dimensions.
    """
    if images.shape[-1] == new_size[-1]:
        return images
    padded_images = []
    for image in images:
        # Create a new image with the desired size and black background
        # 'L' mode is for (8-bit pixels, black and white)
        padded_img = Image.new(image_mode, new_size, color=0)

        pil_img = Image.fromarray(image)

        # Calculate padding sizes
        width, height = pil_img.size
        left = (new_size[0] - width) // 2
        top = (new_size[1] - height) // 2

        padded_img.paste(pil_img, (left, top))
        padded_images.append(np.array(padded_img))

    return np.array(padded_images)


def resize_images(images, new_size=(64, 64)):
    """
    Resize a batch of images to a new size, maintaining aspect ratio.

    Parameters:
    - images (np.ndarray): A numpy array of images to be resized.
      Expected shape: (N, H, W) or (N, H, W, C) where
      N is the number of images, H is height, W is width, and C is channel.
    - new_size (tuple): The target size as a tuple (height, width).

    Returns:
    - np.ndarray: The resized images as a numpy array with the same dimensions.
    """
    resized_images = []
    for image in images:
        # Calculate indices for cropping
        start_x = (image.shape[0] - new_size[0]) // 2
        start_y = (image.shape[1] - new_size[1]) // 2

        # Crop the image
        cropped_image = image[
            start_x : start_x + new_size[0], start_y : start_y + new_size[1]
        ]

        # Convert the cropped image to a PIL image and add it to the list
        pil_img = Image.fromarray(cropped_image)
        resized_images.append(np.array(pil_img))

    return np.array(resized_images)


def unpad_images(padded_images, original_size=(100, 100)):
    # Assuming the padding was added equally on all sides
    # Calculate the starting and ending indices to slice
    pad_height = (padded_images.shape[1] - original_size[0]) // 2
    pad_width = (padded_images.shape[2] - original_size[1]) // 2

    # Adjust if the padding added an extra pixel for odd differences
    pad_height_extra = (padded_images.shape[1] - original_size[0]) % 2
    pad_width_extra = (padded_images.shape[2] - original_size[1]) % 2

    # Slice out the padding to get back the original images
    unpadded_images = padded_images[
        :,  # Keep all images/channels in the first dimension
        pad_height : padded_images.shape[1]
        - pad_height
        - pad_height_extra,  # Remove padding from the second dimension
        pad_width : padded_images.shape[2]
        - pad_width
        - pad_width_extra,  # Remove padding from the third dimension
    ]

    return unpadded_images


def tensor_to_original_array(tensor):
    """
    Send to CPU
    Reverse the normalization, remove channel dimension, convert to numpy int
    """
    numpy_array = tensor.cpu().numpy()
    if tensor.shape[0] == 1:  # Grayscale
        numpy_array = numpy_array.squeeze(0)
    elif tensor.shape[0] == 3:  # RGB
        # Permute the dimensions to [H, W, C]
        numpy_array = numpy_array.transpose(1, 2, 0)
    else:
        raise ValueError("Unsupported channel size: {}".format(tensor.shape[0]))

    # assume the tensor was normalized to [0,1]
    return numpy_array * 199.0


if __name__ == "__main__":
    datafile = [
        ("exp_1_complete_2D.h5", 90, 4),
        # ("exp_4_len_15_2D.h5", 15, 4000),
        # ("exp_4_len_28_2D.h5", 28, 4000),
        # ("exp_5_len_28_2D.h5", 28, 4000),
        # ("exp_6_len_30_2D.h5", 30, 4000),
        # ("exp_6_len_28_2D.h5", 28, 4000),
    ]

    exp_indices = [(1, 0, 90)]
