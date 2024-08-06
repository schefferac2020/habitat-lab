import numpy as np

# Image cropping
def crop_around_pixel(img, center_x, center_y, crop_width, crop_height):
    img_height, img_width = img.shape[:2]

    half_width = crop_width // 2
    half_height = crop_height // 2

    start_x = max(center_x - half_width, 0)
    start_y = max(center_y - half_height, 0)
    end_x = min(center_x + half_width, img_width)
    end_y = min(center_y + half_height, img_height)

    cropped_image = np.zeros((crop_height, crop_width, img.shape[2]), dtype=np.uint8)

    crop_start_x = max(half_width - center_x, 0)
    crop_start_y = max(half_height - center_y, 0)
    crop_end_x = crop_start_x + (end_x - start_x)
    crop_end_y = crop_start_y + (end_y - start_y)

    cropped_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = img[start_y:end_y, start_x:end_x]

    return cropped_image

def center_image(background, overlay):
    # Get dimensions of both images
    bg_height, bg_width = background.shape[:2]
    ol_height, ol_width = overlay.shape[:2]

    # Compute the starting coordinates for centering
    start_x = (bg_width - ol_width) // 2
    start_y = (bg_height - ol_height) // 2

    # Create a copy of the background to place the overlay
    centered_image = background.copy()

    # Determine the region to place the overlay, ensuring it fits within the background
    end_x = min(start_x + ol_width, bg_width)
    end_y = min(start_y + ol_height, bg_height)
    ol_end_x = end_x - start_x
    ol_end_y = end_y - start_y

    # Copy the overlay image into the center of the background
    centered_image[start_y:end_y, start_x:end_x] = overlay[:ol_end_y, :ol_end_x]

    return centered_image

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]