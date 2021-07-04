import os
import random as rnd
import numpy as np
from PIL import Image, ImageFilter

from trdg import computer_text_generator, background_generator, distorsion_generator


def generate(
    text,
    size,
    extension,
    skewing_angle,
    random_skew,
    blur,
    random_blur,
    background_type,
    distorsion_type,
    distorsion_orientation,
    is_handwritten,
    name_format,
    width,
    alignment,
    text_color,
    orientation,
    space_width,
    character_spacing,
    margins,
    fit,
    output_mask,
    word_split,
    image_dir,
    stroke_width=0, 
    stroke_fill="#282828",
    image_mode="RGB", 
):
    image = None
    margin_top, margin_left, margin_bottom, margin_right = margins
    horizontal_margin = margin_left + margin_right
    vertical_margin = margin_top + margin_bottom
    txt_img = Image.open(text) .convert("RGBA")   
    data = np.asarray(txt_img).copy()
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if np.all(data[x, y]==255):
                data[x, y,0]=0
                data[x, y,1]=0
                data[x, y,2]=0
                data[x, y,3]=0
    image = Image.fromarray(data)
    #plt.imshow(txt_img)
    mask = Image.new("RGB", txt_img.size, (0, 0, 0))
    random_angle = rnd.randint(0 - skewing_angle, skewing_angle)
    rotated_img = image.rotate(
        skewing_angle if not random_skew else random_angle, expand=1
    )
    rotated_mask = mask.rotate(
        skewing_angle if not random_skew else random_angle, expand=1
    )
    #############################
    # Apply distorsion to image #
    #############################
    if distorsion_type == 0:
        distorted_img = rotated_img  # Mind = blown
        distorted_mask = rotated_mask
    elif distorsion_type == 1:
        distorted_img, distorted_mask = distorsion_generator.sin(
            rotated_img,
            rotated_mask,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
        )
    elif distorsion_type == 2:
        distorted_img, distorted_mask = distorsion_generator.cos(
            rotated_img,
            rotated_mask,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
        )
    else:
        distorted_img, distorted_mask = distorsion_generator.random(
            rotated_img,
            rotated_mask,
            vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
            horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
        )
    ##################################
    # Resize image to desired format #
    ##################################
    # Horizontal text
    if orientation == 0:
        new_width = int(
            distorted_img.size[0]
            * (float(size - vertical_margin) / float(distorted_img.size[1]))
        )
        resized_img = distorted_img.resize(
            (new_width, size - vertical_margin), Image.ANTIALIAS
        )
        resized_mask = distorted_mask.resize((new_width, size - vertical_margin), Image.NEAREST)
        background_width = width if width > 0 else new_width + horizontal_margin
        background_height = size
    # Vertical text
    elif orientation == 1:
        new_height = int(
            float(distorted_img.size[1])
            * (float(size - horizontal_margin) / float(distorted_img.size[0]))
        )
        resized_img = distorted_img.resize(
            (size - horizontal_margin, new_height), Image.ANTIALIAS
        )
        resized_mask = distorted_mask.resize(
            (size - horizontal_margin, new_height), Image.NEAREST
        )
        background_width = size
        background_height = new_height + vertical_margin
    else:
        raise ValueError("Invalid orientation")
    #############################
    # Generate background image #
    #############################
    if background_type == 0:
        background_img = background_generator.gaussian_noise(
            background_height, background_width
        )
    elif background_type == 1:
        background_img = background_generator.plain_white(
            background_height, background_width
        )
    elif background_type == 2:
        background_img = background_generator.quasicrystal(
            background_height, background_width
        )
    else:
        background_img = background_generator.image(
            background_height, background_width, image_dir
        )
    background_mask = Image.new(
        "RGB", (background_width, background_height), (0, 0, 0)
    )
    #############################
    # Place text with alignment #
    #############################
    new_text_width, _ = resized_img.size
    if alignment == 0 or width == -1:
        background_img.paste(resized_img, (margin_left, margin_top), resized_img)
        background_mask.paste(resized_mask, (margin_left, margin_top))
    elif alignment == 1:
        background_img.paste(
            resized_img,
            (int(background_width / 2 - new_text_width / 2), margin_top),
            resized_img,
        )
        background_mask.paste(
            resized_mask,
            (int(background_width / 2 - new_text_width / 2), margin_top),
        )
    else:
        background_img.paste(
            resized_img,
            (background_width - new_text_width - margin_right, margin_top),
            resized_img,
        )
        background_mask.paste(
            resized_mask,
            (background_width - new_text_width - margin_right, margin_top),
        )
    #######################
    # Apply gaussian blur #
    #######################
    gaussian_filter = ImageFilter.GaussianBlur(
        radius=blur if not random_blur else rnd.randint(0, blur)
    )
    final_image = background_img.filter(gaussian_filter)
    final_mask = background_mask.filter(gaussian_filter)
    
    ############################################
    # Change image mode (RGB, grayscale, etc.) #
    ############################################
    
    final_image = final_image.convert(image_mode)
    final_mask = final_mask.convert(image_mode) 
    #####################################
    # Generate name for resulting image #
    #####################################
    return final_image
