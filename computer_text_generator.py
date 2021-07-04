import random as rnd

from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
def generate(
    text,
    font,
    text_color,
    font_size,
    orientation,
    space_width,
    character_spacing,
    fit,
    word_split,
    stroke_width=0, 
    stroke_fill="#282828",
):
    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text, font, text_color, font_size, space_width, character_spacing, fit,
            stroke_width, stroke_fill, 
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text(
    text, font, text_color, font_size, space_width, character_spacing, fit, word_split, 
    stroke_width=0, stroke_fill="#282828"
):

    txt_img = Image.open(text) .convert("RGBA")   
    data = np.asarray(txt_img).copy()
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if np.all(data[x, y]==255):
                data[x, y,0]=0
                data[x, y,1]=0
                data[x, y,2]=0
                data[x, y,3]=0

    txt_img = Image.fromarray(data)
    #plt.imshow(txt_img)
    txt_mask = Image.new("RGB", txt_img.size, (0, 0, 0))

    return txt_img, txt_mask


def _generate_vertical_text(
    text, font, text_color, font_size, space_width, character_spacing, fit,
    stroke_width=0, stroke_fill="#282828"
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1] 

    stroke_fill = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=(i // (255 * 255), i // 255, i % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask
