import os
import tarfile
import cv2
import numpy
import math
import random
import constants

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

FONTS = [
    "fonts/Aller/Aller_Rg.ttf",
    "fonts/Aller/Aller_It.ttf",
    "fonts/Aller/Aller_Lt.ttf",
    "fonts/Aller/Aller_LtIt.ttf",
    "fonts/Aller/Aller_Bd.ttf",
    "fonts/Aller/Aller_BdIt.ttf",
    "fonts/open-sans/OpenSans-Regular.ttf",
    "fonts/open-sans/OpenSans-Italic.ttf",
    "fonts/open-sans/OpenSans-Light.ttf",
    "fonts/open-sans/OpenSans-LightItalic.ttf",
    "fonts/open-sans/OpenSans-Bold.ttf",
    "fonts/open-sans/OpenSans-BoldItalic.ttf",
    "fonts/pt-sans/PTC55F.ttf",
    "fonts/pt-sans/PTC75F.ttf",
    "fonts/pt-sans/PTS55F.ttf",
    "fonts/pt-sans/PTS56F.ttf",
    "fonts/pt-sans/PTS75F.ttf",
    "fonts/pt-sans/PTS76F.ttf",
    "fonts/raleway/Raleway-Medium.ttf",
    "fonts/raleway/Raleway-MediumItalic.ttf",
    "fonts/raleway/Raleway-Bold.ttf",
    "fonts/raleway/Raleway-BoldItalic.ttf",
    "fonts/raleway/Raleway-Light.ttf",
    "fonts/raleway/Raleway-LightItalic.ttf",
    "fonts/roboto/Roboto-Regular.ttf",
    "fonts/roboto/Roboto-Italic.ttf",
    "fonts/roboto/Roboto-Light.ttf",
    "fonts/roboto/Roboto-LightItalic.ttf",
    "fonts/roboto/Roboto-Bold.ttf",
    "fonts/roboto/Roboto-BoldItalic.ttf",
]

FONT_HEIGHT = 32

OUTPUT_SHAPE = (128, 128)

DATA_FOLDER = 'data/'

def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


def extract_backgrounds(archive_name, force=False):
    if not os.path.exists(DATA_FOLDER + "/bgs"):
        os.mkdir(DATA_FOLDER + "/bgs")
    elif not force:
        print('/bgs already present - Skipping extraction of %s.' % (archive_name))
        return

    t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f =  t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue
        
        if im.shape[0] > im.shape[1]:
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:
            im = cv2.resize(im, (256, 256))
        fname = DATA_FOLDER + "bgs/{:08}.jpg".format(index)
        print(fname)
        rc = cv2.imwrite(fname, im)
        if not rc:
            raise Exception("Failed to write file {}".format(fname))
        index += 1

def make_char_ims(output_height):
    font_size = output_height * 4
    fonts = []
    for font_path in FONTS:
        fonts.append(ImageFont.truetype(font_path, font_size))

    for c in constants.CHARS:
        char_ims = []
        for font in fonts:
            height = max(font.getsize(c)[1] for c in constants.CHARS)
            width = font.getsize(c)[0]
            im = Image.new("RGBA", (width, height), (0, 0, 0))

            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (255, 255, 255), font=font)
            scale = float(output_height) / height
            im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
            char_ims.append(numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.)
        yield c, char_ims


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    text_color = 0
    number_color = 0
    while first or number_color - text_color < 0.3:
        text_color = random.random()
        number_color = random.random()
        if text_color > number_color:
            text_color, number_color = number_color, text_color
        first = False
    return text_color, number_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_digits():
    code = ""
    length = random.randint(1, 5)
    for _ in range(length):
        code += random.choice(constants.CHARS)
    if code[0] == '0':
        return generate_digits()
    return code

def generate_number(font_height, char_ims):
    h_padding = random.uniform(1., 2.) * font_height
    v_padding = random.uniform(1., 2.) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)

    code = generate_digits()
    font_index = random.randint(0, len(FONTS)-1)
    text_width = sum(char_ims[c][font_index].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, number_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c][font_index]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    number = (numpy.ones(out_shape) * number_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)
    number_mask = (numpy.ones(out_shape) * 0.8)
    
    return number, number_mask, code.replace(" ", "")


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = DATA_FOLDER + "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)

    number, number_mask, code = generate_number(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=number.shape,
                            to_shape=bg.shape,
                            min_scale=0.5,
                            max_scale=2.0,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    number = cv2.warpAffine(number, M, (bg.shape[1], bg.shape[0]))
    number_mask = cv2.warpAffine(number_mask, M, (bg.shape[1], bg.shape[0]))

    out = number * number_mask + bg * (1.0 - number_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds


def generate_ims(num_images):
    char_ims = dict(make_char_ims(FONT_HEIGHT))
    num_bg_images = len(os.listdir(DATA_FOLDER + "bgs"))
    for i in range(num_images):
        yield generate_im(char_ims, num_bg_images)