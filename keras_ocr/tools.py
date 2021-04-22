# pylint: disable=invalid-name,too-many-branches,too-many-statements,too-many-arguments
import os
import io
import typing
import hashlib
import urllib.request
import urllib.parse


import cv2
import imgaug
import numpy as np
import pandas as pd
import validators
import matplotlib.pyplot as plt
from shapely import geometry
from scipy import spatial

import tqdm.notebook
from xml.etree import ElementTree
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow_addons import image as image_tfa



from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from keras import layers
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow_addons.image import connected_components
_image_so = LazySO("custom_ops/image/_image_ops.so")

from . import recognition, detection


def read(filepath_or_buffer: typing.Union[str, io.BytesIO]):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, 'read'):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str):
        if validators.url(filepath_or_buffer):
            return read(urllib.request.urlopen(filepath_or_buffer))
        assert os.path.isfile(filepath_or_buffer), \
            'Could not find image at path: ' + filepath_or_buffer
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_rotated_width_height(box):
    """
    Returns the width and height of a rotated rectangle

    Args:
        box: A list of four points starting in the top left
        corner and moving clockwise.
    """
    w = (spatial.distance.cdist(box[0][np.newaxis], box[1][np.newaxis], "euclidean") +
         spatial.distance.cdist(box[2][np.newaxis], box[3][np.newaxis], "euclidean")) / 2
    h = (spatial.distance.cdist(box[0][np.newaxis], box[3][np.newaxis], "euclidean") +
         spatial.distance.cdist(box[1][np.newaxis], box[2][np.newaxis], "euclidean")) / 2
    return int(w[0][0]), int(h[0][0])


# pylint:disable=too-many-locals
def warpBox(image,
            box,
            target_height=None,
            target_width=None,
            margin=0,
            cval=None,
            ):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.

    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        return_transform: Whether to return the transformation
            matrix with the image.
    """  

    if cval is None:
        cval = (0, 0, 0) if len(image.shape) == 3 else 0
        
    w, h =  abs(box[2][0] - box[0][0]), abs(box[2][1] - box[0][1])
    scale = min(target_width / w, target_height / h)
    crop = image[min(int(box[0][1]),int(box[2][1])):max(int(box[0][1]),int(box[2][1])),
                 min(int(box[0][0]), int(box[2][0]) ) : max(int(box[0][0]), int(box[2][0]) )]
    crop = cv2.resize(crop, (int(w*scale), int(h*scale)))
    target_shape = (target_height, target_width, 3) if len(image.shape) == 3 else (target_height,
                                                                                   target_width)    
    full = (np.zeros(target_shape) + cval).astype('uint8')
    full[:crop.shape[0], :crop.shape[1]] = crop
    return full


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def combine_line(line):
    """Combine a set of boxes in a line into a single bounding
    box.

    Args:
        line: A list of (box, character) entries

    Returns:
        A (box, text) tuple
    """
    text = ''.join([character if character is not None else '' for _, character in line])
    box = np.concatenate([coords[:2] for coords, _ in line] +
                         [np.array([coords[3], coords[2]])
                          for coords, _ in reversed(line)]).astype('float32')
    first_point = box[0]
    rectangle = cv2.minAreaRect(box)
    box = cv2.boxPoints(rectangle)

    # Put the points in clockwise order
    box = np.array(np.roll(box, -np.linalg.norm(box - first_point, axis=1).argmin(), 0))
    return box, text


def drawAnnotations(image, predictions, ax=None):
    """Draw text annotations onto image.

    Args:
        image: The image on which to draw
        predictions: The predictions as provided by `pipeline.recognize`.
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(drawBoxes(image=image, boxes=predictions, boxes_format='predictions'))
    predictions = sorted(predictions, key=lambda p: p[1][:, 1].min())
    left = []
    right = []
    for word, box in predictions:
        if box[:, 0].min() < image.shape[1] / 2:
            left.append((word, box))
        else:
            right.append((word, box))
    ax.set_yticks([])
    ax.set_xticks([])
    for side, group in zip(['left', 'right'], [left, right]):
        for index, (text, box) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]
            ax.annotate(s=text,
                        xy=xy,
                        xytext=(-0.05 if side == 'left' else 1.05, y),
                        xycoords='axes fraction',
                        arrowprops={
                            'arrowstyle': '->',
                            'color': 'r'
                        },
                        color='r',
                        fontsize=14,
                        horizontalalignment='right' if side == 'left' else 'left')
    return ax


def drawBoxes(image, boxes, color=(255, 0, 0), thickness=5, boxes_format='boxes'):
    """Draw boxes onto an image.

    Args:
        image: The image on which to draw the boxes.
        boxes: The boxes to draw.
        color: The color for each box.
        thickness: The thickness for each box.
        boxes_format: The format used for providing the boxes. Options are
            "boxes" which indicates an array with shape(N, 4, 2) where N is the
            number of boxes and each box is a list of four points) as provided
            by `keras_ocr.detection.Detector.detect`, "lines" (a list of
            lines where each line itself is a list of (box, character) tuples) as
            provided by `keras_ocr.data_generation.get_image_generator`,
            or "predictions" where boxes is by itself a list of (word, box) tuples
            as provided by `keras_ocr.pipeline.Pipeline.recognize` or
            `keras_ocr.recognition.Recognizer.recognize_from_boxes`.
    """
    if len(boxes) == 0:
        return image
    canvas = image.copy()
    if boxes_format == 'lines':
        revised_boxes = []
        for line in boxes:
            for box, _ in line:
                revised_boxes.append(box)
        boxes = revised_boxes
    if boxes_format == 'predictions':
        revised_boxes = []
        for _, box in boxes:
            revised_boxes.append(box)
        boxes = revised_boxes
    for box in boxes:
        cv2.polylines(img=canvas,
                      pts=box[np.newaxis].astype('int32'),
                      color=color,
                      thickness=thickness,
                      isClosed=True)
    return canvas


def adjust_boxes(boxes, boxes_format='boxes', scale=1):
    """Adjust boxes using a given scale and offset.

    Args:
        boxes: The boxes to adjust
        boxes_format: The format for the boxes. See the `drawBoxes` function
            for an explanation on the options.
        scale: The scale to apply
    """
    if scale == 1:
        return boxes
    if boxes_format == 'boxes':
        return np.array(boxes) * scale
    if boxes_format == 'lines':
        return [[(np.array(box) * scale, character) for box, character in line] for line in boxes]
    if boxes_format == 'predictions':
        return [(word, np.array(box) * scale) for word, box in boxes]
    raise NotImplementedError(f'Unsupported boxes format: {boxes_format}')


def augment(boxes,
            augmenter: imgaug.augmenters.meta.Augmenter,
            image=None,
            boxes_format='boxes',
            image_shape=None,
            area_threshold=0.5,
            min_area=None):
    """Augment an image and associated boxes together.

    Args:
        image: The image which we wish to apply the augmentation.
        boxes: The boxes that will be augmented together with the image
        boxes_format: The format for the boxes. See the `drawBoxes` function
            for an explanation on the options.
        image_shape: The shape of the input image if no image will be provided.
        area_threshold: Fraction of bounding box that we require to be
            in augmented image to include it.
        min_area: The minimum area for a character to be included.
    """
    if image is None and image_shape is None:
        raise ValueError('One of "image" or "image_shape" must be provided.')
    augmenter = augmenter.to_deterministic()

    if image is not None:
        image_augmented = augmenter(image=image)
        image_shape = image.shape[:2]
        image_augmented_shape = image_augmented.shape[:2]
    else:
        image_augmented = None
        width_augmented, height_augmented = augmenter.augment_keypoints(
            imgaug.KeypointsOnImage.from_xy_array(xy=[[image_shape[1], image_shape[0]]],
                                                  shape=image_shape)).to_xy_array()[0]
        image_augmented_shape = (height_augmented, width_augmented)

    def box_inside_image(box):
        area_before = cv2.contourArea(np.int32(box)[:, np.newaxis, :])
        if area_before == 0:
            return False, box
        clipped = box.copy()
        clipped[:, 0] = clipped[:, 0].clip(0, image_augmented_shape[1])
        clipped[:, 1] = clipped[:, 1].clip(0, image_augmented_shape[0])
        area_after = cv2.contourArea(np.int32(clipped)[:, np.newaxis, :])
        return ((area_after / area_before) >= area_threshold) and (min_area is None or
                                                                   area_after > min_area), clipped

    def augment_box(box):
        return augmenter.augment_keypoints(
            imgaug.KeypointsOnImage.from_xy_array(box, shape=image_shape)).to_xy_array()

    if boxes_format == 'boxes':
        boxes_augmented = [
            box for inside, box in [box_inside_image(box) for box in map(augment_box, boxes)]
            if inside
        ]
    elif boxes_format == 'lines':
        boxes_augmented = [[(augment_box(box), character) for box, character in line]
                           for line in boxes]
        boxes_augmented = [[(box, character)
                            for (inside, box), character in [(box_inside_image(box), character)
                                                             for box, character in line] if inside]
                           for line in boxes_augmented]
        # Sometimes all the characters in a line are removed.
        boxes_augmented = [line for line in boxes_augmented if line]
    elif boxes_format == 'predictions':
        boxes_augmented = [(word, augment_box(box)) for word, box in boxes]
        boxes_augmented = [(word, box) for word, (inside, box) in [(word, box_inside_image(box))
                                                                   for word, box in boxes_augmented]
                           if inside]
    else:
        raise NotImplementedError(f'Unsupported boxes format: {boxes_format}')
    return image_augmented, boxes_augmented


def pad(image, width: int, height: int, cval: int = 255):
    """Pad an image to a desired size. Raises an exception if image
    is larger than desired size.

    Args:
        image: The input image
        width: The output width
        height: The output height
        cval: The value to use for filling the image.
    """
    if len(image.shape) == 3:
        output_shape = (height, width, image.shape[-1])
    else:
        output_shape = (height, width)
    assert height >= output_shape[0], 'Input height must be less than output height.'
    assert width >= output_shape[1], 'Input width must be less than output width.'
    padded = np.zeros(output_shape, dtype=image.dtype) + cval
    padded[:image.shape[0], :image.shape[1]] = image
    return padded


def resize_image(image, max_scale, max_size):
    """Obtain the optimal resized image subject to a maximum scale
    and maximum size.

    Args:
        image: The input image
        max_scale: The maximum scale to apply
        max_size: The maximum size to return
    """
    if max(image.shape) * max_scale > max_size:
        # We are constrained by the maximum size
        scale = max_size / max(image.shape)
    else:
        # We are contrained by scale
        scale = max_scale
    return cv2.resize(image,
                      dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale))), scale


# pylint: disable=too-many-arguments
def fit(image, width: int, height: int, cval: int = 255, mode='letterbox', return_scale=False):
    """Obtain a new image, fit to the specified size.

    Args:
        image: The input image
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image
        return_scale: Whether to return the scale used for the image

    Returns:
        The new image
    """
    fitted = None
    x_scale = width / image.shape[1]
    y_scale = height / image.shape[0]
    if x_scale == 1 and y_scale == 1:
        fitted = image
        scale = 1
    elif (x_scale <= y_scale and mode == 'letterbox') or (x_scale >= y_scale and mode == 'crop'):
        scale = width / image.shape[1]
        resize_width = width
        resize_height = (width / image.shape[1]) * image.shape[0]
    else:
        scale = height / image.shape[0]
        resize_height = height
        resize_width = scale * image.shape[1]
    if fitted is None:
        resize_width, resize_height = map(int, [resize_width, resize_height])
        if mode == 'letterbox':
            fitted = np.zeros((height, width, 3), dtype='uint8') + cval
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted[:image.shape[0], :image.shape[1]] = image[:height, :width]
        elif mode == 'crop':
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted = image[:height, :width]
        else:
            raise NotImplementedError(f'Unsupported mode: {mode}')
    if not return_scale:
        return fitted
    return fitted, scale


def read_and_fit(filepath_or_array: typing.Union[str, np.ndarray],
                 width: int,
                 height: int,
                 cval: int = 255,
                 mode='letterbox'):
    """Read an image from disk and fit to the specified size.

    Args:
        filepath: The path to the image or numpy array of shape HxWx3
        width: The new width
        height: The new height
        cval: The constant value to use to fill the remaining areas of
            the image
        mode: The mode to pass to "fit" (crop or letterbox)

    Returns:
        The new image
    """
    image = read(filepath_or_array) if isinstance(filepath_or_array, str) else filepath_or_array
    image = fit(image=image, width=width, height=height, cval=cval, mode=mode)
    return image


def sha256sum(filename):
    """Compute the sha256 hash for a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def get_default_cache_dir():
    return os.environ.get('KERAS_OCR_CACHE_DIR', os.path.expanduser(os.path.join('~',
                                                                                 '.keras-ocr')))


def download_and_verify(url, sha256=None, cache_dir=None, verbose=True, filename=None):
    """Download a file to a cache directory and verify it with a sha256
    hash.

    Args:
        url: The file to download
        sha256: The sha256 hash to check. If the file already exists and the hash
            matches, we don't download it again.
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.
        verbose: Whether to log progress
        filename: The filename to use for the file. By default, the filename is
            derived from the URL.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if filename is None:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    filepath = os.path.join(cache_dir, filename)
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    if verbose:
        print('Looking for ' + filepath)
    if not os.path.isfile(filepath) or (sha256 and sha256sum(filepath) != sha256):
        if verbose:
            print('Downloading ' + filepath)
        urllib.request.urlretrieve(url, filepath)
    #assert sha256 is None or sha256 == sha256sum(filepath), 'Error occurred verifying sha256.'
    return filepath


# pylint: disable=bad-continuation
def get_rotated_box(
    points
) -> typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float], typing.Tuple[
        float, float], typing.Tuple[float, float], float]:
    """Obtain the parameters of a rotated box.

    Returns:
        The vertices of the rotated box in top-left,
        top-right, bottom-right, bottom-left order along
        with the angle of rotation about the bottom left corner.
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation


def fix_line(line):
    """Given a list of (box, character) tuples, return a revised
    line with a consistent ordering of left-to-right or top-to-bottom,
    with each box provided with (top-left, top-right, bottom-right, bottom-left)
    ordering.

    Returns:
        A tuple that is the fixed line as well as a string indicating
        whether the line is horizontal or vertical.
    """
    line = [(get_rotated_box(box)[0], character) for box, character in line]
    centers = np.array([box.mean(axis=0) for box, _ in line])
    sortedx = centers[:, 0].argsort()
    sortedy = centers[:, 1].argsort()
    if np.diff(centers[sortedy][:, 1]).sum() > np.diff(centers[sortedx][:, 0]).sum():
        return [line[idx] for idx in sortedy], 'vertical'
    return [line[idx] for idx in sortedx], 'horizontal'


###### One graph utils ############

@tf.function
def decode_bbox(bboxes, bbox_num, res):
    "decodes from [y, x,  h, w] to [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]"
    for i in tf.range(bbox_num):
        x1, x2, y1, y2 = bboxes[i][1] * 2, (bboxes[i][1] + bboxes[i][3]) * 2, bboxes[i][0] * 2, (
                    bboxes[i][0] + bboxes[i][2]) * 2
        # rescale
        res = res.write(i, [[x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]])

    return res.stack()



class ComputeInputLayer(tf.keras.layers.Layer):

    def call(self, input):
        mean = tf.constant([123.6, 116.3, 103.5])
        variance = tf.constant([58.3, 57.12, 57.38])

        input -= mean  # * 255
        input /= variance  # * 255
        return input


class BboxLayer(tf.keras.layers.Layer):
    # def __init__(self, num_outputs):
    def __init__(self):
        # for dilation

        self.filters = tf.ones([3, 3, 1], dtype=tf.dtypes.int32)
        self.strides = [1., 1., 1., 1.]
        self.padding = "SAME"
        self.dilations = [1., 2., 2., 1.]

        super(BboxLayer, self).__init__()
        # self.num_outputs = num_outputs
        # pass

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()

    def get_bboxes(self, input):
        # : Tensor where every layer is single dlated connected component area

        coords_tensor = tf.where(input)

        coords_tensor = tf.cast(coords_tensor, tf.int32)
        y1 = tf.reduce_max(coords_tensor[:, 0])
        y2 = tf.reduce_min(coords_tensor[:, 0])

        x1 = tf.reduce_max(coords_tensor[:, 1])
        x2 = tf.reduce_min(coords_tensor[:, 1])

        w = tf.cast(x1 - x2, tf.int32)
        h = tf.cast(y1 - y2, tf.int32)

        return y2, x2, h, w

    def clear_coords(self, xyhw):
        # : input - raw list of xyhw tensors

        index = tf.math.logical_and(xyhw[:, 2] < 300, xyhw[:, 2] > 4)
        return xyhw[index]

    def call(self, input):
        input_shape = array_ops.shape(input)

        textmap = tf.identity(input[0, :, :, 0])
        self.textmap = textmap
        linkmap = tf.identity(input[0, :, :, 1])

        textmap = tf.where(textmap > 0.4, 1.0, 0)
        linkmap = tf.where(linkmap > 0.4, 1.0, 0)
        res_img = tf.image.convert_image_dtype(tf.clip_by_value((textmap + linkmap), 0, 1), tf.float32)

        res_img = image_tfa.connected_components(res_img)
        # self.res_img = res_img
        elem_num = tf.reduce_max(res_img) + 1

        # separate connected components to different layers
        # Transform connected components image to matrix [elem_num x H x W]

        # area_ids_vector = tf.range(elem_num)
        components_ids_vector = tf.range(elem_num)

        components_ids_matrix = tf.tensordot(components_ids_vector, tf.ones_like(res_img),
                                             axes=0)  # now every connected components id has its own layer

        # create mask matrix
        connected_components_img_matrix = tf.tensordot(tf.ones_like(components_ids_vector), res_img,
                                                       axes=0)  # create [elem_num x H x W] tensor

        connected_components_img_matrix = tf.where(connected_components_img_matrix == components_ids_matrix, 1, 0)

        # dlate every layers mask to increase bbox area

        connected_components_img_matrix = tf.expand_dims(connected_components_img_matrix, -1)

        dlate_component = tf.nn.dilation2d(connected_components_img_matrix,
                                           self.filters,
                                           self.strides,
                                           self.padding,

                                           "NHWC",
                                           self.dilations)[:, :, :, 0]

        dlate_component = tf.where(dlate_component == 1, False, True)

        # calculate index of dlatet regions where text value < 0.7 and drop them
        # text_index = tf.map_fn(fn=self.calculate_max_text_value_index, elems=dlate_component)

        # dlate_component = dlate_component[text_index]

        xyhw = tf.map_fn(fn=self.get_bboxes, elems=dlate_component, dtype=(tf.int32, tf.int32, tf.int32, tf.int32))

        # result is [x[0..elem_num], y[0..elem_num],h[0..elem_num], w[0..elem_num] ]. Need Stack and transpose for receive single tensor

        xyhw = tf.stack(xyhw)
        xyhw = tf.transpose(xyhw)

        return self.clear_coords(xyhw)


class GrayScaleLayer(tf.keras.layers.Layer):

    def call(self, input):
        input_shape = array_ops.shape(input)

        img_hd = input_shape[1] / 2
        img_wd = input_shape[2] / 2
        input = tf.image.resize(input, [img_hd, img_wd])

        return tf.cast(tf.image.rgb_to_grayscale(input), tf.uint8) / 255


class CropBboxesLayer(tf.keras.layers.Layer):  # (PreprocessingLayer):

    def resize_crop(self,
                    box,
                    target_height=31,
                    target_width=200):
        ## for recognition ##

        crop = tf.image.crop_to_bounding_box(self.image, box[0], box[1], box[2], box[3])
        scale = tf.math.minimum(target_width / box[3], target_height / box[2])

        scaled_shape = [tf.cast(box[2], tf.float64) * scale, tf.cast(box[3], tf.float64) * scale]
        scaled_shape = tf.cast(scaled_shape, tf.int32)
        # calculate padding for target w, h
        pad_h = target_height - tf.cast(scaled_shape[0], tf.int32)
        pad_w = target_width - tf.cast(scaled_shape[1], tf.int32)

        result_img = tf.image.resize(crop, scaled_shape)[0]
        result_img = tf.pad(result_img, [[pad_h, 0], [0, pad_w], [0, 0]], "CONSTANT", constant_values=0)

        return result_img  # tf.image.resize(crop, scaled_shape)

    def call(self, inputs):
        # inputs [image, bboxes]

        self.image = inputs[0]

        return tf.map_fn(fn=self.resize_crop, elems=inputs[1], dtype=tf.float32)

class DecodeCharLayer(tf.keras.layers.Layer):

    def __init__(self, alphabet):
        super(DecodeCharLayer, self).__init__()

        self.alphabet = alphabet

    def call(self, input):
        return decode_prediction(input, self.alphabet)


class DecodeBoxLayer(tf.keras.layers.Layer):

    def call(self, input):
        box_num = array_ops.shape(input)[0]
        res = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)

        return decode_bbox(input, box_num, res)


def get_recognition_part(weights, recognizer_alphabet, build_params):
    backbone, model, training_model, prediction_model = recognition.build_model(recognizer_alphabet, **build_params)


    prediction_model.load_weights(weights)

    return prediction_model


def create_one_grap_model(detector_weights, recognizer_weights, recognizer_alphabet, prod=False, build_params=recognition.DEFAULT_BUILD_PARAMS):
    # if debug - output bbox images, not rectangles
    # build_params - for recognition part
    recognizer_predict_model = get_recognition_part(recognizer_weights, recognizer_alphabet, build_params)
    detector = detection.Detector(weights='clovaai_general')

    detector.model.load_weights(detector_weights)

    rec_inp = tf.keras.Input([None, None, 3])
    normilized_inp = ComputeInputLayer()(rec_inp)

    bbox_model = detector.model(normilized_inp)

    bbox_model = BboxLayer()(bbox_model)

    grayscale_model = GrayScaleLayer()(rec_inp)

    rec_model_func = tf.keras.models.Model(inputs=rec_inp, outputs=[bbox_model, grayscale_model])

    bboxes_model = CropBboxesLayer()([grayscale_model, bbox_model])
    bboxes_model = keras.models.Model(inputs=rec_inp, outputs=bboxes_model)
    final_model = recognizer_predict_model(bboxes_model.output)
    if prod:
        # xyhw bbox format
        return keras.models.Model(inputs=rec_inp, outputs=[bbox_model, final_model, ])

    else:
        decoded_bboxes = DecodeBoxLayer()(bbox_model)
        return keras.models.Model(inputs=rec_inp, outputs=[decoded_bboxes, final_model, ])  # result for  [[4,2]] bbox shape


# create pipeline from one graph model

class OneGraphPipeline():
    def __init__(self, model, alphabet):
        self.model = model
        self.blank_label_idx = len(alphabet)
        self.alphabet = alphabet

    def decode_prediction(self, raw_predict):  # from numerical to char groups

        predictions = [
            ''.join([self.alphabet[idx] for idx in row if idx not in [self.blank_label_idx, -1]])
            for row in raw_predict
        ]

        return predictions

    def get_prediction_groups(self, bboxes, char_groups):
        return [
            list(zip(predictions, boxes))
            for predictions, boxes in zip([char_groups], [bboxes])
        ]

    def compute_input(self, image):
        # should be RGB order
        # image = image.astype('float32')
        mean = np.array([0.485, 0.456, 0.406])
        variance = np.array([0.229, 0.224, 0.225])

        image -= mean
        image /= variance
        return image

    def recognize(self, images):
        # images = self.compute_input(images)

        raw_predict = self.model.predict([images])
        char_groups = self.decode_prediction(raw_predict[1])

        return self.get_prediction_groups(raw_predict[0], char_groups)


def initialize_image_ops():
    try:
        inp = tf.keras.Input([None, None, 2])
        init_model = BboxLayer()(inp)

        print('custom operation initialized...')

        return True

    except Exception as e:

        return e


#### QAcheck utils ####

# check precision part

def objects_bb(xml_file, object_name):
    "    возвращает список словарей с координтами.\
    для объектов мани, кардс , озер.\
    "    
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    list_box = []
    for i in root:
        b_boxes = {}
        if i.tag == 'object' and i[0].text\
        == object_name:  
            for el in i:
                values = []
                if el.tag == 'attributes':                    
                    value = el.text.split('=')[1]  
                    values += [value]                   
                elif el.tag == 'polygon':  
                    coord = []
                    for j in el:                       
                        if j.tag == "pt":
                            point = []
                            for c in j:                            
                                if c.tag == 'x' or c.tag == 'y':                              
                                    point += [int(float(c.text))]                                     
                            coord+= [point]
            b_boxes[values[-1]] = ([coord[0][0], coord[0][1]], \
                                   [coord[1][0], coord[1][1]],
                                   [coord[2][0],coord[2][1]],     
                                   [coord[3][0], coord[3][1]])
            list_box+=[b_boxes]
            
    return list_box

def got_gt_objects(xml_file):
    "возвращает три набора бб"
    cards = objects_bb(xml_file,'cards')
    for i in cards: # drop single suits
        if len(list(i)[0]) == 1:     
            cards.remove(i)
    money = objects_bb(xml_file,'money')
    other = objects_bb(xml_file,'other')
    return [cards,money,other]


def convert_money(money_str):  
    "затирает окончания денег и сравнивает предикты с\
    граунд тру по абсолютному значению"
    
    
    money_str = money_str.replace(',','.')
    end_money = ['p','P','$']
    cut = []
    for i in range(len(money_str)):
        if money_str[i] in end_money:
            cut+=[i]
    if len(cut) == 0:
        return money_str
    else:
        return money_str[:cut[0]]


def precision(gt_object, predict, money_mode=False):
    
    "возвращает точность как отношения длинны\
    предсказаний, которые удовлетворяют условию \
    равенства ключа из граунд тру с распознанным \
    текстом из предикта и условию наложения левой\
    верхней точки координат предикта и граунд тру"
    
    def compare_difference(number_a, number_b):
        "сравнивает координаты для\
        определения наложения точек"
        delta = 10
        max_first = max(number_a,number_b)
        min_first = min(number_a,number_b)
        if min_first+delta >=max_first:
            return True
        else:
            return False   
    
    predict = predict[0]
    correct_answer = 0
    for i in gt_object:# итерация по ключам из хмл 
        key = list(i)[0]
        if money_mode == True:
            gt_val = convert_money(key)
        else:    
            gt_val = key
            
        true_coord = i[key]
        first_true = true_coord[0]
        second_true = true_coord[1]    
        for i in range(len(predict)):
            if money_mode == True:
                predict_val = convert_money(predict[i][0])
            else:
                predict_val = predict[i][0]
            
            if predict_val == gt_val:
                predict_coord = predict[i][1]
                first_predict = predict_coord[0]
                second_predict = predict_coord[2]             
                if compare_difference(first_predict[0],first_true[0]) == True\
                and compare_difference(first_predict[1],first_true[1]) == True:
                    correct_answer+=1
   
    return round((correct_answer/len(gt_object)),2)

def count_precision(xml_file, predict_file):
    "возвращает итоговый пресижен для \
    кардс,мани, озер"
    gt_obj = got_gt_objects(xml_file)
    cards = gt_obj[0]
    money = gt_obj[1]
    other = gt_obj[2]
  
    return precision(cards, predict_file), \
        precision(money, predict_file, money_mode = True), \
        precision(other, predict_file)



  
def got_statistic_word(pred, obj):   
    
    "возвращает по заданному объекту (карты, \
     деньги или прочее)строку, \
    содержащую все не расознанные или ошибочно\
    распознанные символы"    
    
       
    def _compare_words(g_true, predict):

        "находит не распознанные \
        или ошибочно распознанные символы\
        в предикте"

        answer = str()
        for i in g_true:
            if i not in predict:
                answer += i
        return answer


     def _compare_difference(number_a, number_b):

        "сравнивает координаты для\
        определения наложения точек"

        delta = 10 # погрешность +-
        max_first = max(number_a,number_b)
        min_first = min(number_a,number_b)
        if min_first+delta >=max_first:
            return True
        else:
            return False 

    def _compare_coordinate(a,b,i,j):

        "сравнивает 4 координаты диагонали\
            ректов"

        passed = True    
        passed &=_compare_difference(a,i)
        passed &=_compare_difference(b,j)

        return passed    
    
    ans = str()
    for j in pred:  
        c = j[1]
        point_pred = list(c[0])    
        for i in obj:
            point_true = list(i.values())[0][0]
            key = list(i.keys())[0]
            if _compare_coordinate(
                point_pred[0], point_pred[1],\
                point_true[0], point_true[1]) == True:
                ans += (_compare_words(key,j[0]))
     
    return ans


def full_error_statistic(xml_file, predict):
    
    "полная статистика по трем объектам"
    
    ground_true = got_gt_objects(xml_file)
    cards = got_statistic_word(predict, ground_true[0])
    money = got_statistic_word(predict, ground_true[1])
    other = got_statistic_word(predict, ground_true[2])    
  
    return (cards, money, other)


def quality_df(images_paths, xmls_paths, pipeline):

    @contract(my_list='list',returns='dict')
    def _count_frequency(my_list):
      
      "подсчитывает число вхождений\
      символов в список"
      
      count = {}
      for i in my_list:
          count[i] = count.get(i, 0) + 1
      return count

    quality_results = {    
        'image_name': [],
        'card_acc': [],        
        'money_acc': [],    
        'other_acc': [],    
    }
    images_paths.sort()
    xmls_paths.sort()

    cards = str()
    money = str()
    other = str()

    for xml, img in tqdm.notebook.tqdm(zip(xmls_paths, images_paths)):  
        inp = read(img)
        inp = np.expand_dims(inp, 0)
        predict_list = pipeline.recognize(inp)[0] 
        try:            
            res = count_precision(xml, predict_list)            
            quality_results['image_name'].append(os.path.basename(xml))         
            quality_results['card_acc'].append(res[0])         
            quality_results['money_acc'].append(res[1])       
            quality_results['other_acc'].append(res[2]) 

            a = full_error_statistic(xml, predict_list)            
            cards+=a[0]
            money+=a[1]
            other+=a[2]

        except Exception as e:
            print('empty xml', e)

    cards = _count_frequency(sorted(cards))
    money = _count_frequency(sorted(money))
    other = _count_frequency(sorted(other))

    error_statistic = {'cards':cards, 'money':money, 'other':other}

    res_df = pd.DataFrame(quality_results)
    return res_df, error_statistic





