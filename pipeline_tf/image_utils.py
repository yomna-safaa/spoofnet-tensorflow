import tensorflow as tf
from data import helpers, helpers_image
from preprocessing.y_vgg_preprocessing import resize

######################################################################
class ImageCoder_TF(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()
        ## YY: TODO : i used image.rgb_to_grayscale .. but could EASIER use decode_png/decode_jpeg with channels=1 :D

        self._jpg_string = tf.placeholder(dtype=tf.string)
        self._png_string = tf.placeholder(dtype=tf.string)
        self._image_string = tf.placeholder(dtype=tf.string)
        self._uint8_image = tf.placeholder(dtype=tf.uint8)

        ## convert strings jpg <-> png ... decode then encode
        self._png_to_jpg = self._png_string_to_jpg_string(self._png_string)
        self._jpg_to_png = self._jpg_string_to_png_string(self._jpg_string)

        ## decoders:  string to uint8
        self._decoded_jpg_image = tf.image.decode_jpeg(self._jpg_string)
        self._decoded_png_image = tf.image.decode_png(self._png_string)

        self._decoded_jpg_image_rgb = tf.image.decode_jpeg(self._jpg_string, channels=3)
        self._decoded_png_image_rgb = tf.image.decode_png(self._png_string, channels=3)

        self._decoded_jpg_image_grey = tf.image.decode_jpeg(self._jpg_string, channels=1)
        self._decoded_png_image_grey = tf.image.decode_png(self._png_string, channels=1)

        self._encoded_png_str = self._uint8_image_to_png_string(self._uint8_image)
        self._encoded_jpg_str = self._uint8_image_to_jpg_string(self._uint8_image)
        self._encoded_png_str_gray_fromRGB = self._uint8_rgb_image_to_grey_png_string(self._uint8_image)
        self._encoded_jpg_str_gray_fromRGB = self._uint8_rgb_image_to_grey_jpg_string(self._uint8_image)

    def _uint8_image_to_png_string(self, uint8_image):
        return tf.image.encode_png(uint8_image)

    def _uint8_image_to_jpg_string(self, uint8_image): #, format='rgb'):
        return tf.image.encode_jpeg(uint8_image) #, format=format) #, quality=100)

    def _png_string_to_jpg_string(self, png_string):
        # Initializes function that converts PNG to JPEG data.
        image = tf.image.decode_png(png_string) #, channels=3)
        png_to_jpeg = self._uint8_image_to_jpg_string(image)
        return png_to_jpeg

    def _jpg_string_to_png_string(self, jpg_string):
        # Initializes function that converts JPEG to PNG data (str).
        image = tf.image.decode_jpeg(jpg_string) #, channels=3)
        jpg_to_png = self._uint8_image_to_png_string(image)
        return jpg_to_png

    def _uint8_rgb_image_to_grey_png_string(self, rgb_image):
        ## The conversion functions work only on float images, so you need to convert images in other formats using convert_image_dtype
        ## convert encoded string to grayscale:
        # rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
        tmp_grayscale_image = tf.image.rgb_to_grayscale(rgb_image) #_float)
        # tmp_grayscale_image = tf.image.convert_image_dtype(tmp_grayscale_image, tf.uint8)
        encoded_png_str = self._uint8_image_to_png_string(tmp_grayscale_image)
        return encoded_png_str

    def _uint8_rgb_image_to_grey_jpg_string(self, uint8_image):
        # tmp_grayscale_image = tf.image.rgb_to_grayscale(uint8_image)
        ## The conversion functions work only on float images, so you need to convert images in other formats using convert_image_dtype
        ## convert encoded string to grayscale:
        rgb_image_float = tf.image.convert_image_dtype(uint8_image, tf.float32)
        tmp_grayscale_image_float = tf.image.rgb_to_grayscale(rgb_image_float)
        tmp_grayscale_image = tf.image.convert_image_dtype(tmp_grayscale_image_float, tf.uint8)
        encoded_jpg_str = self._uint8_image_to_jpg_string(tmp_grayscale_image) #, format='grayscale')
        return encoded_jpg_str

    def png_to_jpeg(self, image_string):
        return self._sess.run(self._png_to_jpg,
                              feed_dict={self._png_string: image_string})

    def jpeg_to_png(self, image_string):
        return self._sess.run(self._jpg_to_png,
                              feed_dict={self._jpg_string: image_string})

    def convert_jpg_to_graycale(self, image_str_jpg):
        img = self.decode_jpeg(image_str_jpg)
        if img.shape[2] == 1:
            return self._sess.run(self._encoded_jpg_str,
                                  feed_dict={self._uint8_image: img})
        elif img.shape[2] == 3:
            return self._sess.run(self._encoded_jpg_str_gray_fromRGB,
                                  feed_dict={self._uint8_image: img})

    def convert_png_to_graycale(self, image_str_png):
        img = self.decode_png(image_str_png)
        if img.shape[2] == 1:
            return self._sess.run(self._encoded_png_str,
                                  feed_dict={self._uint8_image: img})
        elif img.shape[2] == 3:
            return self._sess.run(self._encoded_png_str_gray_fromRGB,
                                  feed_dict={self._uint8_image: img})

    def decode_png(self, image_string, grayscale=None):
        if grayscale is None:
            image = self._sess.run(self._decoded_png_image,
                                   feed_dict={self._png_string: image_string})
        else:
            if grayscale:
                image = self._sess.run(self._decoded_png_image_grey,
                                       feed_dict={self._png_string: image_string})
                assert len(image.shape) == 3
                assert image.shape[2] == 1
            else:
                image = self._sess.run(self._decoded_png_image_rgb,
                                       feed_dict={self._png_string: image_string})
                assert len(image.shape) == 3
                assert image.shape[2] == 3

        return image

    def decode_jpeg(self, image_string, grayscale=None):
        if grayscale is None:
            image = self._sess.run(self._decoded_jpg_image,
                                   feed_dict={self._jpg_string: image_string})
        else:
            if grayscale:
                image = self._sess.run(self._decoded_jpg_image_grey,
                                       feed_dict={self._jpg_string: image_string})
                assert len(image.shape) == 3
                assert image.shape[2] == 1
            else:
                image = self._sess.run(self._decoded_jpg_image_rgb,
                                       feed_dict={self._jpg_string: image_string})
                assert len(image.shape) == 3
                assert image.shape[2] == 3

        return image

    ## YY:
    def encode_to_png(self, uint8_image):
        return self._sess.run(self._encoded_png_str,
                              feed_dict={self._uint8_image: uint8_image})

    def encode_to_jpg(self, uint8_image):
        return self._sess.run(self._encoded_jpg_str,
                              feed_dict={self._uint8_image: uint8_image})


############################################################
def modify_image_string(image_string, coder, out_grayscale, format):
    if format=='JPEG':
        image = coder.decode_jpeg(image_string, grayscale=out_grayscale)
        image_string = coder.encode_to_jpg(image)
        image = coder.decode_jpeg(image_string)
    elif format == 'PNG':
        image = coder.decode_png(image_string, grayscale=out_grayscale)
        image_string = coder.encode_to_png(image)
        image = coder.decode_png(image_string)

    return image_string, image


######################################################################
def process_image_string(image_string, filename, coder, out_grayscale=False, encode_type='JPEG'):
    # Convert any PNG to JPEG's for consistency.
    if encode_type == 'JPEG':
        if not helpers.is_jpg(filename):
            if helpers.is_png(filename):
                print('Converting PNG to JPEG for %s' % filename)
                image_string = coder.png_to_jpeg(image_string)
            else:
                print('Converting img to JPEG for %s' % filename)
                mode = 'RGB' if not out_grayscale else 'L'
                image = helpers_image.load_and_resize_image(filename, height=0, width=0, mode=mode)
                if out_grayscale:
                    image = image[:, :, np.newaxis]
                image_string = coder.encode_to_jpg(image)
    elif encode_type == 'PNG':
        if not helpers.is_png(filename):
            if helpers.is_jpg(filename):
                print('Converting JPEG to PNG for %s' % filename)
                image_string = coder.jpeg_to_png(image_string)
            else:
                print('Converting img to PNG for %s' % filename)
                mode = 'RGB' if not out_grayscale else 'L'
                image = helpers_image.load_and_resize_image(filename, height=0, width=0, mode=mode)
                if out_grayscale:
                    image = image[:, :, np.newaxis]
                image_string = coder.encode_to_png(image)

    ## YYY: note: the coming lines are just for checking (decoding to encude encoding was done right)
    if encode_type == 'JPEG':
        # Decode the RGB JPEG.
        image = coder.decode_jpeg(image_string)  # , grayscale=grayscale)
    elif encode_type == 'PNG':
        # Decode the RGB JPEG.
        image = coder.decode_png(image_string)  # , grayscale=grayscale)

    ####
    if out_grayscale:
        if image.shape[2] == 3:  # img is color . convert it to gray
            image_string, image = modify_image_string(image_string, coder, out_grayscale, encode_type)
    else:  # output must be color:
        if image.shape[2] == 1:  # img is gray . convert it to rgb
            image_string, image = modify_image_string(image_string, coder, out_grayscale, encode_type)

    return image_string, image


###################################
def process_image_file(filename, coder, out_grayscale=False, encode_type='JPEG'):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder_TF to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_string = f.read()

    image_string, image = process_image_string(image_string, filename, coder, out_grayscale, encode_type)

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    if out_grayscale:
        assert image.shape[2] == 1
    else:
        assert image.shape[2] == 3

    return image_string, height, width, image.shape[2], image


######################################################################
def preprocess_image(preprocessing_name, image_preprocessing_fn, raw_image, height, width,
                             per_image_standardization=False,
                             vgg_sub_mean_pixel=None, vgg_resize_side_in=None,
                             vgg_use_aspect_preserving_resize=None, do_crop=True):
    if not do_crop:
        if (preprocessing_name == 'y_vgg') or (preprocessing_name == 'y_combined'):
            return image_preprocessing_fn(raw_image,
                                           sub_mean_pixel=vgg_sub_mean_pixel,
                                           use_per_img_std=per_image_standardization)
        else:
            return image_preprocessing_fn(raw_image)

    if preprocessing_name == 'vgg':
      vgg_resize_side = height
      image = image_preprocessing_fn(raw_image, height, width,
                                     resize_side_min=vgg_resize_side,
                                     resize_side_max=vgg_resize_side)
    elif (preprocessing_name == 'y_vgg') or (preprocessing_name == 'y_combined'):
      image = image_preprocessing_fn(raw_image, height, width,
                                     resize_side_min=vgg_resize_side_in,
                                     resize_side_max=vgg_resize_side_in,
                                     sub_mean_pixel=vgg_sub_mean_pixel,
                                     use_per_img_std=per_image_standardization,
                                     use_aspect_preserving_resize=vgg_use_aspect_preserving_resize)
    else:
      image = image_preprocessing_fn(raw_image, height, width)

    return image



########################################################################
def get_preprocessed_img_crops(raw_image, preprocessing_name, final_height, final_width,
                               image_preprocessing_fn_at_eval, image_crop_processing_fn,
                               oversample=False,
                               per_image_standardization=False,
                               vgg_sub_mean_pixel=None, vgg_resize_side_in=None,
                               vgg_use_aspect_preserving_resize=None):

    if oversample:
        if vgg_resize_side_in is None:
            raise ValueError('no resize size given during multicrop, only final size')
        elif (final_height == final_width) and (final_height == vgg_resize_side_in):
            raise ValueError('resize size given during multicrop is equal to final crop size')

    crops = []
    if oversample is False:
        print('Running a single image')
        image = preprocess_image(preprocessing_name, image_preprocessing_fn_at_eval, raw_image, final_height,
                                 final_width, per_image_standardization=per_image_standardization,
                                 vgg_sub_mean_pixel=vgg_sub_mean_pixel, vgg_resize_side_in=vgg_resize_side_in,
                                 vgg_use_aspect_preserving_resize=vgg_use_aspect_preserving_resize)

        crops.append(image)
    else:
        print('Running multi-cropped image')

        ################################
        # resized = tf.image.resize_images(raw_image, (vgg_resize_side_in, vgg_resize_side_in))
        resized = resize(raw_image, vgg_resize_side_in, use_aspect_pres_resize=vgg_use_aspect_preserving_resize)
        tf.summary.image('y_resized', tf.expand_dims(resized, 0))
        h = vgg_resize_side_in  # image.shape[0]
        w = vgg_resize_side_in  # image.shape[1]
        hl = h - final_height
        wl = w - final_width
        corners = [(0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl / 2), int(wl / 2))]
        for corner in corners:
            ch, cw = corner
            cropped = tf.image.crop_to_bounding_box(resized, ch, cw, final_height, final_width)
            tf.summary.image('y_cropped', tf.expand_dims(cropped, 0))
            cropped = preprocess_image(preprocessing_name, image_crop_processing_fn, cropped, height=0, width=0,
                                 per_image_standardization=per_image_standardization,
                                 vgg_sub_mean_pixel=vgg_sub_mean_pixel, do_crop=False)
            crops.append(cropped)
            tf.summary.image('y_cropped_processed', tf.expand_dims(cropped, 0))

            flipped = tf.image.flip_left_right(cropped)
            tf.summary.image('y_flipped_processed', tf.expand_dims(flipped, 0))
            crops.append(flipped)

    image_batch = tf.stack(crops)
    return crops, image_batch


########################################################################
def make_batch_from_img_pl(sess, preprocessed_img_crops, raw_images_pl, filename, coder,
                           encode_type='JPEG', summary_op=None, summary_writer=None):

    _, _, _, _, raw_image = process_image_file(filename, coder, encode_type=encode_type)

    if summary_op is not None:
        [crops, image_batch], summaries = sess.run([preprocessed_img_crops, summary_op], feed_dict={raw_images_pl: raw_image})
        summary_writer.add_summary(summaries)
    else:
        [crops, image_batch] = sess.run(preprocessed_img_crops, feed_dict={raw_images_pl: raw_image})
    return crops, image_batch
    # image_batch = tf.stack(crops)
    # return image_batch
