import logging
from itertools import compress
import pathlib
import cv2 as cv
import tensorflow as tf
from .datautils import *

logger = logging.getLogger(__name__)

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

class ICDAR():
    """ ICDAR-2015 """

    structure = {
        '2015': {
            'training': {
                'images': 'ch4_training_images',
                'gt': 'ch4_training_localization_transcription_gt',
                'voc_per_image': 'ch4_training_vocabularies_per_image',
                'voc_all': 'ch4_training_vocabulary.txt'
            },
            'test': {
                'images': 'ch4_test_images',
                'gt': 'Challenge4_Test_Task4_GT',
                'voc_per_image': 'ch4_test_vocabularies_per_image',
                'voc_all': 'ch4_test_vocabulary.txt'
            },
            'voc_generic': 'GenericVocabulary.txt'
        },
        '2013': {
            'training': {
                'images': 'ch2_training_images',
                'gt': 'ch2_training_localization_transcription_gt',
                'voc_per_image': 'ch2_training_vocabularies_per_image',
                'voc_all': 'ch2_training_vocabulary.txt'
            },
            'test': {
                'images': 'Challenge2_Test_Task12_Images',
                'voc_per_image': 'ch2_test_vocabularies_per_image',
                'voc_all': 'ch4_test_vocabulary.txt'
            },
            'voc_generic': 'GenericVocabulary.txt'
        },
    }

    def __init__(self, data_root, year="2015", type="training"):
        data_root = pathlib.Path(data_root)

        if year == '2013' and type == 'test':
            logger.warning('ICDAR 2013 does not contain test ground truth. Fall back to training instead.')

        self.structure = ICDAR.structure[year]
        self.images_root = data_root / self.structure[type]['images']
        self.gt_root = data_root / self.structure[type]['gt']

        images = [str(img) for img in self.images_root.glob("*.jpg")]
        # path_ds = tf.data.Dataset.from_tensor_slices(images)
        # self.images = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.images = images

        gt_paths = [str(gt_path) for gt_path in self.gt_root.glob("*.txt")]
        #path_ds_gt = tf.data.Dataset.from_tensor_slices(gt_paths)
        #self.bboxes = path_ds_gt.map(self.load_gt, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.bboxes = []
        self.transcripts = []

        for gt_path in gt_paths:
            gt_path = str(gt_path)
            bbxs, txts = self.load_gt(gt_path)
            self.bboxes.append(bbxs)
            self.transcripts.append(txts)

    def __getitem__(self, item):
        image = self.images[item]
        bboxes = self.bboxes[item]
        transcripts = self.transcripts[item]
        self.__transform((image, bboxes, transcripts))

    def __transform(self, gt, input_size = 512, random_scale = np.array([0.5, 1, 2.0, 3.0]), background_ratio = 3. / 8):
        image, word_bboxes, transcripts = gt
        im = cv.imread(image)
        num_words = len(word_bboxes)
        text_polys = word_bboxes  # num_words * 4 * 2
        text_tags = [True if (tag == '*' or tag == '###') else False for tag in transcripts]  # ignore '###'

        h, w, _ = im.shape
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

        rd_scale = np.random.choice(random_scale)
        im = cv2.resize(im, dsize = None, fx = rd_scale, fy = rd_scale)
        text_polys *= rd_scale

        rectangles = []

        # print rd_scale
        # random crop a area from image
        if np.random.rand() < background_ratio:
            # crop background
            im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=True)
            if text_polys.shape[0] > 0:
                # cannot find background
                raise RuntimeError('cannot find background')
            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize = (input_size, input_size))
            score_map = np.zeros((input_size, input_size), dtype = np.uint8)
            geo_map_channels = 5
            #geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
            geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype = np.float32)
            training_mask = np.ones((input_size, input_size), dtype = np.uint8)
        else:
            im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
            if text_polys.shape[0] == 0:
                raise RuntimeError('cannot find background')
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize = (resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
            new_h, new_w, _ = im.shape
            score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

        if True:
            import matplotlib.pyplot as plt
            import matplotlib.patches as Patches
            fig, axs = plt.subplots(3, 2, figsize=(30, 30))
            axs[0, 0].set_title("shape: {}".format(im.shape))
            axs[0, 0].imshow(im[:, :, ::-1])
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            for poly in text_polys:
                poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                axs[0, 0].add_artist(Patches.Polygon(
                    poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
            axs[0, 1].imshow(score_map[::, ::])
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 1].set_title("shape: {}".format(score_map.shape))
            axs[1, 0].imshow(geo_map[::, ::, 0])
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 0].set_title("shape: {}".format(geo_map[::, ::, 0].shape))
            axs[1, 1].imshow(geo_map[::, ::, 1])
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            axs[1, 1].set_title("shape: {}".format(geo_map[::, ::, 1].shape))
            axs[2, 0].imshow(geo_map[::, ::, 2])
            axs[2, 0].set_xticks([])
            axs[2, 0].set_yticks([])
            axs[2, 0].set_title("shape: {}".format(geo_map[::, ::, 2].shape))
            axs[2, 1].imshow(training_mask[::, ::])
            axs[2, 1].set_xticks([])
            axs[2, 1].set_yticks([])
            axs[2, 0].set_title("shape: {}".format(training_mask[::, ::].shape))
            plt.tight_layout()
            plt.show()
            plt.close()

        # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
        images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
        score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
        geo_maps = geo_map[::4, ::4, :].astype(np.float32)
        training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

        transcripts = [transcripts[i] for i in selected_poly]
        mask = [not (word == '*' or word == '###') for word in transcripts]
        transcripts = list(compress(transcripts, mask))
        rectangles = list(compress(rectangles, mask)) # [ [pt1, pt2, pt3, pt3],  ]

        return images, score_maps, geo_maps, training_masks, transcripts, rectangles

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        image = tf.image.convert_image_dtype(tf.image.decode_jpeg(image, channels=3), dtype=tf.float32)
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_gt(self, path):
        bboxes = []
        texts = []
        with open(path) as f:
            for line in f:
                text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                transcript = text[8]
                bboxes.append(bbox)
                texts.append(transcript)
        return np.array(bboxes, dtype=np.float32), texts