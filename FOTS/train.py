import tensorflow as tf
import matplotlib.pyplot as plt
import model
from model.modules import *

input_shape = (640, 640, 3)
image_pathes = ["../text1.jpg", "../dog2.jpg"]
images = []
for img in image_pathes:
    img_raw = tf.io.read_file(img)
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [640, 640])
    img_final = img_final / 255.0
    images.append(img_final)
images = tf.convert_to_tensor(images)
# imgs = np.random.randn(2, 640, 640, 3).astype(np.float32)

resnet50_wrapper = Resnet50Wrapper(input_shape=input_shape)
shared_features = SharedConvlution(backbone=resnet50_wrapper)
detector = TextDetector(shared_features=shared_features, input_shape=input_shape)
score_map, geo_map, angle_map = detector(images)

fig, axs = plt.subplots(4, 2)
axs = axs.flatten()

axs[0].imshow(images[0])

# plot scroe-map
s_map = tf.squeeze(score_map[0])
axs[1].imshow(s_map)
axs[1].set_title("s_map")


# plot geo map
g_map = tf.squeeze(geo_map[0, :, :, 0])
axs[2].imshow(g_map)
axs[1].set_title("g_map_0")

g_map = tf.squeeze(geo_map[0, :, :, 1])
axs[3].imshow(g_map)

g_map = tf.squeeze(geo_map[0, :, :, 2])
axs[4].imshow(g_map)

g_map = tf.squeeze(geo_map[0, :, :, 3])
axs[5].imshow(g_map)

a_map = tf.squeeze(angle_map[0])
axs[6].imshow(a_map)

fig.tight_layout()
plt.show()