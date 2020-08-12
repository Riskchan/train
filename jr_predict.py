import sys

def usage():
    print("Usage: {0} <input_filename>".format(sys.argv[0]), file=sys.stderr)
    exit(1)

# === 入力画像のファイル名を引数から取得 ===
if len(sys.argv) != 2:
    usage()
input_filename = sys.argv[1]

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 特徴量の設定
classes = ["E231-yamanote", "E233-chuo", "E235-yamanote"]
num_classes = len(classes)
img_width, img_height = 128, 128
feature_dim = (img_width, img_height, 3)

# === モデル読込み ===
model = tf.keras.models.load_model("weights.30-0.0653-0.0199.hdf5")

# === 入力画像の読み込み ===
img = image.load_img(input_filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# 学習時と同様に値域を[0, 1]に変換する
x = x / 255.0
# 車両形式を予測
pred = model.predict(x)[0]
# 結果を表示する
for cls, prob in zip(classes, pred):
    print("{0:18}{1:8.4f}%".format(cls, prob * 100.0))