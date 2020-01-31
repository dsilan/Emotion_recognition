import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
raw_data_csv_file_name = '../emotion_dataset/fer2013/fer2013.csv'
raw_data = pd.read_csv(raw_data_csv_file_name)


img = raw_data["pixels"][0]
val = img.split(" ")
x_pixels = np.array(val, 'float32')
x_pixels /= 255

#display
x_reshaped = x_pixels.reshape(48,48)

plt.imshow(x_reshaped, cmap= "gray", interpolation="nearest")
plt.axis("off")
plt.show()