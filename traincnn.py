#Các dòng này import các thư viện NumPy và pandas, thường được sử dụng cho các nhiệm vụ phân tích số liệu và dữ liệu.
import numpy as np
import pandas as pd 

import os #cc hàm tương tác với hdh
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH #hiện thị font chữ
import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
# các module và class từ keras
from keras.preprocessing.image  import ImageDataGenerator #tiền xử lý ảnh
# chuyển đỏi dl ảnh thành mảng munpy
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
# thuât toán tối ưu bộ tối ưu hóa để huấn luyện
from keras.optimizers import Adam

from glob import glob #tìm các folder con trong folder huấn luyên

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# đường dẫn đến folder test và train cho model
train_path = "D:/python/DACN2/FruitClassifier/datasets/data/train/"
test_path = 'D:/python/DACN2/FruitClassifier/datasets/data/test/'

# test thử môt ảnh trong dl, chuyển đổi nó thành mảng numpy
img = load_img(train_path + "ripe/ripe_orange_2.jpg")
shape_of_image = img_to_array(img)
print(shape_of_image.shape)


classes = glob(train_path + "/*")
number_of_class = len(classes)
print("Number of classes:", number_of_class)


train_datagen = ImageDataGenerator(
    rescale=1./255, #điều chỉnh gtr bx
    shear_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3
)
test_datagen = ImageDataGenerator(rescale = 1./255) #


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=shape_of_image.shape[:2],
    batch_size=64,
    color_mode='rgb',
    class_mode='categorical',
)
#
test_generator = test_datagen.flow_from_directory(test_path,
                                                   target_size = shape_of_image.shape[:2],
                                                   batch_size = 64,
                                                   color_mode = 'rgb',
                                                   class_mode = 'categorical')




model = Sequential()

model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(number_of_class,activation = 'softmax'))

#Dòng này biên dịch mô hình bằng cách chỉ định hàm mất mát, bộ tối ưu hóa và các chỉ số đánh giá được sử dụng trong quá trình huấn luyện.
#  Hàm mất mát là categorical cross-entropy, bộ tối ưu hóa là Adam với tốc độ học và giảm dần, và độ chính xác được sử dụng làm chỉ số đánh giá.
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, decay=1e-6),
              metrics=['accuracy'])


batch_size = 32
number_of_batch = len(train_generator)

model.fit(
    train_generator,
    steps_per_epoch=number_of_batch,
    epochs=15,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

model.save('modelCNN.h5')
