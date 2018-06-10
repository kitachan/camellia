# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt 
import os


os.environ["PATH"] += os.pathsep + 'D:\Program Files (x86)\Anaconda\Graphviz2.38\bin'

#Stage 1 - Preparing CNN

# Initialize CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64,64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
plot_model(classifier, to_file='class_plot.png', show_shapes = True, show_layer_names=True)

# Compilation of CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

#Comment out when loading model

classifier.save('camweight.h5')

#Comment out when running CNN for the first time

#classifier.load_weights('camweight.h5')

#Stage 2 - adding images to CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_dataset = train_datagen.flow_from_directory(
                                                        'data/train',
                                                        target_size = (64,64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

test_dataset = test_datagen.flow_from_directory(
                                                        'data/test',
                                                        target_size = (64,64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')


#Comment out when attempting prediction

history = classifier.fit_generator(training_dataset,
                        steps_per_epoch = 800,
                        epochs = 25,
                      validation_data = test_dataset,
                        validation_steps = 400)

print(history.history.keys())
plt.figure(1)

 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
 # summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  


#Stage 3 - Make new prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('data/prediction/camellia_or_other.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_dataset.class_indices
if result[0][0] == 1:
    prediction = 'other'
else:
    prediction = 'camellia'

print("This is a: ", prediction)
