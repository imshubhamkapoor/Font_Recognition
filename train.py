from utils import crop_dataset, list_dataset, load_dataset, model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the path for training and validation dataset
train_path = r"C:\Users\shiva\Deepfont\train"
valid_path = r"C:\Users\shiva\Deepfont\valid"

### Run only when required
## crop_dataset(train_path)
## crop_dataset(valid_path)

# Loading and pre-processing the training and validation dataset
train_data, image_count_train, CLASS_NAMES = list_dataset(train_path)
valid_data, image_count_valid, CLASS_NAMES = list_dataset(valid_path)
print('Number of training images: {} \nNumber of validation images: {}'.format(image_count_train, image_count_valid))

# Showing sample images from the training dataset
Aller_Bd = list(train_data.glob('Aller_Bd/*'))
for image_path in Aller_Bd[:3]: Image.open(str(image_path)).show()

# Defining the input parameters for training the model
TRAIN_BATCH_SIZE = 300
VALID_BATCH_SIZE = 100
IMG_HEIGHT = 100
IMG_WIDTH = 100
OUTPUT_CLASSES = 100
STEPS_PER_EPOCH = np.ceil(image_count_train / TRAIN_BATCH_SIZE)

# Loading the training and validation dataset for training the model
train_data_gen = load_dataset(train_data, TRAIN_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES)
valid_data_gen = load_dataset(valid_data, VALID_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES)

# Plotting the sample images from the training dataset
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(3,3))
    for n in range(9):
        ax = plt.subplot(3,3,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

model.compile(optimizer='Adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=6,
    validation_data=valid_data_gen,
    validation_steps=image_count_valid // VALID_BATCH_SIZE
)

# Saving the trained model for future use
model.save('CNN_Font_Classification.h5', overwrite=True)