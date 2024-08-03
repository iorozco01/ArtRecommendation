from keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


# Load the pre-trained VGG16 model without the top layers
# Replace top layers as those are specific to ImageNet Classification and we want personalized
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


# Add new top layers for your specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)  # Assuming 10 classes for the paintings

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)


