
# Dinh Hoang Viet Phuong - 301123263

# Import necessary library
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, Model

# A. Import TensorFlow library

# Load the fashion_mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Creating the ds1_phuong dictionary with the first 60,000 samples
ds1_phuong = {'images': train_images, 'labels': train_labels}

# Creating the ds2_phuong dictionary with the next 10,000 samples
ds2_phuong = {'images': test_images, 'labels': test_labels}

# B. Data normalization and shape display

# 1. Normalize the images in ds1_phuong and ds2_phuong to be between -1 and 1
ds1_phuong['images'] = ds1_phuong['images'].astype('float32') / 127.5 - 1
ds2_phuong['images'] = ds2_phuong['images'].astype('float32') / 127.5 - 1

# 2. Display the shape of ds1_phuong['images'] and ds2_phuong['images']
print("Shape of ds1_phuong['images']:", ds1_phuong['images'].shape)
print("Shape of ds2_phuong['images']:", ds2_phuong['images'].shape)

# 3. Filter pants images (class label 1) from both datasets
pants_images_ds1 = ds1_phuong['images'][ds1_phuong['labels'] == 1]
pants_images_ds2 = ds2_phuong['images'][ds2_phuong['labels'] == 1]

# Concatenate the filtered images to create a new dataset
dataset_phuong = np.concatenate((pants_images_ds1, pants_images_ds2), axis=0)

# 4. Display the shape of dataset_phuong
print("Shape of dataset_phuong:", dataset_phuong.shape)

# 5. Display the first 12 images from dataset_phuong
plt.figure(figsize=(8, 8)) # Setting the figure size to 8x8
for i in range(12): # Loop through the first 12 images
    plt.subplot(4, 3, i+1) # Arrange images in a 4x3 grid
    plt.imshow(dataset_phuong[i], cmap='gray') # Display images in grayscale
    plt.xticks([]) # Remove xticks
    plt.yticks([]) # Remove yticks
plt.show()

# 6. TensorFlow dataset preparation
# Creating a TensorFlow dataset from dataset_phuong
train_dataset_phuong = tf.data.Dataset.from_tensor_slices(dataset_phuong)

# Shuffling the dataset
train_dataset_phuong = train_dataset_phuong.shuffle(7000)

# Batching the dataset with a batch size of 256
train_dataset_phuong = train_dataset_phuong.batch(256)

# C. Generator Model Definition and Visualization
# 1. Define the generator model
generator_model_phuong = tf.keras.Sequential([
    # Input layer: Vector with dimension size 100
    layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
    # Reshape the output to feed into the transposed convolution layer
    layers.Reshape((7, 7, 256)),
    
    # 2nd Layer: Batch Normalization
    layers.BatchNormalization(),
    
    # 3rd Layer: Leaky ReLU activation
    layers.LeakyReLU(),
    
    # 4th Layer: Transposed Convolution Layer
    layers.Conv2DTranspose(128, kernel_size=5, strides=(1, 1), padding='same', use_bias=False),
    
    # 5th Layer: Batch Normalization
    layers.BatchNormalization(),
    
    # 6th Layer: Leaky ReLU
    layers.LeakyReLU(),
    
    # 7th Layer: Transposed Convolution Layer
    layers.Conv2DTranspose(64, kernel_size=5, strides=(2, 2), padding='same', use_bias=False),
    
    # 8th Layer: Batch Normalization
    layers.BatchNormalization(),
    
    # 9th Layer: Leaky ReLU
    layers.LeakyReLU(),
    
    # 10th Layer: Transposed Convolution Layer with tanh activation
    layers.Conv2DTranspose(1, kernel_size=5, strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])

# 2. Display a summary of the model
generator_model_phuong.summary()

# Generate a plot of the generator model
plot_model(generator_model_phuong, to_file='generator_model_phuong.png', show_shapes=True, show_layer_names=True)

# D. Image Generation with Untrained Generator
# 1. Create a sample vector
noise = tf.random.normal([1, 100])

# 2. Generate an image from the untrained generator
generated_image = generator_model_phuong(noise, training=False)

# 3. Display the generated image
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

# E. Discriminator Model Definition and Visualization

# 1. Define the generator model
discriminator_model_phuong = tf.keras.Sequential([
    # Input layer
    layers.Input(shape=(28, 28, 1)),
    
    # 1st Layer: Convolution
    layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding='same'),
    
    # 2nd Layer: Leaky ReLU activation
    layers.LeakyReLU(),
    
    # 3rd Layer: Dropout
    layers.Dropout(0.3),
    
    # 4th Layer: Convolution
    layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding='same'),
    
    # 5th Layer: Leaky ReLU activation
    layers.LeakyReLU(),
    
    # 6th Layer: Dropout
    layers.Dropout(0.3),
    
    # Flatten the output before the output layer
    layers.Flatten(),
    
    # Output layer with a single neuron
    layers.Dense(1)
])

# 2. Display a summary of the model
discriminator_model_phuong.summary()

# Drawing a diagram of the neural network model
plot_model(discriminator_model_phuong, to_file='discriminator_model_phuong.png', show_shapes=True, show_layer_names=True)

# F. GAN Training Components
# 1. Define the loss function for the generator and discriminator
cross_entropy_phuong = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 2. Create Adam optimizers for both the generator and the discriminator
generator_optimizer_phuong = tf.keras.optimizers.Adam()
discriminator_optimizer_phuong = tf.keras.optimizers.Adam()

# Function to calculate generator loss
def generator_loss(fake_output):
    return cross_entropy_phuong(tf.ones_like(fake_output), fake_output)

# Function to calculate discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy_phuong(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_phuong(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# 3. Training step function
def training_step(images):
    # Generating noise to feed into the generator
    noise = tf.random.normal([batch_size, noise_dim])
    
    with tf.GradientTape(persistent=True) as tape:
        # Generating images from noise
        generated_images = generator_model_phuong(noise, training=True)
        
        # Discriminator's prediction on real and generated images
        real_output = discriminator_model_phuong(images, training=True)
        fake_output = discriminator_model_phuong(generated_images, training=True)
        
        # Calculating losses for both models
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Calculating gradients for both models
    gradients_of_generator = tape.gradient(gen_loss, generator_model_phuong.trainable_variables)
    gradients_of_discriminator = tape.gradient(disc_loss, discriminator_model_phuong.trainable_variables)
    
    # Applying gradients to optimizers for both models to adjust weights and improve model performance in the next iteration
    generator_optimizer_phuong.apply_gradients(zip(gradients_of_generator, generator_model_phuong.trainable_variables))
    discriminator_optimizer_phuong.apply_gradients(zip(gradients_of_discriminator, discriminator_model_phuong.trainable_variables))
    
    return gen_loss, disc_loss

# G. Training Setup and Execution
# Constants for training
batch_size = 256
noise_dim = 100

# Define loss functions
def generator_loss(fake_output):
    return cross_entropy_phuong(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy_phuong(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_phuong(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Training loop
num_epochs = 10
epoch_times = []

for epoch in range(num_epochs):
    start = time.time()
    
    for image_batch in train_dataset_phuong:
        gen_loss, disc_loss = training_step(image_batch)
    
    # Time taken for the epoch
    epoch_duration = time.time() - start
    epoch_times.append(epoch_duration)
    print(f'Time for epoch {epoch + 1} is {epoch_duration} sec')

# Calculate the average time per epoch from the given data
average_time_per_epoch = sum(epoch_times) / len(epoch_times)

# The total number of batches for 70,000 samples given a batch size of 256
total_batches = 70000 / 256

# The total time for one epoch using 70,000 samples
time_for_one_epoch = total_batches * average_time_per_epoch

# The total time for 100 epochs
estimated_total_time = time_for_one_epoch * 100
print(f"Estimated total time for training on 70,000 samples for 100 epochs: {estimated_total_time} seconds")

# H. Generating and Displaying Images

# 1. Create 16 sample vectors
random_vectors = tf.random.normal([16, noise_dim])

# 2. Generate images from the generator model
generated_images = generator_model_phuong(random_vectors, training=False)

# 3. Normalize the pixels in the generated images to the [0, 255] range for display
normalized_images = (generated_images * 127.5) + 127.5
normalized_images = tf.cast(normalized_images, tf.uint8)

# 4. Display the images
plt.figure(figsize=(8, 8))
for i in range(normalized_images.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(normalized_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()