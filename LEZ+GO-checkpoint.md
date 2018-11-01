
### import tensorflow and packages, then import keras from tensorflow


```python

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
import sys
from PIL import Image
import os
import shutil
#check all versions
print('tensorflow version is', tf.__version__, ) #should be >= 1.11.0 
print('keras version is', keras.__version__) #shoudl be >= 2.1.6-tf
print('numpy version is', np.__version__) #should be >= 1.15.3
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-65-f5d125d2434c> in <module>()
          1 
    ----> 2 from tensorflow.python.keras.applications.vgg16 import VGG16
          3 from tensorflow.python.keras import models
          4 from tensorflow.python.keras import layers
          5 from tensorflow.python.keras.preprocessing import image
    

    ModuleNotFoundError: No module named 'tensorflow.python.keras.applications.vgg16'


###### since we have a small dataset, we can use a pre-trained network that has already been trained on a large-scale image classification task. 


```python
#let's use a convnet trained on the ImageNet dataset (1.4e6 labeled images, and 1000 clases); we will be particularly basing out model from a binary classification between dogs and cats
conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150,150,3))
#weights - to specify which weight checkpoint to initialize the model from
#include_top - we set to False because this is a DNN top layer default for 1000 classes, but we only need it for our own DNN with only 2 classes
#input_shape - shape of image tensors that we will feed into the network. Argument is not important.
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 9s 0us/step
    

#### didactic aside: this is the convolutional base of the standard VGG16


```python
conv_base.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 150, 150, 3)       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    

#### we can see that  this base has ~15e6 parameters aka yuge. 
not only that but it's an immense amount of layers
I can't get pictures to work on here atm, but imagine all these layers being a long vertical column (trained convolutional base) which feeds into the trained classifier to produce predictions. 
convnets generally work in two parts: input layer which feeds into a conv layer which is pooled until the next conv layer which is pooled. 
As you can see the dimensions ever decrease with each maxpool, which would make more sense in image form (I guess I'll add that to the read me)

#### either way, we exploit this by running new data (our data) through this conv base, and simply replacing the trained classifier
basically, the idea is to get as far away from this DNN as possible. 
it's reusable and generalizable because of the process of taking in data in the earlier layers which extract local and highly generic feature maps (visual edges, colors, textures) while layers higher up extract more abstract concepts (such as cat ear or dog eye, or in our case colormaps). 

#### let's do two things: 1. running the convnet as is over our dataset, recording its output to a numpy array (on disk), use this data as input to a standalone densely-connected classifier. there's no leverage here though, so I may skip this for now. ; 2. extending the model we have (conv base) by adding a dense layer to the top.

## Build Keras Model
### let's just leverage this 
you will also see a model summary, which you can compare to the earlier one.


```python
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False
```


```python
#let's check it out
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Model)                (None, 4, 4, 512)         14714688  
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 8192)              0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 256)               2097408   
    _________________________________________________________________
    dense_11 (Dense)             (None, 1)                 257       
    =================================================================
    Total params: 16,812,353
    Trainable params: 2,097,665
    Non-trainable params: 14,714,688
    _________________________________________________________________
    

### check it out we have 2mil trainable params - great!
### let's now apply this Keras model to our TF estimator


```python
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
```


```python
#model_dir will be our locationt o store trained tensorflow models. We will view this progress using tensorboard (hopefully). 
model_dir = os.path.join(os.getcwd(), "models//catvsdog").replace("//", "\\")
os.makedirs(model_dir, exist_ok=True)
print("model_dir: ",model_dir)
est_catvsdog = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                    model_dir=model_dir)
```

    model_dir:  C:\Users\Scott\tensorflowstuffadnan\Nuke the rest - new beginning\models\catvsdog
    INFO:tensorflow:Using the Keras model provided.
    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': 'C:\\Users\\Scott\\tensorflowstuffadnan\\Nuke the rest - new beginning\\models\\catvsdog', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000002A897EC2630>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
    


```python
# input layer name
input_name = model.input_names[0]
input_name
#issues here, msot likely because using same function names for both examples
```




    'vgg16_input'




```python
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
```

# Download the Dataset; ##let's just do a trial with cats and dogs
this will consist of 3 subsets: 1. training set of 1.6x samples of each class (dog and cat aka Task1 and Task2); 2. test set of x samples of each class. 


```python
#this is where I ran into problems before because this jupyter notebook has
#become awfully cluttered and this particular cell creates directories

#original directory of dataset
original_dataset_dir = 'C:\\Users\\Scott\\AppData\\Local\\Temp\\HamsterArc4\\train\\train'

# The directory where we will
# store our smaller dataset
base_dir = './data3/dog_vs_cat_small' #change to data(n+last) whenever prompted file already exists; must faster than deleting and reloading
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
```


```python
import shutil

os.makedirs(base_dir, exist_ok=True)
# Directories for our training,
# validation and test splits
os.mkdir(train_dir)
os.mkdir(test_dir)
# Directory with our training cat pictures
os.mkdir(train_cats_dir)
# Directory with our training dog pictures
os.mkdir(train_dogs_dir)
# Directory with our validation cat pictures
os.mkdir(test_cats_dir)
# Directory with our validation dog pictures
os.mkdir(test_dogs_dir)
# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)    
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
```


```python
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))
```

    total training cat images: 1000
    total training dog images: 1000
    total test cat images: 500
    total test dog images: 500
    

# YES FINALLY
now to create a function to shuffle the images with associated labels


```python
def unison_shuffled_copies(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
```


```python
CAT_LABEL = 0
DOG_LABEL = 1
train_cats = [os.path.join(train_cats_dir, file_name) for file_name in os.listdir(train_cats_dir)]
train_dogs = [os.path.join(train_dogs_dir, file_name) for file_name in os.listdir(train_dogs_dir)]
train_files = train_cats + train_dogs
train_labels = [CAT_LABEL]*len(train_cats)+[DOG_LABEL]*len(train_dogs)
train_files, train_labels = unison_shuffled_copies(train_files, train_labels)
test_cats = [os.path.join(test_cats_dir, file_name) for file_name in os.listdir(test_cats_dir)]
test_dogs = [os.path.join(test_dogs_dir, file_name) for file_name in os.listdir(test_dogs_dir)]
test_files = test_cats + test_dogs
test_labels = [CAT_LABEL]*len(test_cats)+[DOG_LABEL]*len(test_dogs)
test_files, test_labels = unison_shuffled_copies(test_files, test_labels)
```


```python
#first 10 shuffled images and labels
print(train_files[:10])
print(train_labels[:10])
```

    ['./data3/dog_vs_cat_small\\train\\dogs\\dog.819.jpg'
     './data3/dog_vs_cat_small\\train\\cats\\cat.764.jpg'
     './data3/dog_vs_cat_small\\train\\dogs\\dog.619.jpg'
     './data3/dog_vs_cat_small\\train\\cats\\cat.529.jpg'
     './data3/dog_vs_cat_small\\train\\dogs\\dog.511.jpg'
     './data3/dog_vs_cat_small\\train\\cats\\cat.538.jpg'
     './data3/dog_vs_cat_small\\train\\cats\\cat.202.jpg'
     './data3/dog_vs_cat_small\\train\\cats\\cat.201.jpg'
     './data3/dog_vs_cat_small\\train\\cats\\cat.823.jpg'
     './data3/dog_vs_cat_small\\train\\dogs\\dog.774.jpg']
    [1 0 1 0 1 0 0 0 0 1]
    

# BEAUTIFUL.
## now to make that TFRecord so we can use this on our own data
you will start to see a lot of tf prefixes now


```python
path_tfrecords_train = os.path.join(base_dir, "train.tfrecords")
path_tfrecords_test = os.path.join(base_dir, "test.tfrecords")
```


```python
#functioni for printing this conversion process 
def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.writer(msg)
    sys.stdout.flush()
```


```python
#function for wrapping an integer so it can be saved to the TFRecord file
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#i always have issues with integers so i'm playing it safe using int64
#recall 8 bits in a byte; so 64 bits is 8 bytes! speakign of which
#we do the same by wrapping raw bytes to the TFRecord file
```


```python
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
```


```python
#function for reading images from disk and writing them along with class-labels
#to a TFRecord file
def convert(image_paths, labels, out_path, size=(150,150)):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.    
    print("Converting: " + out_path)
    # Number of images. Used when printing the progress.
    num_images = len(image_paths)    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:        
        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images-1)
            # Load the image-file using matplotlib's imread function.
            img = Image.open(path)
            img = img.resize(size)
            img = np.array(img)
            # Convert the image to raw bytes.
            img_bytes = img.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }
            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)
            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)
            # Serialize the data.
            serialized = example.SerializeToString()        
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
```


```python
convert(image_paths=train_files,
        labels=train_labels,
        out_path=path_tfrecords_train)
convert(image_paths=test_files,
        labels=test_labels,
        out_path=path_tfrecords_test,
        size=img_size[:2])
```

    Converting: ./data3/dog_vs_cat_small\train.tfrecords
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-82-60ac895644d3> in <module>()
          1 convert(image_paths=train_files,
          2         labels=train_labels,
    ----> 3         out_path=path_tfrecords_train)
          4 convert(image_paths=test_files,
          5         labels=test_labels,
    

    <ipython-input-81-4812c75d9227> in convert(image_paths, labels, out_path, size)
         13     num_images = len(image_paths)
         14     # Open a TFRecordWriter for the output-file.
    ---> 15     with tf.python_io.TFRecordWriter(out_path) as writer:
         16         # Iterate over all the image-paths and class-labels.
         17         for i, (path, label) in enumerate(zip(image_paths, labels)):
    

    AttributeError: module 'tensorflow' has no attribute 'python_io'



```python
tensorflow
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-68-f830eff1fc11> in <module>()
          1 import tensorflow as tf
    ----> 2 tf.__version__()
    

    AttributeError: module 'tensorflow' has no attribute '__version__'



```python

```
