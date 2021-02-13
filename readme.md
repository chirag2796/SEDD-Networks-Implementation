# Shallow Encoder Deep Decoder (SEDD) Networks for Image Encryption and Decryption

This is the experimental implementation and validation done for the independent research project 'SEDD Networks for Neural Image Cryptography', leading to a research paper.


[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)
[![Generic badge](https://img.shields.io/badge/tensorflow-1.14-orange.svg)](https://shields.io/)


## About the Research

*   A Recurent Neural Network with LSTM nodes implementation for text generation, trained on Donald Trump tweets dataset using contextual labels, and can generate realistic ones from random noise.
*   A novel framework for image cryptography using neural networks, with a shallow encoder and a deep decoder network. Involved both basic and applied research.
*   Encoder is computationally simple,  designed for real-time encoding on edge devices.
*   Decoder is a sophisticated network which is run on a secure, computationally powerful machine.
*   Peer review from JVIS (Elsevier) pointed that the research and approach is both interesting and novel, but more sophisticated methodology must be tested. I Intend to pursue this further, and send an updated manuscript for review.

## Resources

*   Research Paper. "[https://arxiv.org/abs/2001.03017](https://arxiv.org/abs/2001.03017)" (2019).


## Usage

```python
# Each step must be done independently by commenting out other steps
from sedd_model import model_encoder
from sedd_model.model_decoder import Decoder

# Step 1: Encoder train on dataset images
model_encoder.train()

# Step 2: Enocode test set images from the trained encoder
model_encoder.encode()

# Step 3: Trained deep decoder network from the encodings
generator = Decoder.create_generator()
Decoder.train(generator)

# Step 4: Test set inferencing
encoder_model = tf.keras.models.load_model("files\\models\\encoder.h5")
decoder_model = tf.keras.models.load_model("files\\models\\decoder.h5")

image_path = r"D:\Dev\Datasets\Images\cats_and_dogs\train\train\cat.3002.jpg"
Decoder.test_image(image_path, encoder_model, decoder_model)
Decoder.plot_generated_images(generator)
```

## Abstract

This paper explores a new framework for lossy image encryption and decryption using a simple shallow encoder neural network E for encryption, and a complex deep decoder neural network D for decryption. E is kept simple so that encoding can be done on low power and portable devices and can in principle be any nonlinear function which outputs an encoded vector. D is trained to decode the encodings using the dataset of image - encoded vector pairs obtained from E and happens independently of E. As the encodings come from E which while being a simple neural network, still has thousands of random parameters and therefore the encodings would be practically impossible to crack without D. This approach differs from autoencoders as D is trained completely independently of E, although the structure may seem similar. Therefore, this paper also explores empirically if a deep neural network can learn to reconstruct the original data in any useful form given the output of a neural network or any other nonlinear function, which can have very useful applications in Cryptanalysis. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the decoded images from D along with some limitations.


## Architecture and Algorithm
* Image Vectorization
![RNN Architecture](https://github.com/chirag2796/SEDD-Networks-Implementation/blob/master/sedd_model/arch_pictures/image_vectorization.JPG)

<br>

* Encoder Architecture
![Encoder Architecture](https://github.com/chirag2796/SEDD-Networks-Implementation/blob/master/sedd_model/arch_pictures/encoder.JPG)

<br>

* Decoder Architecture
![Decoder Architecture](https://github.com/chirag2796/SEDD-Networks-Implementation/blob/master/sedd_model/arch_pictures/decoder.JPG)

<br>

* Training Error and Optimization
![Training Error and Optimization](https://github.com/chirag2796/SEDD-Networks-Implementation/blob/master/sedd_model/arch_pictures/mse.JPG)

<br>

* Results
![Results](https://github.com/chirag2796/SEDD-Networks-Implementation/blob/master/sedd_model/arch_pictures/results.JPG)


## License
[MIT](https://choosealicense.com/licenses/mit/)
##
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) [![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)