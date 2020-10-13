# Machine Learning Engineer Nanodegree - Capstone Proposal
# Dog Breed Classifier

Gustavo Cruz 
October, 2020

### Domain Background

The automatic classification of images is a fascinating problem; the objective is to generate a fully automated process to add metadata to digital images to group them based on common features; the process is also known as automatic image annotation. This kind of classification tasks were usually assigned to human experts. But, with the advent of new and more powerful technologies like machine learning, deep learning, and the almost infinite computer power provided by cloud computing, we can automate these processes in a scalable way.

One field of interest is the classification of dog breeds; the Fédération Cynologique Internationale officially recognizes 360 breeds. However, for non-specialized people can be somewhat difficult to distinguish one from another. Having an automatic application to classify a dog by their breed can potentially predict if it is or not a pure breed and can help identify his ancestry. One interesting thing about data analysis and classification using AI is that it can uncover relationships or causation. For example, by analyzing the picture of a dog, we can infer their ancestry or breed and predict how easy or hard to train is.

### Problem Statement

Pets and especially dogs are an essential part of our lives, however, how well do we know our furry partners? Are we able to identify the breed of our dogs? or their ancestry just by seeing them? It can be difficult for a regular person because it is not easy to differentiate and associate the different visual characteristics of a dog. What about seeing the pictures of different dogs? Are we able to identify if they are of the same breed or related to some degree? This kind of analysis could take time and resources if we do it manually. With machine learning, this could be done in milliseconds in a repeatable and scalable manner. We can establish metrics to measure the prediction's accuracy based on the number of right guesses and potentially measure the degree of relationship between dogs by their breed.

### Datasets and Inputs

We will be using two datasets of images, the first one filled with dog images and the second one with people photos. The objective is to not only classify dog breeds but differentiate between persons and pets. Udacity provided the human dataset. While it shows their real names and faces, they are from public figures and easily available to anyone to avoid privacy concerns.

* [Dog dataset] (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)  8351 images (train: 6680, test: 836, valid: 835) It has 133 folders for each of the dog breeds, the pictures have different height and width, so to use this data in training, we should normalize the data resizing the images to the same dimensions. Analyzing the data shows that the data distribution is not balanced by labels.
* [Human dataset] (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) 13234 images. Each of the images has the same 250x250 size; this set's data is also not balanced since there are labels(persons) with a single photo while others have many.

### Solution Statement

The problem is a multiclass classification problem. Artificial neural networks are very effective solutions for high dimensionality problems; they are theoretically complex; however, frameworks like TensorFlow and others can help ease the development of solution models. The solution algorithm will use Convolutional neural networks, and it would be structured in two main parts:

* Feature learning. This part consist of the convolutional layers and pooling layers
* Classification. This part is in charge of classifying the images on categories extracted by the features layer

### Benchmark Model

The problem is a multiclass classification problem. Artificial neural networks are very effective solutions for high dimensionality problems; they are theoretically complex; however, frameworks like TensorFlow and others can help ease the development of solution models. The solution algorithm will use Convolutional neural networks, and it would be structured in two main parts:

### Evaluation Metrics

For this project, I chose the F1 score method to evaluate the model's effectiveness since it is well suited to classification problems. This approach assigns the resulting prediction a value between 0 and 1 and is the harmonic mean of precision and recall.

### Project Design

The project would generate the model using the following steps:

* Step 1: Import the datasets into the Jupyter environment.
* Step 2: Implement human image recognition.
* Step 3: Implement dog image recognition.
* Step 4: Create a CNN from scratch, and train, validate and test that model.
* Step 5: Create a CNN to classify dog breeds using transfer learning.
* Step 6: Implement an algorithm combining those models.
* Step 7: Evaluate the following test cases:
    * If a human is detected, return a resembling dog breed.
    * If a dog photo is detected, return their breed.
    * If the image cannot be classified, return an error.

### References

* https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
* https://www.hillspet.com/dog-care/behavior-appearance/how-many-dog-breeds-are-there 
* https://medium.com/swlh/convolutional-neural-networks-for-multiclass-image-classification-a-beginners-guide-to-6dbc09fabbd 
* https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa
* https://towardsdatascience.com/intuitively-create-cnn-for-fashion-image-multi-class-classification-6e31421d5227
* https://missinglink.ai/guides/neural-network-concepts/classification-neural-networks-neural-network-right-choice/ 
* https://en.wikipedia.org/wiki/Convolutional_neural_network