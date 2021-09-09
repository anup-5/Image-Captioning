# Image Captioning with Keras
![](https://camo.githubusercontent.com/1c502d149c62da4bd1055404c29743154b7bdd316aab0d466025751c2df7e163/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f507974686f6e2d332e382d626c756576696f6c6574)

Image Captioning is a Deep Learning model which generates textual description for given images.


## Motivation 

Image captioning has a huge amount of real world applications:

- Self driving cars — Automatic driving is one of the biggest challenges and if we can properly caption the scene around the car, it can give a boost to the self driving system.
- Aid to the blind — We can create a product for the blind which will guide them travelling on the roads without the support of anyone else. We can do this by first converting the scene into text and then the text to voice. Both are now famous applications of Deep Learning. Refer this [link](https://www.youtube.com/watch?v=rLyF4XQLwr0) where its shown how Nvidia research is trying to create such a product.
- CCTV cameras are everywhere today, but along with viewing the world, if we can also generate relevant captions, then we can raise alarms as soon as there is some malicious activity going on somewhere. This could probably help reduce some crime and/or accidents.
- Automatic Captioning can help, make Google Image Search as good as Google Search, as then every image could be first converted into a caption and then search can be performed based on the caption.

## Prerequisites

You must be familiar with basic Deep Learning concepts like Multi-layered Perceptrons, Convolution Neural Networks, Recurrent Neural Networks, Transfer Learning, Gradient Descent, Backpropagation, Overfitting, Probability, Text Processing, Python syntax and data structures, Keras library, etc.

## Data Collection

There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But I have used the Flickr 8k dataset. This dataset contains 8000 images each with 5 captions.

Dataset - https://www.kaggle.com/shadabhussain/flickr8k

## Data Cleaning

When we deal with text, we generally perform some basic cleaning like lower-casing all the words, removing special tokens, eliminating words which contain numbers, etc. After performing data cleaning I created a vocabulary of of 8424 unique words across all the 40000 image captions. Finally I considered only those words which occur at least 10 times in the entire corpus which reduces my vocabulary to 1845 unique words.

## Data Preprocessing — Images

Images are nothing but input (X) to our model. As we know that any input to a model must be given in the form of a vector.
We need to convert every image into a fixed sized vector which can then be fed as input to the neural network. For this purpose, we opt for __Transfer Learning__ by using the __ResNet50 model__ which was the winner of the ImageNet challenge in 2015 with an error rate of 3.57%.

This model was trained on Imagenet dataset to perform image classification on 1000 different classes of images. However, our purpose here is not to classify the image but just get fixed-length informative vector for each image. This process is called __automatic feature engineering__. Hence, I just removed the last softmax layer from the model and extract a 2048 length vector (__bottleneck features__) for every image.

![Image](https://miro.medium.com/max/2000/1*9VoYufkvd-hBxK3p2NEWmw.png)

## Data Preprocessing — Captions

We must note that captions are something that we want to predict. So during the training period, captions will be the target variables (Y) that the model is learning to predict. But the prediction of the entire caption, given the image does not happen at once. We will predict the caption __word by word__. Thus we need to encode each word into a fixed sized vector.

### Word Embeddings

We will map the every word (index) to a 50-dimensional vector and for this purpose, I have used a pre-trained GLOVE Model: glove.6B.50d.
Now, for all the 1845 unique words in our vocabulary, I have created an embedding matrix which will be loaded into the model before training.

## Model Architecture

Since the input consists of two parts, an image vector and a partial caption, we cannot use the Sequential API provided by the Keras library. For this reason, we use the Functional API which allows us to create Merge Models.

![Image](https://miro.medium.com/max/2000/1*rfYN2EELhLvp2Van3Jo-Yw.jpeg)

Model Architecture as follows:

![Screenshot (1042)](https://user-images.githubusercontent.com/86401425/132658081-37324e8f-c0e0-45a7-adcb-d9f992a1ba86.png)

Model Summary as follows:

![Screenshot (1040)](https://user-images.githubusercontent.com/86401425/132658190-0e9fcbeb-d3a7-4988-b285-69892b5e371d.png)

We had created an embedding matrix from a pre-trained Glove model which we need to include in the model before starting the training:

`model.layers[2].set_weights([embedding_matrix])`

`model.layers[2].trainable = False`

Notice that since we are using a pre-trained embedding layer, we need to freeze it (trainable = False), before training the model, so that it does not get updated during the backpropagation.

Finally we compile the model using the adam optimizer:

`model.compile(loss=’categorical_crossentropy’, optimizer=’adam’)`

Finally the weights of the model will be updated through backpropagation algorithm and the model will learn to output a word, given an image feature vector and a partial caption. So in summary, we have:

Input_1 -> Partial Caption

Input_2 -> Image feature vector

Output -> An appropriate word, next in the sequence of partial caption provided in the input_1 (or in probability terms we say conditioned on image vector and the partial caption)

__Time Taken:__ I have trained this model on Google Colab using GPU and it took me around 5 to 6 hours to train the model completely. However if you train it on a PC without GPU, it could take anywhere from 8 to 16 hours depending on the configuration of your system.

## Inference and Evaluation

Now let's see how good the model is and try to generate captions on images from the test dataset.

![Screenshot (1057)](https://user-images.githubusercontent.com/86401425/132666957-04e7154d-f662-4010-9b31-20f6012261c2.png)

![Screenshot (1058)](https://user-images.githubusercontent.com/86401425/132667085-12516dba-a0b1-4c12-bf28-61a8a14df2f2.png)

![Screenshot (1056)](https://user-images.githubusercontent.com/86401425/132667239-f81122e9-825e-47a0-a473-7eb8907ea403.png)

![Screenshot (1060)](https://user-images.githubusercontent.com/86401425/132667304-810fa344-6761-4924-8ce1-898feb843e0b.png)

![Screenshot (1059)](https://user-images.githubusercontent.com/86401425/132667402-c4b3b9dd-2698-4028-8f0e-add6d5cf0db7.png)


Of course, no model in the world is ever perfect and this model also makes mistakes. Let’s look at some examples where the captions are not very relevant and sometimes even irrelevant:

![Screenshot (1061)](https://user-images.githubusercontent.com/86401425/132668665-9e9ba84b-7206-4d23-8690-ad7350a63b21.png)

![Screenshot (1062)](https://user-images.githubusercontent.com/86401425/132668781-b5156f4f-375d-4805-b9d5-675e942eb87b.png)

![Screenshot (1063)](https://user-images.githubusercontent.com/86401425/132668828-f4e05d90-b317-4693-9f33-13b6cd498414.png)

![Screenshot (1064)](https://user-images.githubusercontent.com/86401425/132668871-7fb6ac2f-940d-4089-997d-5a4a904137b7.png)


Clearly, the model tried its best to understand the scenarios but still some captions are not the good one.

So all in all, I must say that my Image Captioning model, without any rigorous hyper-parameter tuning does a decent job in generating captions for images.


## Improvements and Future Work

Of course this is just a first-cut solution and a lot of modifications can be made to improve this solution like:

- Using a larger dataset.
- Changing the model architecture, e.g. include an attention module.
- Doing more hyper parameter tuning (learning rate, batch size, number of layers, number of units, dropout rate, batch normalization etc.).
- Use the cross validation set to understand overfitting.

## Reference

https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
