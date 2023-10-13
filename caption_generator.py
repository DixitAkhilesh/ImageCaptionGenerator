from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import pyttsx3
from PIL import Image
import matplotlib.pyplot as plt
import requests
import io

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# load the tokenizer
tokenizer = load(open('./tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 35
# load the model
model = load_model('./best_model.h5')


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, tokenizer, image, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


def process_image(image_name, model, tokenizer, max_length):
    # Extract features from the image
    image = extract_features(image_name)
    
    # Open the image for display
    Image_prev = Image.open(image_name)
    # plt.imshow(Image_prev)

    # Generate a caption for the image
    query = predict_caption(model, tokenizer, image, max_length)
    stopwords = ['startseq', 'endseq']
    querywords = query.split()

    # Remove stopwords from the caption
    resultwords = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)

    # Determine whether the caption starts with a vowel
    def starts_with_vowel(result):
        # Define a set of vowels
        vowels = {'a', 'e', 'i', 'o', 'u'}
        if result and result[0] in vowels:
            return True
        else:
            return False

    # Generate the response
    if starts_with_vowel(result):
        ans = 'I think, It is an ' + result
    else:
        ans = 'I think, It is a ' + result

    return ans

def get_image_input(image_name):
    return process_image(image_name, model, tokenizer, max_length)