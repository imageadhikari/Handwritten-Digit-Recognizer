import gradio as gr
from tensorflow import keras
import numpy as np

model = keras.models.load_model("savedmodel/saved_model.h5")

def classify(image):
    im_reshape = image.reshape(1,784)
    im_resize = im_reshape/255.0
    prediction = model.predict(im_resize)
    pred = np.argmax(prediction)
    return pred

title="Digit recognition app"
description="A Neural Network model trained on MNIST dataset. Start by drawing a single digit, and let the model recognize it."

draw = gr.Interface(fn=classify, 
             inputs="sketchpad",
             outputs="label",
             title=title,
             description=description,
             )

draw.launch(share=True)





