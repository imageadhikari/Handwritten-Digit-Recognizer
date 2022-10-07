import gradio as gr
from tensorflow import keras
import numpy as np

model = keras.models.load_model("saved_model/")

def classify(image):
    im_reshape = image.reshape(1,784)
    im_resize = im_reshape/255.0
    prediction = model.predict(im_resize)
    pred = np.argmax(prediction)
    return pred

draw = gr.Interface(fn=classify, 
             inputs="sketchpad",
             outputs="label")

draw.launch(share=True)

