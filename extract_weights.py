import numpy as np
from keras.models import load_model

model = load_model('home/zwang/cosmic-ray-nn/best_model [l: 0.1395, vl: 0.1262].h5')

with open(f'home/zwang/cosmic-ray-nn/model_weights', 'w') as file:
        for layer in model.layers:
            weights = layer.get_weights()
            file.write(f"Weights for layer {layer.name} are: {weights} \n")

