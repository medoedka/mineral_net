import keras
from tensorflow.keras.preprocessing import image
from flask import Flask
from flask import request
from PIL import Image
import json
import os

app = Flask(__name__)

model_path = os.path.abspath('mineral_model.h5')
classes_path = os.path.abspath('mineral_classes.json')
print(classes_path)
print(model_path)

model = keras.models.load_model(model_path)


def image_preprocessor(img):
    test_img = image.img_to_array(img)
    test_img = test_img.reshape(1, 380, 380, 3)
    return test_img


def make_prediction(img):
    image_to_predict = image_preprocessor(img)
    predictions = model.predict(image_to_predict)
    top10_minerals = (-predictions).argsort()[0][:10]
    top10_preds = [predictions[0][place] for place in top10_minerals]
    return top10_minerals, top10_preds


with open(classes_path) as json_file:
    mineral_class_names = json.load(json_file)


@app.route('/upload_image', methods=['POST'])
def predictions():
    file = request.files['file']
    pil_image = Image.open(file).resize((380,380), Image.ANTIALIAS)
    minerals, props = make_prediction(pil_image)
    total_predictions = []
    for index in range(len(minerals)):
        prediction = {}
        prediction["plant_name"] = mineral_class_names[str(minerals[index])]
        prediction["propability"] = round(props[index] * 100, 2)
        total_predictions.append(prediction)
    return json.dumps(total_predictions)


if __name__ == "__main__":
    app.run(host='157.230.25.122', port=5000)
