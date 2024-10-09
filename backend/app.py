from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from collections import Counter

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

model = load_model("CNN_model.h5", compile=False)
mapper = {
    0: {
        "name": "Arborio",
        "remedy": "Arborio is a short-grain rice variety renowned for its high starch content, imparting a creamy texture to dishes like risotto. Originating in Italy, it is favored for its ability to absorb flavors and create delectably rich and velvety rice-based dishes."
    },
    1: {
        "name": "Basmati",
        "remedy": " Hailing from the Indian subcontinent, Basmati rice is characterized by its long grains and distinctive aroma. Its fluffy texture, fragrant scent, and nutty flavor make it a popular choice for a variety of dishes, especially in South Asian and Middle Eastern cuisines."
    },
    2: {
        "name": "Ipsala",
        "remedy": "Ipsala is a type of Turkish rice known for its unique flavor and texture. Grown in the Ipsala region of Turkey, this medium-grain rice is often used in traditional Turkish cuisine, contributing to dishes with a delightful blend of taste and consistency."
    },
    3: {
        "name": "Jasmine",
        "remedy": "With its long, slender grains and floral aroma, Jasmine rice is a staple in Southeast Asian cuisine. Originating in Thailand, this fragrant rice complements a wide range of dishes, providing a light and fluffy texture that enhances the overall dining experience."
    },
    4: {
        "name": "Karacadag",
        "remedy": "Karacadag rice is a local variety grown in the Karacadag region of Turkey. With its medium grains, it adds a hearty and substantial character to Turkish dishes. This rice variety is valued for its versatility and ability to adapt to diverse culinary applications in Turkish cuisine."
    },
}



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"message": "no message part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 244))
        print(img.size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.transpose(img_array, (0, 2, 1, 3))

        prediction1 = model.predict(img_array)
        # prediction2 = model2.predict(img_array)
        # prediction3 = model3.predict(img_array)
        prediction_1_confidence = max(prediction1[0])
        # prediction_2_confidence = max(prediction2[0])
        # prediction_3_confidence = max(prediction3[0])
        predicted_classes_idx1 = np.argmax(prediction1[0])
        # predicted_classes_idx2 = np.argmax(prediction2[0])
        # predicted_classes_idx3 = np.argmax(prediction3[0])
        disease_name1 = mapper[int(predicted_classes_idx1)]["name"]
        disease_remedy1 = mapper[int(predicted_classes_idx1)]["remedy"]
        # disease_name2 = mapper[int(predicted_classes_idx2)]["name"]
        # disease_remedy2 = mapper[int(predicted_classes_idx2)]["remedy"]
        # disease_name3 = mapper[int(predicted_classes_idx3)]["name"]
        # disease_remedy3 = mapper[int(predicted_classes_idx3)]["remedy"]
        # predicted_classes = [
        #     int(predicted_classes_idx1),
        #     int(predicted_classes_idx2),
        #     int(predicted_classes_idx3),
        # ]
        # class_counts = Counter(predicted_classes)

        # majority_class = class_counts.most_common(1)[0][0]

        # ensembled_disease_name = mapper[majority_class]["name"]
        # ensembled_disease_remedy = mapper[majority_class]["remedy"]
        return jsonify(
            {
                "result": [
                    {
                        "model": "VGG19",
                        "predicted_class": int(predicted_classes_idx1),
                        "accuracy" : str(prediction_1_confidence),
                        "name": disease_name1,
                        "remedy": disease_remedy1,
                    },
                    # {
                    #     "model": "InceptionResNetv2",
                    #     "predicted_class": int(predicted_classes_idx2),
                    #     "accuracy" : str(prediction_2_confidence),
                    #     "name": disease_name2,
                    #     "remedy": disease_remedy2,
                    # },
                    # {
                    #     "model": "Custom Model",
                    #     "predicted_class": int(predicted_classes_idx3),
                    #     "accuracy" : str(prediction_3_confidence),
                    #     "name": disease_name3,
                    #     "remedy": disease_remedy3,
                    # },
                ]
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
