from flask import Flask, render_template, request
from config import FEATURES, FEATURE_OPTIONS, MOST_COMMON, MODEL_FILES
import pandas as pd
import pickle

app = Flask(__name__)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    selected_model = None

    if request.method == "POST":

        selected_model = request.form.get("method")
        model_filename = MODEL_FILES.get(selected_model)
        with open(model_filename, "rb") as f:
            model = pickle.load(f)

        user_input = {}
        for feature in FEATURES:
            val = request.form.get(feature)
            if val in [None, "", "none", "?"]:
                val = MOST_COMMON[feature]
            user_input[feature] = str(val)  

        input_df = pd.DataFrame([user_input])
        encoded_input = encoder.transform(input_df)
        prediction = model.predict(encoded_input)[0]
        result = "Beracun" if prediction == 1 else "Tidak Beracun/Dapat Dimakan"

        print("Input user:", user_input)
        print("Encoded input:", encoded_input)
        print("Prediction raw:", prediction)

    return render_template(
        "index.html",
        features=FEATURES,
        FEATURE_OPTIONS=FEATURE_OPTIONS,
        models=MODEL_FILES.keys(),
        result=result,
        selected_model=selected_model
    )

if __name__ == "__main__":
    app.run(debug=True)
