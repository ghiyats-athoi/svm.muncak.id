from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler dari pickle
with open("svm_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = []
        for i in range(1, 15):
            value = request.form.get(f'q{i}')
            if value is None:
                return jsonify({"error": f"Pertanyaan q{i} belum diisi!"})
            data.append(float(value))

        data_array = np.array(data).reshape(1, -1)

        # Apply scaler sebelum prediksi
        scaled_input = scaler.transform(data_array)
        prediction = model.predict(scaled_input)[0]

        # Menentukan pesan risiko berdasarkan prediksi
        risk_message = ""
        if prediction == "SANGAT AMAN":
            risk_message = "Anda diperbolehkan mendaki gunung untuk pendaki ahli ataupun gunung lain yang ingin didaki oleh Anda."
        elif prediction == "AMAN":
            risk_message = "Anda diperbolehkan mendaki gunung untuk pendaki menengah seperti Gunung Karang via Kadu enggang, namun tetap perlu memperhatikan risiko-risiko yang ada pada gunung yang disarankan untuk pendaki level menengah."
        elif prediction == "BAHAYA":
            risk_message = "Anda diperbolehkan melakukan pendakian di gunung untuk pemula berdasarkan kategori gunung di muncak.id seperti gunung Wareksi via waisai atau gunung Cibareno via cibareno, namun perbaikan kekuatan fisik dan mental secara bertahap tetap diperlukan."
        elif prediction == "SANGAT BAHAYA":
            risk_message = "Disarankan untuk tidak melakukan pendakian terlebih dahulu. Fokuskan pada perbaikan faktor fisik individu sebagai fondasi utama, seperti rutin berolahraga (misalnya jogging, latihan kekuatan ringan, atau aktivitas kardio lainnya)."

        return jsonify({
            "input_values": data,
            "prediction": prediction,
            "risk_message": risk_message  # Kirimkan pesan risiko
        })

    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True)
