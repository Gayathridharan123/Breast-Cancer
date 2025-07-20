from flask import Flask, render_template_string, request 
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained 30-feature model
model = pickle.load(open('model/model.pkl', 'rb'))

all_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

visible_features = [
    'radius_mean',
    'perimeter_mean',
    'area_mean',
    'concave points_mean',
    'concavity_mean',
    'compactness_mean',
    'radius_worst',
    'perimeter_worst',
    'area_worst',
    'concave points_worst'
]

max_values = {
    'radius_mean': 30,
    'perimeter_mean': 190,
    'area_mean': 2500,
    'concave points_mean': 0.2,
    'concavity_mean': 0.6,
    'compactness_mean': 0.35,
    'radius_worst': 40,
    'perimeter_worst': 270,
    'area_worst': 4000,
    'concave points_worst': 0.35
}

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Breast Cancer Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #e9f0f8, #f7fafc, #dbe9f4);
            padding: 40px;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            background: linear-gradient(145deg, #ffffff, #f0f5fa);
            padding: 30px 40px;
            max-width: 600px;
            margin: auto;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            border: 1px solid #c3d0e8;
        }
        h1 {
            text-align: center;
            color: #2d4c62;
            margin-bottom: 30px;
            text-shadow: 0 1px 2px rgba(45,76,98,0.3);
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        label {
            font-weight: 600;
            margin-bottom: 6px;
            color: #3a5068;
            letter-spacing: 0.03em;
        }
        input[type="number"] {
            padding: 12px 15px;
            border-radius: 10px;
            border: 1.8px solid transparent;
            font-size: 15px;
            font-weight: 500;
            background: linear-gradient(120deg, #f5f8fc, #d9e2ec);
            box-shadow: inset 2px 2px 5px #c0cee7, inset -2px -2px 5px #f9fdff;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        input[type="number"]:focus {
            border-color: #627d98;
            box-shadow: 0 0 8px 1px #7898b3;
            outline: none;
            background: #f0f5fb;
        }
        input[type="submit"] {
            margin-top: 25px;
            padding: 14px;
            background: linear-gradient(120deg, #2d4c62, #466b8a);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 6px 12px rgba(45,76,98,0.4);
            transition: background 0.3s ease;
        }
        input[type="submit"]:hover {
            background: linear-gradient(120deg, #1e3751, #3a5a7a);
        }
        .result {
            margin-top: 30px;
            font-size: 20px;
            text-align: center;
            color: #2a7a2a;
            font-weight: 700;
            text-shadow: 0 1px 2px #a3d3a3;
        }
        .error {
            margin-top: 20px;
            font-size: 15px;
            color: #b44242;
            text-align: center;
            font-weight: 600;
            text-shadow: 0 1px 1px #d9a3a3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Breast Cancer Prediction</h1>

        <form method="POST">
            {% for name in visible_features %}
                <label>{{ name.replace('_', ' ').capitalize() }}</label>
                <input type="number" step="0.01" name="{{ name }}" required max="{{ max_values[name] }}"
                       placeholder="Max: {{ max_values[name] }}">
            {% endfor %}

            <!-- Hidden inputs for the other 20 features -->
            {% for name in all_features %}
                {% if name not in visible_features %}
                    <input type="hidden" name="{{ name }}" value="0">
                {% endif %}
            {% endfor %}
            
            <input type="submit" value="Predict">
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
        {% if result %}
            <div class="result">{{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    error = None
    if request.method == 'POST':
        try:
            input_data = []
            for feature in all_features:
                val = request.form.get(feature)
                if val is None:
                    error = f"Missing input for {feature}"
                    break
                try:
                    fval = float(val)
                except ValueError:
                    error = f"Invalid number for {feature}"
                    break

                if feature in max_values and fval > max_values[feature]:
                    error = f"Value for {feature.replace('_',' ').capitalize()} exceeds max allowed ({max_values[feature]})"
                    break

                input_data.append(fval)

            if not error:
                input_array = np.array([input_data])
                prediction = model.predict(input_array)[0]
                result = 'Malignant (Cancer)' if prediction == 0 else 'Benign (Non-cancerous)'

        except Exception as e:
            error = f"Error in prediction: {e}"

    return render_template_string(html_template,
                                  visible_features=visible_features,
                                  all_features=all_features,
                                  max_values=max_values,
                                  result=result,
                                  error=error)

if __name__ == '__main__':
    app.run(debug=True)
