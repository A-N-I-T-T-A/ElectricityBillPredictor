from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

def predict_bill(units):
    """Predict electricity bill using trained model."""
    return round(model.predict([[units]])[0], 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    bill_amount = None
    if request.method == 'POST':
        try:
            units = float(request.form['units'])
            bill_amount = predict_bill(units)
        except ValueError:
            bill_amount = "Invalid input! Please enter a valid number."
    return render_template('index.html', bill_amount=bill_amount)

if __name__ == '__main__':
    app.run(debug=True)
