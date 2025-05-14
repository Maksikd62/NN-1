import numpy as np
from data.energy_data import load_monthly_data
from model.energy_nn import create_energy_model, train_energy_model

X, y = load_monthly_data()

model = create_energy_model()
model, history = train_energy_model(model, X, y)

test_months = np.array([[4], [7], [12]])
predictions = model.predict(test_months)

months_names = ['April', 'July', 'December']
for name, pred in zip(months_names, predictions):
    print(f"Energy consumption forecast for {name}: {pred[0]:.2f} kWh")
