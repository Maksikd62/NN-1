import numpy as np
from model.nn import create_nn_model, train_nn_model
from model.polynomial_regression import create_polynomial_model, predict_with_polynomial_model
from data.trip_data import generate_data
from plot_results import plot_results

X, y = generate_data()

model_nn = create_nn_model(input_shape=(1,))
model_nn, history = train_nn_model(model_nn, X, y, epochs=100)

def time_to_float(hhmm: str):
    h, m = map(int, hhmm.split(':'))
    return h + m / 60

test_times_str = ['10:30', '00:00', '02:40']
test_times = np.array([time_to_float(t) for t in test_times_str]).reshape(-1, 1)

predictions_nn = model_nn.predict(test_times)
for t, p in zip(test_times_str, predictions_nn):
    print(f"NN prediction for {t} — {p[0]:.2f} min")

model_poly, poly = create_polynomial_model(X, y, degree=3)
predictions_poly = predict_with_polynomial_model(model_poly, poly, test_times)
for t, p in zip(test_times_str, predictions_poly):
    print(f"Polynomial regression for {t} — {p:.2f} min")

x_test_grid = np.linspace(0, 24, 500).reshape(-1, 1)
y_pred_nn = model_nn.predict(x_test_grid)
y_pred_poly = predict_with_polynomial_model(model_poly, poly, x_test_grid)

plot_results(X, y, y_pred_nn, y_pred_poly, x_test_grid)
