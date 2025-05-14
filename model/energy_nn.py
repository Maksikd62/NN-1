from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_energy_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(1,)),
        Dense(5, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_energy_model(model, X, y, epochs=300):
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return model, history
