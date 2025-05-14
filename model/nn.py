from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_nn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)  
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_nn_model(model, X, y, epochs=100):
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return model, history
