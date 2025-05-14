import numpy as np

def generate_data():
    np.random.seed(42)
    times = np.linspace(0, 23.99, 500)  
    durations = 30 + 20 * np.sin((times - 7) / 24 * 2 * np.pi) + np.random.normal(0, 3, size=times.shape)
    
    X = times.reshape(-1, 1)  
    y = durations  
    return X, y

if __name__ == "__main__":
    X, y = generate_data()
    print(f"First 5 entries: {list(zip(X[:5], y[:5]))}")
