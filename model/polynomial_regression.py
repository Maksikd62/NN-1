from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def create_polynomial_model(X, y, degree=3):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    return model, poly

def predict_with_polynomial_model(model, poly, X):
    X_poly = poly.transform(X)
    return model.predict(X_poly)
