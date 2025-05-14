import matplotlib.pyplot as plt

def plot_results(times, durations, y_pred_nn, y_pred_poly, x_test_grid):
    plt.plot(times, durations, 'b.', alpha=0.4, label='Real data')
    plt.plot(x_test_grid, y_pred_nn, 'r-', label='Neural Network')
    plt.plot(x_test_grid, y_pred_poly, 'g--', label='Polynomial Regression')
    plt.xlabel('Time of Day')
    plt.ylabel('Trip Duration (min)')
    plt.legend()
    plt.grid()
    plt.show()
