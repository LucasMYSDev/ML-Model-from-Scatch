from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from adaboost import Adaboost  # Replace "your_module" with the actual module name containing your Adaboost implementation

def test_model():
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target


    y[y==0] = -1

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Create and train the AdaBoost model
    adaboost = Adaboost(n_clf=5)  # Initialize your Adaboost model with appropriate parameters
    adaboost.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = adaboost.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Test the model and print the accuracy
accuracy = test_model()
print("Accuracy:", accuracy)
