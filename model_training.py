from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from data_preprocessing import load_data

df = load_data()

X = df[['ENGINESIZE']]
y = df[['CO2EMISSIONS']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "slr_model.joblib")
