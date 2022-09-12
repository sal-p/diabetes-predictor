import pickle
import numpy as np
import os
import sklearn

print(os.getcwd())
loaded_model = pickle.load(open("trained_model.sav", "rb"))


# a random data point from the data
input = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input = np.array(input)
input = input.reshape(1, -1)

# standardize data
# scaled_input = scaler.transform(input)

# make prediction
prediction = loaded_model.predict(input)
print(prediction)

if prediction[0] == 0:
  print("Person is non-diabetic")
else:
  print("Person is diabetic")

print("code successfully run")