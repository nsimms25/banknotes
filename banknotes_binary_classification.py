"""
Design a network to see if a given dollar bill is real or fake base on training examples.
banknotes.csv:
    features - variance, skewness, kurtosis, entropy.
    labels - real or fake
"""

# Check if the variables are linearly separable
# Import seaborn
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

banknotes = pd.read_csv('banknotes.csv')

'''
# Use pair plot and set the hue to be our class
# Looks like our data is linearly separable.
import matplotlib.pyplot as plt
import seaborn as sns 
sns.pairplot(banknotes, hue='class', diag_kind='hist')
plt.show()

# Describe the data
print('Data set stats: \n', banknotes.describe())
print('Observations per class: \n', banknotes['class'].value_counts())
'''

# Train test split data
labels = banknotes['class']
banknotes = banknotes.drop('class', axis=1)

# Split data into train and test to verify learning has occurred.
X_train, X_test, y_train, y_test = train_test_split(banknotes, labels)

# Create a sequential model
model = Sequential()

# Add a dense layer
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

'''
#Final output with accuracy of 0.97 
#Which may change due to pseudo-random splitting of data in sklearn train_test_split. 
#32/343 [=>............................] - ETA: 0s
#343/343 [==============================] - 0s 70us/step
#Accuracy: 0.9737609329446064
'''
