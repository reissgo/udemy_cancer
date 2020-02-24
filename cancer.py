print("Here we go, imports...")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

print("now get an preprocess data...")

bcdata_class_of_type_Bunch = datasets.load_breast_cancer()

print(type(bcdata_class_of_type_Bunch))

print(bcdata_class_of_type_Bunch.keys())

# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

bc_data_input_data_part = bcdata_class_of_type_Bunch["data"]
bc_data_output_data_part = bcdata_class_of_type_Bunch["target"]
print(type(bc_data_input_data_part))

print(bc_data_input_data_part.shape)
print(bc_data_output_data_part.shape)

print(bc_data_input_data_part[0])
print(bc_data_output_data_part[0])

print(bc_data_input_data_part[1])
print(bc_data_output_data_part[1])

print(bcdata_class_of_type_Bunch["target_names"])  # ['malignant' 'benign']


bc_data_input_data_part_train, \
bc_data_input_data_part_test, \
bc_data_output_data_part_train, \
bc_data_output_data_part_test = train_test_split(bc_data_input_data_part,bc_data_output_data_part,test_size=0.33)


# now normalise inputs!

# my guess as to what has to be done
# normalise_this_np_array_part1(bc_data_input_data_part_train)
# kind_of_normalise_this_np_array_using_mean_and_var_from_part1(bc_data_input_data_part_test)
# and now the real code...

my_scaler_for_bc_inputs = StandardScaler()

bc_data_input_data_part_train = my_scaler_for_bc_inputs.fit_transform(bc_data_input_data_part_train)
bc_data_input_data_part_test = my_scaler_for_bc_inputs.transform(bc_data_input_data_part_test)

print("Now configure the neural net & training...")
# define the network topology and activation functions

model = tf.keras.models.Sequential([
                        tf.keras.layers.Input(shape=(30,)),
                        tf.keras.layers.Dense(1, activation="sigmoid")
                        ])

# "build the model" - I'd say "define the learning algorithm"
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# "train the model" - I'd say "feed the data to the previously defined topology and learning algorithm"
print("Now train...")
training_callback_history = model.fit(bc_data_input_data_part_train, bc_data_output_data_part_train,
                                      validation_data=(bc_data_input_data_part_test, bc_data_output_data_part_test),
                                      epochs=1000,
                                      verbose=0)


plt.plot(training_callback_history.history["loss"], label='loss')
plt.plot(training_callback_history.history["val_loss"], label='Validation loss')
plt.legend()
plt.show()
plt.plot(training_callback_history.history["accuracy"], label='loss')
plt.plot(training_callback_history.history["val_accuracy"], label='Validation loss')
plt.legend()
plt.show()
# Evaluate the model
print("Now evaluate...")
trains_score = model.evaluate(bc_data_input_data_part_train,bc_data_output_data_part_train)
print(f"trains_score={trains_score}")
test_score = model.evaluate(bc_data_input_data_part_test,bc_data_output_data_part_test)
print(f"test_score={test_score}")
