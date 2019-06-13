# Hall-B-classification
Classifies image data to detect real and simulated image data

The order of the scripts is as follows:
  1)numpy_data.py (converts images to numpy arrays)
  2)data_organizer.py (shuffles data and organizes it into testing and training sets)
  3)vector.py (runs the data through a pretrained CNN to generate vectors for each image)
  4)log_reg.py (uses a Logistic Regression algorithm to classify the vectors made by the previous script)
