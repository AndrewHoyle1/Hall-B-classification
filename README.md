# Hall-B-classification
Classifies image data to detect real and simulated image data from drift chambers in Jefferson Lab in Hall_B





# Files associated
The order of the scripts is as follows:
  1. numpy_data.py (converts images to numpy arrays)
  
  2. data_organizer.py (shuffles data and organizes it into testing and training sets)
  
  3. vector.py (runs the data through a pretrained CNN to generate vectors for each image)
  
  4. log_reg.py (uses a Logistic Regression algorithm to classify the vectors made by the previous script)
  
  5. vector_4.py (this is almost the same as the first vector file with some small modifications)
  
  6. vector_4_models file (this is a file that has each model in a sepeate script for easy modifications for each model
                          ,Implement click in vector.py can solve this but has some problems getting it to work)
  
# models tested:


  



# How to use
This is a step by step walkthrough on how to use the vector and other associated files to begin training and testing:

1. Use the **numpy_data.py** to convert the images to h5py dataset with a conversion of 112*112
2. Use **data_organizer.py** to seperate the postive and  negative images into a 75/25 sepeation for training and testing
3. Use Either the **vector.py** or **vector_4.py**  to train the pretrained model you want. This wil create a confusion matrix, a model,    checkpoints, and a loss and accuracy graph. The **vector.py** only has the loss accracy graph but you can just copy and paste.
4. Use **model_test_script** to test the model and checkpoints accuracy. This will give a Confusion Maxtrix,and  f1 report text file.


# Important information for vector,vector_4, and model_test_script
1.
2.
3.

# Resources
This is a place where you can find the websites that are we used to build these scripts and learn how it all works.

1.
