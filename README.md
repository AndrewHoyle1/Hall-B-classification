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
                          
  

  
