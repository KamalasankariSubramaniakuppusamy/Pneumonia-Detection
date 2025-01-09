Pneumonia Detection from Chest X-Ray Images
A deep learning-based model that detects pneumonia in chest X-ray images using a Convolutional Neural Network (CNN). The project uses a pre-trained model (MobileNetV2) fine-tuned on a pneumonia dataset for accurate classification. The model is deployed as a user-friendly web application using Streamlit for real-time predictions.

🚀 Features
Real-time Prediction: Upload a chest X-ray image and get an immediate prediction of whether the patient has pneumonia or not.
User-Friendly Web Interface: Built with Streamlit for an interactive and intuitive user experience.
Image Preprocessing: Prepares uploaded images for model inference with resizing and normalization.
🧑‍💻 Requirements
To run this project locally, you'll need to install the following dependencies:

TensorFlow: For building and running the deep learning model.
Streamlit: For creating the interactive web application.
NumPy: For numerical operations.
OpenCV: For image manipulation.
Scikit-learn: For additional machine learning utilities.
You can install all required dependencies by running:

bash
Copy code
pip install -r requirements.txt
The requirements.txt file includes the following:

text
Copy code
tensorflow
numpy
opencv-python
scikit-learn
streamlit
📂 Project Structure
The project follows a simple folder structure:

text
Copy code
pneumonia-detection/
│
├── app.py               # Streamlit application for interacting with the model
├── model.py             # Code for training and saving the model
├── pneumonia_model.h5   # Trained model file
├── /dataset             # Folder containing the chest X-ray images dataset
├── requirements.txt     # List of project dependencies
└── README.md            # Project documentation
app.py: The main file that runs the Streamlit app and loads the pre-trained model for prediction.
model.py: Contains the model architecture, training code, and saving the trained model (pneumonia_model.h5).
/dataset: A folder where you place your chest X-ray images for training or inference.
requirements.txt: List of Python libraries required to run the project.
pneumonia_model.h5: The trained deep learning model saved in HDF5 format.
📈 Model Training
Data Preparation: The dataset contains chest X-ray images of patients labeled as normal or pneumonia. The images are split into training and validation sets.
Model Architecture: The model uses MobileNetV2 as the base pre-trained model for transfer learning, fine-tuned to classify pneumonia vs. normal X-ray images.
Training: The model is trained using images resized to 224x224 and normalized to a [0, 1] scale.
Model Saving: The trained model is saved as pneumonia_model.h5, which is then used for making predictions in the Streamlit app.
To train the model and save it:

bash
Copy code
python model.py
🌐 Running the Streamlit App
After setting up your environment and installing dependencies, run the Streamlit app using:
bash
Copy code
streamlit run app.py
Open the URL provided by Streamlit (typically http://localhost:8501) in your web browser.
Upload a chest X-ray image in .jpg, .jpeg, or .png format to predict if the patient has PNEUMONIA or is NORMAL.
⚙️ How the Prediction Works
The app loads the pre-trained model from the file pneumonia_model.h5.
It processes the uploaded image by resizing it to the input size required by the model (224x224 pixels) and normalizing it.
The model makes a prediction, and the app shows whether the image is classified as NORMAL or PNEUMONIA.
🛠️ Technologies Used
TensorFlow / Keras: Deep learning framework for building and training the model.
Streamlit: A Python library used for building the interactive web application.
OpenCV: Used for image manipulation and preprocessing.
NumPy: Used for numerical operations on image arrays.
Scikit-learn: Utilities for machine learning tasks.
📄 License
This project is open source and available under the MIT License.

📬 Contact
If you have any questions or suggestions, feel free to reach out to me via:

GitHub: 
Email: iamkamalamskls@gmail.com
