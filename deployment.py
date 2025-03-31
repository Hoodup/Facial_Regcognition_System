import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import pickle
from PIL import Image

# Load your pre-trained SVM model
def load_svm_model():
    with open('svm_model_100.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

# Load VGG16 model
def load_vgg16_model():
    from keras.applications import VGG16
    return VGG16(weights='imagenet', include_top=False, pooling='avg')

# Detect faces in an image
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = image[y:y+h, x:x+w]
        cropped_faces.append(cropped_face)
    return cropped_faces

# Get embeddings for the detected faces
def get_embeddings(face_images, model):
    embeddings = []
    for face in face_images:
        face_resized = cv2.resize(face, (160, 160))
        face_array = image.img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)
        embedding = model.predict(face_array)
        embeddings.append(embedding.flatten())
    return embeddings

# Classify the faces with the trained SVM model
def classify_faces(embeddings, clf):
    predictions = []
    for embedding in embeddings:
        prediction = clf.predict([embedding])
        predictions.append(prediction[0])
    return predictions

# Streamlit app interface
st.title('Face Recognition App')

# Load models
clf = load_svm_model()
vgg16_model = load_vgg16_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save uploaded image to a temporary file
    img = Image.open(uploaded_file)
    img.save("temp_image.jpg")

    # Detect faces and generate embeddings
    faces = detect_faces("temp_image.jpg")
    embeddings = get_embeddings(faces, vgg16_model)
    
    # Classify faces
    predictions = classify_faces(embeddings, clf)

    # Display results
    for idx, prediction in enumerate(predictions):
        st.write(f"Face {idx+1} is predicted as: {prediction}")

# Take picture using webcam
if st.button('Take a Picture'):
    run = st.empty()
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the video feed
        cv2.imshow('Webcam', frame)

        # Break the loop when user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Capture and save the image when 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("captured_image.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Process the captured image
    faces = detect_faces("captured_image.jpg")
    embeddings = get_embeddings(faces, vgg16_model)

    # Classify faces
    predictions = classify_faces(embeddings, clf)

    # Display results
    for idx, prediction in enumerate(predictions):
        st.write(f"Face {idx+1} is predicted as: {prediction}")
