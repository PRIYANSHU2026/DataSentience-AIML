import torch
import torchvision.transforms as transforms
import cv2
from torchvision.models import resnet18
from PIL import Image
import torch.nn as nn

# Emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with matching structure
model = resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # For grayscale input
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 7)  # Match the trained structure
)
model.load_state_dict(torch.load("models/face_emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Match training normalization
])

# Load face detector
face_cascade = cv2.CascadeClassifier(
    f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_img = Image.fromarray(face)
        face_tensor = transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output)
            label = classes[pred.item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Facial Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
