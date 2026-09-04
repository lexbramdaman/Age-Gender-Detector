import cv2
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent
image = cv2.imread(str(base_dir / "image0.jpg"))
if image is None:
    raise FileNotFoundError(f"Could not load input image from {base_dir / 'image0.jpg'}")
image = cv2.resize(image, (720, 640))


face1 = "opencv_face_detector.pbtxt"
face2 = "opencv_face_detector_uint8.pb"
age1 = "age_deploy.prototxt"
age2 = "age_net.caffemodel"
gen1 = "gender_deploy.prototxt"
gen2 = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

face = cv2.dnn.readNet(str(base_dir / face2), str(base_dir / face1))
age = cv2.dnn.readNet(str(base_dir / age2), str(base_dir / age1))
gender = cv2.dnn.readNet(str(base_dir / gen2), str(base_dir / gen1))

agecat = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gendercat = ["Male", "Female"]

framecv = image.copy()

frameHeight = framecv.shape[0]
frameWidth = framecv.shape[1]
blob = cv2.dnn.blobFromImage(framecv, 1.0, (300, 300), [104, 117, 123], True, False)

face.setInput(blob)
detections = face.forward()

faceBoxes = []
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)

        faceBoxes.append([x1, y1, x2, y2])

        cv2.rectangle(framecv, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

if not faceBoxes:
    print("No face detected")

for faceBox in faceBoxes:
    face = framecv[max(0, faceBox[1]-15): min(faceBox[3]+15, framecv.shape[0]-1), max(0, faceBox[0]-15):min(faceBox[2]+15, framecv.shape[1]-1)]

blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

gender.setInput(blob)
genderPrediction = gender.forward()
gender = gendercat[genderPrediction[0].argmax()]

age.setInput(blob)
agePrediction = age.forward()
age = agecat[agePrediction[0].argmax()]

cv2.putText(framecv, f"{gender}, {age}", (faceBox[0]-150, faceBox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (217, 0, 0), 4, cv2.LINE_AA)

plt.figure(figsize=(7, 7))
plt.imshow(cv2.cvtColor(framecv, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()