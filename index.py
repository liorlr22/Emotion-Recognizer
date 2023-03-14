import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(r"opencv\data\haarcascades\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face, actions=['emotion'])
        except Exception as e:
            result = "neutral"

        finally:
            for face_result in result:
                try:
                    emotion = max(face_result['emotion'], key=face_result['emotion'].get)
                except Exception as e:
                    emotion = "neutral"
                finally:
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
