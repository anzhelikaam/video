import cv2

# Завантаження каскаду
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Відкриття відеофайлу
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("❌ Не вдалося відкрити відео.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Відтінки сірого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Розпізнавання облич
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 🔽 Зменшення кадру до ширини 320 пікселів
    width = 320
    height = int(frame.shape[0] * (width / frame.shape[1]))
    resized_frame = cv2.resize(frame, (width, height))

    # Відображення
    cv2.imshow('Face Detection Video', resized_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
