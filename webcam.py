from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Captura de vídeo da câmera padrão (geralmente a câmera integrada do notebook)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

while True:
    # Captura frame por frame
    ret, frame = cap.read()

    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Exibe o frame capturado
    cv2.imshow('Camera', annotated_frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()