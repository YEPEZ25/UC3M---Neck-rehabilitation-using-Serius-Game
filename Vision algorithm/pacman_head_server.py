import cv2
import mediapipe as mp
import numpy as np
import socket

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dimensiones del cuadrado de referencia
REFERENCE_BOX_WIDTH = 100
REFERENCE_BOX_HEIGHT = 100

# Establecer el servidor socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)
print("Esperando conexión...")

# Aceptar la conexión entrante
client_socket, address = server_socket.accept()
print(f"Conectado a {address}")

def get_direction(nose_position, frame_width, frame_height):
    nose_x, nose_y = nose_position
    direction_vertical = "Frente"
    direction_horizontal = "Frente"
    direction_movement = ""

    # Determinar si el movimiento es principalmente vertical o horizontal
    if abs(nose_y - frame_height // 2) > REFERENCE_BOX_HEIGHT // 2:
        direction_vertical = "Vertical"
        if nose_y < frame_height // 2:
            direction_movement = "Arriba"
        elif nose_y > frame_height // 2:
            direction_movement = "Abajo"
    elif abs(nose_x - frame_width // 2) > REFERENCE_BOX_WIDTH // 2:
        direction_horizontal = "Horizontal"
        if nose_x < frame_width // 2:
            direction_movement = "Derecha"
        elif nose_x > frame_width // 2:
            direction_movement = "Izquierda"

    return direction_vertical, direction_horizontal, direction_movement

def draw_reference_box(frame, frame_width, frame_height):
    # Calcular las coordenadas del cuadrado de referencia en el centro de la pantalla
    ref_box_x = (frame_width - REFERENCE_BOX_WIDTH) // 2
    ref_box_y = (frame_height - REFERENCE_BOX_HEIGHT) // 2

    # Dibujar el cuadrado de referencia
    cv2.rectangle(frame, (ref_box_x, ref_box_y), (ref_box_x + REFERENCE_BOX_WIDTH, ref_box_y + REFERENCE_BOX_HEIGHT), (0, 255, 0), 2)

# Configurar el socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_x = int(face_landmarks.landmark[1].x * frame_width)
                nose_y = int(face_landmarks.landmark[1].y * frame_height)
                nose_position = (nose_x, nose_y)

                # Utilizar la lógica para determinar si el movimiento es principalmente vertical o horizontal
                direction_vertical, direction_horizontal, direction_movement = get_direction(nose_position, frame_width, frame_height)

                # Dibujar el cuadrado de referencia en el centro de la pantalla
                draw_reference_box(frame, frame_width, frame_height)

                if direction_vertical != "Frente":
                    cv2.putText(frame, f"Direccion Vertical: {direction_movement}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif direction_horizontal != "Frente":
                    cv2.putText(frame, f"Direccion Horizontal: {direction_movement}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    direction_movement = "Frente"
                # Enviar los datos de movimiento por el socket
                movement_data = f"{direction_movement}"
                print(movement_data)
                client_socket.send(movement_data.encode())


                cv2.circle(frame, nose_position, 5, (0, 255, 0), -1)  # Dibujar un círculo en la nariz

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
