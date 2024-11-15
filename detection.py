import cv2
import mediapipe as mp

# Chemin de l'image
image_path = r"c:\Users\asus\Downloads\oval-visage-tremoille-paris-2048x1365.jpg"  # Remplace par le chemin de ton image
image = cv2.imread(image_path)

# Vérifier si l'image a bien été chargée
if image is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Initialiser MediaPipe FaceMesh
    mpfacemesh = mp.solutions.facemesh
    mpdraw = mp.solutions.drawing_utils
    facemesh = mpfacemesh.FaceMesh(max_num_faces=2)
    drawspec = mpdraw.DrawingSpec(thickness=2, circle_radius=2)

    # Convertir l'image en RGB (MediaPipe travaille en RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans l'image
    results = facemesh.process(rgb_image)

    # Vérifier s'il y a des visages détectés
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
           # Dessiner les points et les connexions du visage
            mpdraw.draw_landmarks(image, face_landmarks, mpfacemesh.FACEMESH_TESSELATION, drawspec, drawspec)

    # Afficher l'image avec les annotations
    cv2.imshow("Image avec Face Mesh", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
