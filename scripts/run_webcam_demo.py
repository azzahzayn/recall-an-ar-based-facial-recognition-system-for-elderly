# pyrefly: ignore [missing-import]
import os
import sys
import time
import cv2
import numpy as np
import logging
from common.logging import setup_logger
from modules.capture.config import CameraConfig
from modules.capture.webcam_camera import WebcamCamera
from modules.capture.mock_camera import MockCamera
from modules.detection.detector import FaceDetector
from modules.embedding.config import EmbedderConfig
from modules.embedding.embedder import FaceEmbedder
from modules.database.config import DatabaseConfig
from modules.database.repository import IdentityRepository
from modules.reconstruction.model_wrapper import FaceReconstructor
from modules.color_projection.projector import ColorProjector
from modules.pose_generation.renderer import PoseGenerator
from modules.recognition.pipeline import RecognitionPipeline
from modules.ar_output.overlay_renderer import OverlayRenderer

# Set up logging
logger = setup_logger("face_ar_system", level=logging.INFO)

def main():
    logger.info("Initializing Live Webcam Face Recognition & 3D Pose Augmentation Demo...")
    
    # 1. Initialize camera
    cam_cfg = CameraConfig(width=640, height=480, fps=30)
    use_mock = False
    
    try:
        camera = WebcamCamera(cam_cfg)
        camera.start()
        logger.info("Webcam started successfully.")
    except Exception as e:
        logger.warning(f"Failed to start physical webcam: {e}. Falling back to Mock Camera (Simulation mode).")
        camera = MockCamera(cam_cfg)
        camera.start()
        use_mock = True

    # 2. Initialize algorithms and models
    detector = FaceDetector()
    
    # Use mock embedder if camera is mock, otherwise use real ArcFace embedder for accurate face verification
    embed_cfg = EmbedderConfig(use_mock=use_mock)
    embedder = FaceEmbedder(embed_cfg)
    
    # Database initialization: Persist database across runs
    db_file = "live_demo_identities.db"
    db_cfg = DatabaseConfig(db_path=db_file, similarity_threshold=0.55)
    repository = IdentityRepository(db_cfg)
    
    # Check database for model version mismatches
    current_model_version = embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8)).model_version
    all_identities = repository.get_all_identities()
    mismatched_identities = []
    
    for identity in all_identities:
        for emb in identity.embeddings:
            if emb.model_version != current_model_version:
                mismatched_identities.append((identity.name, emb.model_version))
                break
                
    if mismatched_identities:
        logger.warning(
            f"The database contains identities enrolled with a different model version than the current "
            f"embedder model ('{current_model_version}'). These identities cannot be recognized and should be re-enrolled:"
        )
        for name, db_ver in mismatched_identities:
            logger.warning(f"  - '{name}' (enrolled with model version: '{db_ver}')")
        logger.warning("To re-enroll an identity, press 'E' when their face is visible in the frame, or clear the database by pressing 'R'.")
    
    reconstructor = FaceReconstructor()
    projector = ColorProjector()
    pose_gen = PoseGenerator(target_size=(112, 112))
    
    pipeline = RecognitionPipeline(detector, embedder, repository)
    overlay_renderer = OverlayRenderer(640, 480)
    
    window_name = "Face AR Recognition & 3D Augmentation Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "="*70)
    print("                LIVE MULTI-USER WEBCAM INTERACTIVE DEMO")
    print("="*70)
    print(" Instructions:")
    print("   - Align a face in front of the camera.")
    print("   - If the face is unrecognized, it will be labeled as 'Unknown'.")
    print("   - Press 'E' (or 'e') on your keyboard to enroll the Unknown face.")
    print("   - Go to your terminal window to enter the name when prompted.")
    print("   - Once enrolled, the system dynamically tracks and recognizes you.")
    print("   - Press 'R' (or 'r') to reset/clear database.")
    print("   - Press 'Q' (or 'q') to exit the demo.")
    print("="*70 + "\n")

    try:
        while True:
            # Capture frame
            try:
                packet = camera.read()
            except Exception as read_err:
                logger.error(f"Failed to read from camera: {read_err}")
                break
                
            rgb_frame = packet.rgb
            
            # Live Recognition Path: always run recognition on the frame
            results = pipeline.process_frame(packet)
            
            # Render HUD style brackets & labels over frame
            annotated_rgb = overlay_renderer.render(rgb_frame, results)
            display_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            
            # Determine if there's any unknown face in the frame
            unknown_detected = False
            for res in results:
                if res.identity is None:
                    unknown_detected = True
                    break
            
            # Render instruction overlay bar on screen
            cv2.rectangle(display_frame, (10, 10), (630, 45), (0, 0, 0), -1)
            if unknown_detected:
                status_text = "Unknown face detected! Press 'E' to Enroll."
                text_color = (0, 165, 255) # Orange/Coral warning
            else:
                status_text = "Tracking... Press 'E' to enroll a new face, 'R' to reset."
                text_color = (0, 240, 200) # Neon cyan
                
            cv2.putText(display_frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
            # Show display window
            cv2.imshow(window_name, display_frame)
            
            # Read keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                logger.info("Exiting demo.")
                break
                
            elif key == ord('e') or key == ord('E'):
                # Check detection to make sure there is a face present
                det_res = detector.detect(rgb_frame)
                if not det_res.boxes:
                    logger.warning("No face detected in the frame! Reposition yourself and try again.")
                    continue
                
                # Draw freezing message in the window
                enroll_msg = display_frame.copy()
                cv2.rectangle(enroll_msg, (20, 150), (620, 330), (0, 0, 0), -1)
                cv2.rectangle(enroll_msg, (20, 150), (620, 330), (0, 240, 200), 2)
                cv2.putText(enroll_msg, "ENROLLMENT IN PROGRESS", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(enroll_msg, "Please go to your terminal window", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(enroll_msg, "and type the name for this identity.", (50, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window_name, enroll_msg)
                cv2.waitKey(1)
                cv2.waitKey(1)
                
                # Prompt user in terminal
                print("\n" + "="*50)
                print("             NEW IDENTITY ENROLLMENT")
                print("="*50)
                try:
                    name = input("Enter name for this person: ").strip()
                except (KeyboardInterrupt, SystemExit):
                    break
                    
                if not name:
                    name = f"User_{int(time.time())}"
                identity_id = f"id_{int(time.time())}"
                
                print(f"Registering '{name}'... Fitting 3D Face Mesh & generating views...")
                
                bbox = det_res.boxes[0]
                landmarks = det_res.landmarks[0]
                face_matrix = packet.intrinsics.to_matrix()
                
                try:
                    mesh, rvec, tvec = reconstructor.reconstruct(landmarks, face_matrix)
                    textured_mesh = projector.project_texture(mesh, rgb_frame, packet.intrinsics)
                    
                    # Synthesize 7 angles
                    poses = pose_gen.generate_pose_gallery(textured_mesh, packet.intrinsics)
                    
                    pose_embeddings = []
                    pose_labels = []
                    for img, yaw in poses:
                        emb = embedder.embed(img)
                        pose_embeddings.append(emb)
                        pose_labels.append(f"yaw_{int(yaw)}")
                    
                    # Save high-fidelity live crop and mirrored crop for accurate identification
                    from modules.detection.alignment import align_face
                    aligned_face = align_face(rgb_frame, landmarks)
                    frontal_emb = embedder.embed(aligned_face)
                    pose_embeddings.append(frontal_emb)
                    pose_labels.append("frontal_highres")
                    
                    mirrored_face = cv2.flip(aligned_face, 1)
                    mirrored_emb = embedder.embed(mirrored_face)
                    pose_embeddings.append(mirrored_emb)
                    pose_labels.append("mirrored_highres")
                        
                    # Save identity to database
                    repository.enroll_identity(identity_id, name, pose_embeddings, pose_labels)
                    logger.info(f"Successfully registered '{name}' with 3D pose augmentation and high-res profiles!")
                    
                    # Reset tracker to clear old 'Unknown' label history so recognition updates immediately
                    pipeline.tracker.tracks.clear()
                except Exception as enroll_err:
                    logger.error(f"Enrollment failed: {enroll_err}")
                    
            elif key == ord('r') or key == ord('R'):
                # Reset database
                logger.info("Clearing all registered identities...")
                repository.close()
                if os.path.exists(db_file):
                    try:
                        os.remove(db_file)
                    except OSError:
                        pass
                repository = IdentityRepository(db_cfg)
                pipeline = RecognitionPipeline(detector, embedder, repository)
                logger.info("Database reset complete. All identities cleared.")
                
    finally:
        camera.stop()
        repository.close()
        cv2.destroyAllWindows()
        logger.info(f"Database persisted at: {db_file}")

if __name__ == "__main__":
    main()
