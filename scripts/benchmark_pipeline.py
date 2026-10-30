import os
import time
import numpy as np
import cv2
import logging
from common.schemas import FramePacket, CameraIntrinsics, Landmarks5pt
from common.logging import setup_logger
from modules.capture.config import CameraConfig
from modules.capture.mock_camera import MockCamera
from modules.detection.detector import FaceDetector
from modules.detection.alignment import align_face
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
    logger.info("Initializing Modular Rebuild components...")
    
    # 1. Initialize core hardware/camera config
    cam_cfg = CameraConfig(width=640, height=480, fps=30)
    camera = MockCamera(cam_cfg)
    
    # 2. Initialize algorithms and models
    detector = FaceDetector()
    
    # Use mock embedder for offline safety if ONNX is not downloaded yet
    embed_cfg = EmbedderConfig(use_mock=True)
    embedder = FaceEmbedder(embed_cfg)
    
    # SQLite temp database config
    db_file = "test_identities_demo.db"
    if os.path.exists(db_file):
        os.remove(db_file) # Clean start
    db_cfg = DatabaseConfig(db_path=db_file, similarity_threshold=0.6)
    repository = IdentityRepository(db_cfg)
    
    reconstructor = FaceReconstructor()
    projector = ColorProjector()
    pose_gen = PoseGenerator(target_size=(112, 112))
    
    # End-to-end pipeline orchestrator
    pipeline = RecognitionPipeline(detector, embedder, repository)
    overlay_renderer = OverlayRenderer(640, 480)

    # ============================================================
    # STEP 1: ENROLLMENT - 3D PARAMETRIC MULTI-POSE AUGMENTATION
    # ============================================================
    logger.info("--- STEP 1: Starting Enrollment via 3D Multi-Pose Augmentation ---")
    camera.start()
    packet = camera.read()
    camera.stop()
    
    # Run detector
    det_res = detector.detect(packet.rgb)
    if not det_res.boxes:
        logger.error("No face detected for enrollment!")
        return
        
    bbox = det_res.boxes[0]
    landmarks = det_res.landmarks[0]
    
    logger.info("Face detected. Reconstructing 3D Face Mesh...")
    # Reconstruct 3D shape & pose
    face_matrix = packet.intrinsics.to_matrix()
    mesh, rvec, tvec = reconstructor.reconstruct(landmarks, face_matrix)
    
    logger.info("Projecting face texture colors onto 3D vertices...")
    # Texture mapping
    textured_mesh = projector.project_texture(mesh, packet.rgb, packet.intrinsics)
    
    logger.info("Synthesizing 7 virtual gallery poses (-45 to +45 degrees yaw)...")
    # Multi-view pose rendering
    poses = pose_gen.generate_pose_gallery(textured_mesh, packet.intrinsics)
    
    # Generate embeddings for all synthesized poses
    pose_embeddings = []
    pose_labels = []
    for img, yaw in poses:
        emb = embedder.embed(img)
        pose_embeddings.append(emb)
        pose_labels.append(f"yaw_{int(yaw)}")
        
    logger.info(f"Generated {len(pose_embeddings)} synthetic embeddings for gallery.")
    
    # Save target subject to database
    repository.enroll_identity("id_john_doe", "Dr. John Doe", pose_embeddings, pose_labels)
    
    # ============================================================
    # STEP 2: REAL-TIME RECOGNITION SIMULATION & BENCHMARKING
    # ============================================================
    logger.info("--- STEP 2: Starting Real-Time Recognition Simulation ---")
    camera.start()
    
    frame_times = []
    success_matches = 0
    total_frames = 20
    
    logger.info(f"Processing {total_frames} live video frames...")
    
    for i in range(total_frames):
        start_t = time.time()
        
        # Capture frame
        live_packet = camera.read()
        
        # Execute pipeline query (optimized: skips 3D reconstruction at query-time)
        results = pipeline.process_frame(live_packet)
        
        # Render visual overlay frame
        annotated_frame = overlay_renderer.render(live_packet.rgb, results)
        
        duration = (time.time() - start_t) * 1000.0 # to ms
        frame_times.append(duration)
        
        # Print status of matches
        match_str = "None"
        if results:
            match = results[0]
            if match.identity:
                match_str = f"{match.identity} (conf: {match.confidence:.2f})"
                if match.identity == "Dr. John Doe":
                    success_matches += 1
            else:
                match_str = "Unknown"
                
        logger.info(f"Frame {i+1:02d}/{total_frames} | Latency: {duration:6.2f}ms | Identified: {match_str}")
        time.sleep(0.03) # Simulate 30 FPS delay
        
    camera.stop()
    repository.close()
    
    # ============================================================
    # STEP 3: BENCHMARK SUMMARY REPORT
    # ============================================================
    avg_latency = sum(frame_times) / len(frame_times)
    fps = 1000.0 / avg_latency
    success_rate = (success_matches / total_frames) * 100.0
    
    print("\n" + "="*50)
    print("           PIPELINE BENCHMARK REPORT")
    print("="*50)
    print(f"Total Frames Processed:   {total_frames}")
    print(f"Average Frame Latency:   {avg_latency:.2f} ms")
    print(f"Pipeline Running Speed:  {fps:.1f} FPS")
    print(f"Identity Match Success:  {success_matches}/{total_frames} ({success_rate:.1f}%)")
    print(f"Database Storage File:   {db_file}")
    print("="*50 + "\n")
    
    # Clean up DB file
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except OSError:
            pass

if __name__ == "__main__":
    main()
