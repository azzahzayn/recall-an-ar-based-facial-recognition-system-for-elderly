# pyrefly: ignore [missing-import]
import sqlite3
import numpy as np
import os
import sys

def main():
    db_file = "live_demo_identities.db"
    if len(sys.argv) > 1:
        db_file = sys.argv[1]
        
    if not os.path.exists(db_file):
        print(f"Error: Database file '{db_file}' not found.")
        print("Please run the live demo and enroll yourself first.")
        return

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print(f"      SQLITE DATABASE INSPECTION: {db_file}")
    print("="*60)
    
    # 1. Inspect Identities Table
    print("\n[Identities Table]")
    cursor.execute("SELECT * FROM identities")
    identities = cursor.fetchall()
    
    if not identities:
        print("No identities registered in the database.")
    else:
        print(f"{'Identity ID':<20} | {'Name':<20} | {'Created At'}")
        print("-"*60)
        for identity_id, name, created_at in identities:
            print(f"{identity_id:<20} | {name:<20} | {created_at}")
            
    # 2. Inspect Embeddings Table
    print("\n[Embeddings Table]")
    cursor.execute("SELECT id, identity_id, model_version, pose_label, length(vector) FROM embeddings")
    embeddings = cursor.fetchall()
    
    if not embeddings:
        print("No embeddings registered in the database.")
    else:
        print(f"{'ID':<4} | {'Identity ID':<15} | {'Pose Label':<10} | {'Model':<12} | {'Vector Bytes'}")
        print("-"*60)
        for emb_id, identity_id, model_version, pose_label, vec_len in embeddings:
            print(f"{emb_id:<4} | {identity_id:<15} | {pose_label:<10} | {model_version:<12} | {vec_len} bytes")
            
    # Show vector shape example
    cursor.execute("SELECT vector FROM embeddings LIMIT 1")
    row = cursor.fetchone()
    if row:
        vec_bytes = row[0]
        # Convert bytes back to numpy array
        vector = np.frombuffer(vec_bytes, dtype=np.float32)
        print("\n[Sample Embedding Information]")
        print(f"- Vector Data Type: {vector.dtype}")
        print(f"- Vector Shape:     {vector.shape} (512-dimensional face vector)")
        print(f"- First 5 values:   {vector[:5]}")
        print(f"- L2 Norm length:   {np.linalg.norm(vector):.6f}")

    conn.close()
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
