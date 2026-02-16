import os

full_path = "/home/sulci/JS/A_joyCODE/DEV_LIN/AtrilBioscaCermoi_time1/Calc/mesh_flip_R00030CA_V1/flip-R001079LP_E1.mesh"

def debug_path(p):
    parts = p.split('/')
    current = "/"
    for part in parts:
        if not part: continue
        current = os.path.join(current, part)
        exists = os.path.exists(current)
        print(f"{' [OK] ' if exists else '[MISS]'} {current}")
        if not exists:
            # If it's missing, let's see what IS there
            parent = os.path.dirname(current)
            if os.path.exists(parent):
                print(f"  --> Directory exists, but '{part}' not found.")
                print(f"  --> Contents of {parent}: {os.listdir(parent)[:5]}...")
            break

debug_path(full_path)
