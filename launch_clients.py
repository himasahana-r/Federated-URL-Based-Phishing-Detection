import subprocess
import time

client_ids = [3, 4, 5, 6, 7, 8, 9, 10]

for cid in client_ids:
    subprocess.Popen(["start", "cmd", "/k", f"python client_clustered.py cluster_1 {cid}"], shell=True)
    time.sleep(0.5)  # optional delay between windows
