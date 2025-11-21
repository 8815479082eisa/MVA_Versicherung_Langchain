"""
Wartet bis genug Audit-Log-Einträge vorhanden sind und führt dann die Evaluation aus
"""
import time
import subprocess
import os

TARGET_ENTRIES = 110  # 60 alte + 50 neue
CHECK_INTERVAL = 30  # Sekunden
MAX_WAIT_TIME = 3600  # 1 Stunde

def count_audit_entries():
    """Zählt die Anzahl der Einträge in audit.log"""
    if not os.path.exists("audit.log"):
        return 0
    with open("audit.log", "r", encoding="utf-8") as f:
        return len([line for line in f if line.strip()])

def main():
    print("="*80)
    print("Waiting for query generation to complete...")
    print(f"Target: {TARGET_ENTRIES} entries in audit.log")
    print("="*80)
    
    start_time = time.time()
    last_count = count_audit_entries()
    
    while True:
        current_count = count_audit_entries()
        elapsed = time.time() - start_time
        
        if current_count != last_count:
            print(f"[{elapsed:.0f}s] Current entries: {current_count}/{TARGET_ENTRIES}")
            last_count = current_count
        
        if current_count >= TARGET_ENTRIES:
            print(f"\n✓ Target reached! ({current_count} entries)")
            print("Starting evaluation...\n")
            break
        
        if elapsed > MAX_WAIT_TIME:
            print(f"\n⚠ Max wait time reached ({MAX_WAIT_TIME}s)")
            print(f"Current entries: {current_count}")
            print("Starting evaluation with available entries...\n")
            break
        
        time.sleep(CHECK_INTERVAL)
    
    # Führe Evaluation aus
    print("="*80)
    print("Running extended evaluation...")
    print("="*80)
    subprocess.run(["python", "evaluate_extended.py"])

if __name__ == "__main__":
    main()

