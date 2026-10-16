import json
import sys
from pathlib import Path
if __name__=="__main__":
    report=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for key in ["test_gnn","test_training_median","test_role_fingerprint_random_forest"]:
        print(f"{key}: MAE={report[key]['mae_percentage_points']:.2f} percentage points")
