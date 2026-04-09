import json

# Paths to your two files
file1 = "MotionCLIP/data/babel_v1.0_release/train.json"
file2 = "MotionCLIP/data/babel_v1.0_release/extra_train.json"


files = [file1, file2]
merged = {}

for path in files:
    with open(path, "r") as f:
        data = json.load(f)
        for k, v in data.items():
            # Use babel_sid or key string as unique identifier
            sid = str(v.get("babel_sid", k))
            merged[sid] = v  # overwrites duplicates safely

# Save deduplicated merged file
with open("MotionCLIP/data/babel_v1.0_release/train_merged.json", "w") as f:
    json.dump(merged, f)
