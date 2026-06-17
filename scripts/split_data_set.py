from pathlib import Path
import random
import shutil

src_dir = Path("/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_train_voxel")
out_dir = Path("/mnt/efs/dl_jrc/student_data/S-XZ/train_janelia/omezarr_split")

train_dir = out_dir / "train"
val_dir = out_dir / "val"

# Remove old split folder if it exists
if out_dir.exists():
    shutil.rmtree(out_dir)

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Find all .ome.zarr folders
samples = sorted(src_dir.glob("*.ome.zarr"))

arrth_samples = [s for s in samples if s.name.startswith("arrth")]
norm_samples = [s for s in samples if s.name.startswith("norm")]

print("Arrth samples:")
for s in arrth_samples:
    print(" ", s.name)

print("\nNorm samples:")
for s in norm_samples:
    print(" ", s.name)

assert len(arrth_samples) == 2, f"Expected 2 arrth samples, found {len(arrth_samples)}"
assert len(norm_samples) == 6, f"Expected 6 norm samples, found {len(norm_samples)}"

# Random but reproducible split
random.seed(42)
random.shuffle(arrth_samples)
random.shuffle(norm_samples)

# Train: 1 arrth + 5 norm = 6 total
# Val:   1 arrth + 1 norm = 2 total
train_samples = [arrth_samples[0]] + norm_samples[:5]
val_samples = [arrth_samples[1]] + norm_samples[5:]

print("\nTrain samples:")
for s in train_samples:
    print(" ", s.name)

print("\nValidation samples:")
for s in val_samples:
    print(" ", s.name)

# Create symlinks instead of copying large data
for s in train_samples:
    link_path = train_dir / s.name
    link_path.symlink_to(s, target_is_directory=True)

for s in val_samples:
    link_path = val_dir / s.name
    link_path.symlink_to(s, target_is_directory=True)

print("\nDone.")
print("Train dir:", train_dir)
print("Val dir:", val_dir)