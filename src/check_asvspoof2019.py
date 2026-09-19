from collections import Counter

from datasets import load_dataset


print("Loading ASVspoof2019 LA validation split...")

ds = load_dataset(
    "Bisher/ASVspoof_2019_LA",
    split="validation"
)

print()
print("Dataset loaded successfully!")

print("Number of samples:", len(ds))

print()
print("Columns:")
print(ds.column_names)

print()
print("Label distribution:")

labels = [
    sample["key"]
    for sample in ds
]

print(Counter(labels))

print()
print("First 5 samples:")

for i in range(5):

    sample = ds[i]

    print()
    print("Sample", i + 1)
    print("Speaker:", sample["speaker_id"])
    print("File:", sample["audio_file_name"])
    print("Key:", sample["key"])
    print("System:", sample["system_id"])