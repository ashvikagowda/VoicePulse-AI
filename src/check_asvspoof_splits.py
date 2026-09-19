from datasets import get_dataset_split_names

dataset_name = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF"

print("Checking available dataset splits...")

splits = get_dataset_split_names(dataset_name)

print()
print("Available splits:")

for split in splits:
    print("-", split)