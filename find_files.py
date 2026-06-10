# ==========================================
# DEEP SEARCH - Find everything
# Save as: deep_search.py
# ==========================================

import os

print("=" * 60)
print("SEARCHING FOR ALL FILES IN PROJECT")
print("=" * 60)

current_dir = os.getcwd()
print(f"Current directory: {current_dir}\n")

# List everything in current directory
print("Contents of current directory:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"  📁 {item}/")
    else:
        print(f"  📄 {item}")

print("\n" + "=" * 60)
print("SEARCHING FOR SPECIFIC FILES")
print("=" * 60)

# Search for common file types
found_files = {
    'Python files (.py)': [],
    'CSV files (.csv)': [],
    'Audio files (.au)': [],
    'PNG files (.png)': []
}

for root, dirs, files in os.walk('.'):
    # Skip virtual environment and hidden folders
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '.venv']

    for file in files:
        full_path = os.path.join(root, file)

        if file.endswith('.py'):
            found_files['Python files (.py)'].append(full_path)
        elif file.endswith('.csv'):
            found_files['CSV files (.csv)'].append(full_path)
        elif file.endswith('.au'):
            found_files['Audio files (.au)'].append(full_path)
        elif file.endswith('.png'):
            found_files['PNG files (.png)'].append(full_path)

# Display results
for file_type, file_list in found_files.items():
    print(f"\n{file_type}: {len(file_list)} found")
    if file_list:
        # Show first 10 files
        for f in file_list[:10]:
            print(f"  {f}")
        if len(file_list) > 10:
            print(f"  ... and {len(file_list) - 10} more")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

# Give recommendations based on what was found
if found_files['CSV files (.csv)']:
    csv_path = found_files['CSV files (.csv)'][0]
    print(f"✓ Found CSV files! Use this path: {csv_path}")
elif found_files['Audio files (.au)']:
    print(f"✓ Found {len(found_files['Audio files (.au)'])} audio files")
    print("  You need to run these scripts first:")
    print("  1. python spectrogramExtraction.py  (creates PNG spectrograms)")
    print("  2. python spectogramDataset.py      (creates CSV files)")
elif found_files['PNG files (.png)']:
    print(f"✓ Found {len(found_files['PNG files (.png)'])} PNG files")
    print("  You need to run:")
    print("  1. python spectogramDataset.py      (creates CSV files)")
else:
    print("⚠ No data files found!")
    print("  Expected file structure:")
    print("    Data/train/[genre]/*.au          (audio files)")
    print("    spectrograms/train/[genre]/*.png (spectrograms)")
    print("    csv/train/*.csv                  (CSV files)")

# Check for your Python scripts
print("\n" + "=" * 60)
print("YOUR PYTHON SCRIPTS")
print("=" * 60)

expected_scripts = [
    'spectrogramExtraction.py',
    'spectogramDataset.py',
    'spectogramCNN.py',
    'spectogramTransferLearning.py',
    'structuredMLP.py'
]

for script in expected_scripts:
    if any(script in f for f in found_files['Python files (.py)']):
        print(f"  ✓ {script}")
    else:
        print(f"  ✗ {script} (not found)")