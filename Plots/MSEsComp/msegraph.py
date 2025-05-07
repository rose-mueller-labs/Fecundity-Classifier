import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing your txt files
data_dir = 'data'  # Change to your folder

# Prepare storage for all data
all_data = {}

# Loop through all txt files
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.txt'):
        with open(os.path.join(data_dir, fname)) as f:
            lines = f.readlines()
        # Find the start of the HumanSum section
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == 'HumanSum':
                start_idx = i + 1
                break
        if start_idx is not None:
            # Read until a blank line or non-numeric start
            data = []
            for line in lines[start_idx:]:
                if not line.strip():
                    break
                match = re.match(r'^\s*(\d+)\s+([\d\.]+)', line)
                if match:
                    key, value = int(match.group(1)), float(match.group(2)) ** 0.5
                    data.append((key, value))
            # Store as a dictionary
            file_label = os.path.splitext(fname)[0]
            all_data[file_label] = dict(data)

# Get all unique HumanSum keys
all_humansum = sorted(set(k for d in all_data.values() for k in d.keys()))

# Build DataFrame for plotting
df = pd.DataFrame(index=all_humansum)
for label, d in all_data.items():
    df[label] = pd.Series(d)

# Plot
# fig, ax = plt.subplots(figsize=(14, 7))
fig, ax = plt.subplots(figsize=(16, 8))
df.plot(kind='bar', ax=ax)
ax.set_xlabel('HumanSum')
ax.set_ylabel('Value')
ax.set_title('HumanSum Values from All TXT Files')
plt.ylim(0,10)
plt.legend(title='File')
plt.tight_layout()
plt.savefig("msesv7")