#!/usr/bin/env python3

import sys
sys.path.append('..')

import deepfakeecg
import matplotlib.pyplot
import numpy

# Generate a single ECG sample
ecg_data = deepfakeecg.generate_as_numpy()

# Plot the first lead (Lead I)
matplotlib.pyplot.figure(figsize=(15, 3))
matplotlib.pyplot.plot(ecg_data[:, 0], label="Lead I")
matplotlib.pyplot.plot(ecg_data[:, 1], label="Lead II")
matplotlib.pyplot.plot(ecg_data[:, 2], label="V1")
matplotlib.pyplot.plot(ecg_data[:, 3], label="V2")
matplotlib.pyplot.plot(ecg_data[:, 4], label="V3")
matplotlib.pyplot.plot(ecg_data[:, 5], label="V4")
matplotlib.pyplot.plot(ecg_data[:, 6], label="V5")
matplotlib.pyplot.plot(ecg_data[:, 7], label="V6")
matplotlib.pyplot.legend()
matplotlib.pyplot.title("Generated ECG - Lead I")
matplotlib.pyplot.xlabel("Sample")
matplotlib.pyplot.ylabel("Amplitude (μV)")
matplotlib.pyplot.grid(True)

# Print shape and basic stats
print(f"ECG data shape: {ecg_data.shape}")
print(f"Value range: [{numpy.min(ecg_data)}, {numpy.max(ecg_data)}]")
print("\nLead order: [I, II, V1, V2, V3, V4, V5, V6]")

matplotlib.pyplot.show()
