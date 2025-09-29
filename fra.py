import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Generate FRA Data (Healthy vs Faulty)
# -------------------------------

# Frequency range (Hz)
freq = np.linspace(20, 20000, 500)  # 20 Hz to 20 kHz, 500 points
# Healthy transformer response (smooth sine wave-like)
healthy_response = np.sin(0.0005 * np.pi * freq) + np.random.normal(0, 0.05, len(freq))

# Faulty transformer response (distorted curve)
faulty_response = np.sin(0.0005 * np.pi * freq) + 0.5*np.sin(0.001 * np.pi * freq) + np.random.normal(0, 0.1, len(freq))

# Put into DataFrame
df = pd.DataFrame({
    "Frequency_Hz": freq,
    "Healthy_Response": healthy_response,
    "Faulty_Response": faulty_response
})

# Save as CSV (simulate FRA input file)
df.to_csv("fra_simulated_data.csv", index=False)

# Plot
plt.figure(figsize=(10,5))
plt.plot(freq, healthy_response, label="Healthy Transformer", color="green")
plt.plot(freq, faulty_response, label="Faulty Transformer", color="red")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Response (p.u.)")
plt.title("Simulated FRA Signatures")
plt.legend()
plt.grid(True)
plt.show()
