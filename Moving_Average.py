import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
n = 200
t = np.arange(n)
freq = 0.03
signal_clean = np.sin(2 * np.pi * freq * t)
noise = np.random.normal(scale=0.5, size=n)
signal_noisy = signal_clean + noise


def moving_average_order(x, k):
    s = pd.Series(x)
    window = k + 1
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


ma_order_1 = moving_average_order(signal_noisy, k=1)  # window=2
ma_order_2 = moving_average_order(signal_noisy, k=2)  # window=3

df = pd.DataFrame(
    {
        "t": t,
        "clean_signal": np.round(signal_clean, 4),
        "noisy_signal": np.round(signal_noisy, 4),
        "MA_order_1 (window=2)": np.round(ma_order_1, 4),
        "MA_order_2 (window=3)": np.round(ma_order_2, 4),
    }
)

print(df.head(20).to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(t, signal_noisy, label="Noisy signal")
plt.plot(t, ma_order_1, label="MA order 1 (window=2)")
plt.plot(t, ma_order_2, label="MA order 2 (window=3)")
plt.plot(t, signal_clean, linestyle="--", label="Clean (true) signal")
plt.title("Noisy signal and moving-average filters (order 1 and 2)")
plt.xlabel("Time (t)")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
