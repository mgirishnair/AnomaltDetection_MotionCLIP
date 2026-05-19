import numpy as np

y = np.load("y.npy")
unique, counts = np.unique(y, return_counts=True)

print("min label:", unique.min())
print("max label:", unique.max())
print("num classes:", len(unique))
print("first labels:", unique[:20])
print("last labels:", unique[-20:])

for cls, cnt in zip(unique, counts):
    print(cls, cnt)
