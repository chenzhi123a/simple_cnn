from data_io import load_image


image = load_image(
    "raw_dataset.npz",
    index=0,
)

print(image)
print(image.shape)
print(image.dtype)