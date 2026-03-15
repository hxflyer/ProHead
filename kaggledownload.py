import kagglehub

# Download latest version
path = kagglehub.dataset_download("selfishgene/synthetic-faces-high-quality-sfhq-part-3")

print("Path to dataset files:", path)