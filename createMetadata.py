import json
import os

# Path to the data folder
data_folder = "frontend/data"

# # Optional: Manually add categories for files (can leave empty if not needed)
# categories = {
#     "amazon.txt": "E-commerce",
#     "bmw.txt": "Automotive",
#     "costco.txt": "Retail",
#     "linkedin.txt": "Social Media",
#     # Add more as needed
# }

# Initialize metadata list
metadata_list = []

# Loop through all .txt files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        # Create metadata entry
        metadata = {
            "title": filename.replace(".txt", "").replace("_", " ").title(),
            "filename": filename,
            # "category": categories.get(filename, "")  # Get category if defined
        }
        metadata_list.append(metadata)

# Save metadata to a JSON file
output_path = os.path.join(data_folder, "metadata.json")
with open(output_path, "w") as json_file:
    json.dump(metadata_list, json_file, indent=4)

print(f"Metadata saved to {output_path}")
