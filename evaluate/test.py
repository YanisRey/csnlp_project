import urllib.request

url = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/MEN/MEN_dataset_natural_form_full"
save_path = "MEN_dataset_natural_form_full.txt"

urllib.request.urlretrieve(url, save_path)
print("✅ MEN dataset downloaded successfully.")
