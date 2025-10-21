# File: src/01_get_data.py
import os
import requests

DATASOURCE = {
    "memoirs_of_grant": "https://www.gutenberg.org/ebooks/4367.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "sleepy_hollow": "https://www.gutenberg.org/ebooks/41.txt.utf-8",
    "origin_of_species": "https://www.gutenberg.org/ebooks/2009.txt.utf-8",
    "makers_of_many_things": "https://www.gutenberg.org/ebooks/28569.txt.utf-8",
    "common_sense": "https://www.gutenberg.org/ebooks/147.txt.utf-8",
    "economic_peace": "https://www.gutenberg.org/ebooks/15776.txt.utf-8",
    "the_great_war_3": "https://www.gutenberg.org/ebooks/29265.txt.utf-8",
    "elements_of_style": "https://www.gutenberg.org/ebooks/37134.txt.utf-8",
    "problem_of_philosophy": "https://www.gutenberg.org/ebooks/5827.txt.utf-8",
    "nights_in_london": "https://www.gutenberg.org/ebooks/23605.txt.utf-8",
}

def download_books(output_dir="data/raw"):
    # Create folder if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, url in DATASOURCE.items():
        path = os.path.join(output_dir, f"{filename}.txt")
        
        if not os.path.exists(path):
            print(f"📚 Downloading {filename}...")
            response = requests.get(url)
            with open(path, "wb") as f:
                f.write(response.content)
    
    print("All books downloaded!")


if __name__ == "__main__":
    download_books()
