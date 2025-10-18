from icrawler.builtin import BingImageCrawler
import os

# Target folder: where new images will go
save_dir = os.path.expanduser("~/Desktop/dataset_processed/train/infection")

# Ensure directory exists
os.makedirs(save_dir, exist_ok=True)

# Initialize Bing Image Crawler
crawler = BingImageCrawler(storage={"root_dir": save_dir})

# Realistic infection wound search query (avoids charts, illustrations, and fake data)
crawler.crawl(
    keyword=(
        "real human skin infection wound close-up OR infected skin lesion OR cellulitis skin infection "
        "OR pus wound OR abscess on skin OR red swollen infected wound OR inflamed skin infection photo OR "
        "bacterial skin infection close-up OR infected cut wound photo OR cellulitis arm photo "
        "-diagram -chart -illustration -drawing -vector -cartoon -animation -infographic -stock -poster -educational"
    ),
    max_num=2000,          # tries to download up to 2000 candidates
    min_size=(256, 256),   # ensures usable resolution
    file_idx_offset="auto"
)

print(f"✅ Finished downloading realistic infection images into {save_dir}")
