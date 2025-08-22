# AnimeULike

This is the data collection, preprocessing and preparation code for the AnimeULike dataset as well as training a LLM-powered recommender system for anime recommendation.

## Navigation

    scripts
    ├── anime_feats_data     # Show-specific features (synopsis, reviews, popularity, etc.)
    ├── pref_data            # Processed preference matrix and latent factors from WMF
    ├── rating_data          # User supplied ratings
    ├── recs                 # Written inter-item recommendations
    ├── scrape               # Map and reduce pipeline for first scraping popular shows then discovering users from reviews
    ├── split                # Script used to split for training, val and test
    training                 # Code used for experiments

