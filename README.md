# Movie Recommendation Systems

In this project for my DSC 201 (Applied Data Science) class, I created and evaluated three movie recommenders built in Python on the HetRec 2011 MovieLens 2k dataset: a content-based model, a collaborative filtering model using SVD, and a hybrid of the two. All three are evaluated on held-out data and compared against a popularity baseline.

## Results

| Model | Hit Rate @ 10 |
|---|---|
| Collaborative (SVD) | **75.3%** |
| Popularity Baseline | 66.3% |
| Hybrid | 53.3% |
| Content-Based | 25.0% |

- **SVD rating prediction:** test RMSE of **0.841** vs. **0.924** for a user-average baseline (about 9% better), with `k = 20` latent features chosen on a separate validation set
- SVD was the only model to beat the popularity baseline
- The hybrid more than doubled the content-based hit rate, which shows that re-ranking similar movies with SVD predictions helps a lot

**Hit rate @ 10:** for each user, the model recommends 10 movies they haven't rated in training. It counts as a hit if at least one of those is a movie they rated 4+ stars in the hidden test set. Measured on 300 randomly sampled users.

## The Models

**Content-Based:** combines each movie's genres, director, top 5 billed actors, and tags into one string, vectorizes it with TF-IDF, and recommends the most similar movies using cosine similarity.

**Collaborative (SVD):** builds a user-movie rating matrix (2,113 users x 4,732 movies, about 92% empty), centers ratings by each user's average, and uses truncated SVD to predict the missing ratings.

**Hybrid:** takes the 100 movies most similar to a given movie (content-based), then ranks them by the user's predicted rating (SVD), so two users asking about the same movie get different lists.

## Data Cleaning

A big part of the project was cleaning the dataset, which has a few problems of its own:
- 775 duplicate movie titles caused by IMDb linking errors (for example, 6 separate entries for Hamlet (1996)), resolved by keeping the entry with the most ratings
- Inconsistent director IDs, so cleaned director names were used instead
- Punctuation and formatting issues in actor names, director names, and tags
- Movies with fewer than 22 ratings (the median) were left out of the SVD model so it has enough data to learn from

## Dataset

This project uses the **HetRec 2011 MovieLens 2k** dataset from GroupLens (855,598 ratings from 2,113 users on 10,197 movies). The data is not included in this repo. It can be downloaded here: https://grouplens.org/datasets/hetrec-2011/



## Limitations

- Shared actors sometimes outweigh genre in the content model (for example, Toy Story recommending unrelated Tim Allen movies)
- Some linking errors in the dataset weren't caught. For example, movieID 47 is labeled as Seven Samurai (1954), but its genres match Se7en (1995), which is what that ID actually is in MovieLens
- The content-based and hybrid models were only given each user's single highest rated movie in the hit rate test, which puts them at a disadvantage compared to SVD
- Hit rate was measured on 300 users rather than all 2,042 users with hidden favorites
