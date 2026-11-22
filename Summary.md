# Face Recognition with Eigenfaces – Summary

## What was done
- **Data:** 380 face photos of 38 people (10 photos each). 85% (323 photos) were used for training and 15% (57 photos) for testing. Every person appears in both groups.
- **Training (PCA):** using the Chapter 4 code, the program builds an "average face" from the training photos and then finds the **eigenfaces**: the main patterns that make one face different from another. They are sorted from most important (largest eigenvalue) to least important.
- **Testing:** each test photo is turned into eigenface numbers and compared with every training photo. The training photo with the **smallest distance** is the match. If it shows the same person, that test counts as correct.

## Graph 1 – Number of eigenfaces vs. accuracy
- With only **1–5 eigenfaces**, accuracy is very low (under 11%). That is not enough information to tell 38 people apart.
- Accuracy climbs quickly: **56% with 10**, **75% with 15**, **90% with 30**.
- The **best result is 98.2% (56 of 57 correct) with 37 eigenfaces.**
- After that, adding more eigenfaces doesn't help. Accuracy stays around 90–95%, then slowly drops to **86% when all 323 are used**. The last eigenfaces mostly capture noise (lighting, tiny details), which confuses the matching a little.

**In short:** you need enough eigenfaces, but not too many. About 35–40 is the sweet spot.

## Graph 2 – Similarity threshold vs. accuracy
- Each match gets a **similarity score** from 0 to 1 (1 = identical). A match is only accepted if its score is at least the **threshold**.
- For every number of eigenfaces, accuracy stays the same as the threshold rises, **until it reaches a certain point**. After that the program starts rejecting correct matches and accuracy falls.
- That point is the **upper similarity threshold**:
  - **0.402 for the best case (37 eigenfaces)**, which keeps 98.2% accuracy
  - 0.622 for 15 eigenfaces
  - 0.170 when all 323 eigenfaces are used
- The more eigenfaces are used, the lower this threshold gets. More eigenfaces make every distance larger, so every similarity score comes out smaller.

**In short:** for the best setup, any threshold up to about 0.40 is safe. Set it higher and the program starts throwing away correct answers.

## Conclusion
Eigenfaces (PCA) work well for face recognition when the right number of eigenfaces is used. Too few can't tell people apart. About **37 eigenfaces gave the best accuracy (98.2%)**, and using all of them made results slightly worse. The similarity threshold shows how strict the program can be before it loses correct matches: about **0.40** for the best case, and lower as more eigenfaces are added. Because the test set is small (57 photos) and only one random split was used, these numbers are a good guide rather than exact values.
