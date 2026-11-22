# Task 5 – Conclusion

*Results from `eigenfaces_assignment.m` in MATLAB with `SEED = 42`, which splits the data into 323 training and 57 test photos (85% / 15%) of 38 people. A different seed gives a different random split, so the exact numbers change a little, but the pattern stays the same.*

## What Task 3 showed: how many eigenfaces do we need?

- **Too few eigenfaces don't work.** With 1–5 eigenfaces the program recognizes at most about 10% of the test faces. A handful of "face ingredients" is not enough to tell 38 people apart.
- **Accuracy rises quickly as more are added.** 10 eigenfaces give 56%, 15 give 75%, 25 give 84% and 30 give 90%.
- **The best result is 98.2% (56 of 57 faces) with 37 eigenfaces.** The eigenfaces with the largest eigenvalues hold the most important differences between faces, so the first few dozen do almost all the work.
- **Adding many more makes it slightly worse.** From about 40 to 150 eigenfaces, accuracy stays around 91–95%. With all 323 it drops to 86%. The last eigenfaces have tiny eigenvalues and mostly describe noise, lighting and small details, which add confusion instead of useful information.
- **The 15 eigenfaces used in the original code are too few for this data.** They give only 75%. About 35–100 eigenfaces works much better, with the sweet spot near 37.

## What Task 4 showed: the similarity threshold

- The similarity score of the best match is between 0 and 1, where 1 means identical. A match is only accepted when its similarity is at least the threshold.
- **For every number of eigenfaces, the accuracy stays the same up to a certain threshold, then falls.** Past that point the program starts throwing away correct matches because it is being too strict.
- **The highest safe threshold (the "upper similarity threshold") gets lower as more eigenfaces are used.** It is about 0.67 with 10 eigenfaces, 0.62 with 15, **0.40 with 37 (the best case)**, 0.28 with 100 and 0.17 with all 323. More eigenfaces make every distance bigger, so every similarity score smaller, and the threshold has to come down to match.
- **For the best case (37 eigenfaces)**, any threshold up to 0.40 keeps the full 98.2%. Above that, accuracy drops quickly: to about 80% near 0.5 and to about a third near 0.6.
- **In this data, the threshold rarely catches mistakes.** Wrong matches often look just as similar as right ones, so a stricter threshold mostly removes correct matches. For example, at 37 eigenfaces the one wrong match is not rejected. The threshold is useful as a limit on how strict you can be, not as a way to raise accuracy.
- *Odd detail:* with very few eigenfaces (1–8) the upper threshold is 1.0. The original code stores values as `uint8` (whole numbers from 0 to 255), which squashes some features to exactly the same value. A few test photos then get a "perfect" similarity of 1.0 with their match.

## In simple terms

Eigenfaces work well for recognizing faces, but only with the right number of them. Too few (under about 20) and the program can't tell people apart. Around 37 it recognizes 98% of the test faces. Using every eigenface makes it slightly worse, because the extra ones mostly add noise. The similarity threshold shows how strict the program can be before it starts rejecting correct answers: about 0.40 for the best setting, and lower when more eigenfaces are used. Because the test set is small (57 photos, so each one is worth about 1.75%) and only one random split was used, these numbers are a good guide rather than exact values.
