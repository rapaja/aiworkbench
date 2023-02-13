# IDEAS

## Lesion Detection Uncertainty Estimation

Use multiple bounding boxes (masks) for each lesion. 
For example, it seems reasonable to use 3: **certainty region**,
**base region**, and **suspicious region**. The **base** is the
mask annotated by a doctor, the two other masks are automatically
extracted from the base mask by shrinking and expanding the
annotation by a given percentage (say 25 percent), maybe more.
For small lesions, the **certainty region** may be equal to the
**base region**, while the suspicious area could be selected
slightly larger.

During training, the three regions are handled as if they
belong to different classes. However, false positives and false
negatives are not penalized in the same manner for all of them.
For the **certain** class, false positives are considered more
severe compared to false negatives, for the **base** class the
two errors are considered equally severe, while for **suspicious**
class false negative is considered more severe than false positive.

In this way, the model will learn to be conservative and cautious
in the same time, however, different behavior will be evident in
different classes.