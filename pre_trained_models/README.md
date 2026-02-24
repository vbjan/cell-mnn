# Explanation for Pre-trained Model Weights
This folder contains the model weights from training on the **less preprocessed** Embryoid data. For evaluating against the TRRUST database this is required as this Embryoid dataset still contains the gene symbols.

If you intend to train your own Cell-MNN model and want to evaluate it on the TRRUST database, place it into its own folder under `pre_trained_models/MY_MODEL`. You can place multiple models to build an ensemble as in `pre_trained_models/cellmnn`.