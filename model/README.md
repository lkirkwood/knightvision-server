
# Knight Vision - YOLOv8 Training Assets

This folder contains all necessary files to fully reproduce the training process of the Knight Vision chess piece detector.

## Contents

- **Training Notebook (`Knight Vision - YOLOv8 Model Training Notebook.ipynb`)**  
  Main notebook used to train the YOLOv8 model on the chess piece dataset. Follows the exact pipeline and parameters used in the final Knight Vision model.

- **Dataset (`yolo_data/`)**  
  Includes the `train`, `val`, and `test` splits of the chess dataset, already augmented and corrected. This dataset is ready to use without modification.

- **Dataset Configuration (`data.yaml`)**  
  Defines the dataset splits and class names for YOLOv8. Used directly during training.

- **Requirements (`training_requirements.txt`)**  
  Python dependencies required to run the training notebook. Can be installed via `pip install -r training_requirements.txt`.

- **Test Image (`test_board.jpg`)**  
  Optional test image to quickly verify the trained model after completion.

## Usage

1. Install requirements:

```bash
pip install -r training_requirements.txt
```

2. Open and run `Knight Vision - YOLOv8 Model Training Notebook.ipynb` to begin training.  
   - Training logs and checkpoints will be saved automatically.
   - Parameters (epochs, batch size, etc.) are preset as per the final model's training run.

3. After training, use the test image provided or your own images to verify predictions using the trained model.

## Notes

- This is **exactly** the version used for the Knight Vision YOLOv8 model in the final system.
- No extra setup needed, the dataset and configuration are already provided.
- You can change training parameters or continue training from the checkpoint if needed.

---

**Knight Vision Project - Deep Learning & CNN (42028)**  
Adam Cameron (14327074), Shifali Lakshmanan, Linus Kirkwood
