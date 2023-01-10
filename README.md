# NLP-Notebook

- `T5/` uses T5 with Hugging Face to fine-tune a model that maps from English to reverse-English
    - `python run_T5.py` runs the model
    - `predictions_{epoch}.csv` shows the sample output at epoch `{epoch}`
    - `output.log` stores the logging result
    - [T5 output](T5/predictions_1.csv)
- `SRU/` implements SRU to do sequence prediction, i.e. `{x1, x2, x3} = {1, 2, 3}, x4 = ?`
    - `python run_SRU.py` runs the model (can be done with cpu for short sequences)
    - note that the model takes an input sequence of length 3 and predict the next value in the sequence, so the following sample output show only the ground truth next value and the predicted ones (in the same picture/file)
    - `predictions_{epoch}.png` shows the sample plot at epoch `{epoch}`
    - `predictions_{epoch}.csv` shows the sample output at epoch `{epoch}`
    - `output.log` stores the logging result
    - [SRU output](SRU/predictions_20.png)