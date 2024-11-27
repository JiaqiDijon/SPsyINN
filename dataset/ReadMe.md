## Data Directory Description

- `duolingo` directory: Contains the training, validation, and test data for the Duolingo, En2De, and En2Es datasets.
- `MaiMemo` directory: Contains the training, validation, and test data for the MaiMemo dataset.
- To facilitate the validation of symbolic regression, we have organized the corresponding data into CSV files stored in the `data(csv)` directory. You need to extract the compressed files in this directory, and the extracted data structure is consistent with the datasets mentioned above.
- The data includes:
  - $\delta_{1:6}$: Represents user memory features.
  - $Recall$: Represents memory states.
