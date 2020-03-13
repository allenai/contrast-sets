### What's in here
- `perspectrum_contrast_sets.csv`: contains the constrast sets we use for evaluation.

- `perspectrum_model.py`: RoBERTa model
- `run_evaluation.py`: evaluation script for computing model performance and consistency on contrast sets.


### How to run evaluation
- Download the pretrained RoBERTa stance classification model from [here](https://drive.google.com/drive/folders/1etmXIYisMU5B9D6UoKaYK6bCgsD3cXlL?usp=sharing)
- Install required packages in requirement.txt
- Run the following command in the `perspectrum/` directory. Remove the `--gpu` flag if you prefer to run it on CPU. 
```
python run_evaluation.py --model_dir <directory of the model files you downloaded> --gpu 
```
 
