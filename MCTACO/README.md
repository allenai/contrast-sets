We selected and perturbated 100 questions from the original [MCTACO](https://github.com/CogComp/MCTACO) 

# File description:
- `original.tsv`: The original questions and their answers and labels. Each question has a unique index.
- `changed.tsv`: Perturbated questions and their ansnwers and labels. Each question has a unique index. The question with a matching index in `original.tsv` is the original unchanged question as published in the MCTACO paper.
- `original_roberta_out.txt` contains the predictions on the original questions. Please refer to the contrast set paper for model details.
- `changed_roberta_out.txt` contains the predictions on the perturbed questions of the same model. 