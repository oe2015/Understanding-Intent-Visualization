# Data for SemEval 2023 task 3

[The website of the shared task](https://propaganda.math.unipd.it/semeval2023task3/), includes the submission instructions, updates on the competition and a live leaderboard.

__Table of contents:__

- [List of Versions](#list-of-versions)
- [Task Description](#task-description)
- [Data Format](#data-format)
- [Scorers](#scorers)
- [Baselines](#baseline)
- [Data Preparation for the Shared Task](#Data Preparation for the Shared Task)
- [Licensing](#licensing)
- [Citation](#citation)
  
  
* __v0.1 [2022/8/19]__ - data for subtasks 3 in English released.


## Task Description

**Subtask 1:** Given a news article, determine whether it is an opinion piece, aims at objective news reporting, or is a satire piece. This is a multi-class (single-label) task at article-level.

**Subtask 2:** Given a news article, identify the frames used in the article. This is a multi-label task at article-level.

**Subtask 3:** Given a news article, identify the persuasion techniques in each paragraph. This is a multi-label task at paragraph level.


## Data Format

The input documents are news articles. 
Each article appears in one .txt file. 
For the English subtasks, the title is on the first row, followed by an empty row. The content of the article starts from the third row.
The training and dev articles are in folder ```data/{en,it,fr,ge,ru,pl}/{train,dev}-articles-subtask{1,2,3}```.
For example the training articles for subtask 3 in English are in folder ```data/en/train-articles-subtask-3```. 

### File name conventions 

- The name of the files with the input articles have the following structure: article{ID}.txt, where {ID} is a unique numerical identifier (the starting digits of the ID identify the language). For example article111111112.txt
- The name of the files with the gold labels have the following structure: article{ID}-labels-subtask-{N}.txt, where {ID} is the numerical identifier of the corresponding article, and {N} is the index of the subtask. For example article111111112-labels-subtask-3.txt 


### Input data format

#### Subtask 1 and 2:

As said above, the input for the subtasks are in the folders ```data/{en,it,fr,ge,ru,pl}/{train,dev}-articles-subtask{1,2,3}```, each article is in its own txt file. 
In addition, for subtask 2, the list of frames is in the txt file ```scorers/frames_subtask2.txt```, one frame per line (these are the exact strings you are supposed to use when making predictions). This is the full list of frames, if for a language we considered a shorter list, then we a corresponding file  ```scorers/frames_subtask2_en.txt``` (here ```en``` stands for the language) is provided.  

#### Subtask 3:

In principle the input for subtask 3 could again be the articles in the folders ```data/{en,it,fr,ge,ru,pl}/{train,dev}-articles-subtask3```, each article is in its own txt file. 
However, participants are supposed to identify the techniques in each paragraph. In order to make sure the identification of the paragraphs by the participants is consistent with ours, we provide ```.template``` in the folders ```{train,dev}-labels-subtask-3/*.template```. There is one such file per article. Each row is made by three columns TAB separated: the first column is the ID of the article, the second column the index of the paragraph (starting from 1), the third column is the content of the paragraph. 
Note that , empty lines in the input files are not reported in the .template files and therefore predictions for those lines is not expected. 
For example the text in the file article1234.txt
```
The book is on the table

The table is under the book
```
would result in the following .template file
```
1234	1	The book is on the table
1234	3	The table is under the book
```


##### How to create paragraph level input and gold label files.

We need to use software from another repository and add it to this one as a submodule. 
Check if the folder ```scorers/scorer-subtask-3-spans/``` is not empty. 
<!--
If it is you need first to add the other repository as a submodule before acccessing it. Type the following from the starting folder of the repository: 
```
git submodule add https://gitlab.com/joedsm/propaganda-techniques-scorer.git scorers/scorer-subtask-3-spans
```
If the repo has been added as a submodule but it does not appear in the destination folder, 
-->
If it is, go to the parent folder of the destination folder, i.e. ```scorers```, and type
```
git submodule init
git submodule update
```
To update the submodule, go to its folder and type ```git pull```. 
If changes have been made, the status of the submodule could be detached; do a ```git checkout main``` to put it back on track. 

to create paragraph level annotations for a dataset:
1) go to the scorers folder: ```cd scorers```
2) configure and run the script ```./extract_paragraph_level_annotations.sh```. 
Examples of invokations of the script are in the script itself. Note that it creates a log file listing all the annotations overflowing in the next paragraph (see the output for the file name). The script assumes the naming conventions specified in this document. 
The script  extract_paragraph_level_annotations.sh invokes ```scorer-subtask-3-spans/extract_paragraph_level_annotations.py```. 
The function that does most of the computation in extract_paragraph_level_annotations.py is ```extract_paragraph_level_annotations()```, which is in ```src/article_annotations.py``` (~row 320). 


### Prediction Files Format

For all subtasks, a prediction file, for example for the development set, must be one single txt file.

#### Subtask 1

The format of a tab-separated line of the gold label and the submission files for subtask 1 is:
```
 article_id     label
```	    
where article_id is the numeric id in the name of the input article file (e.g. the id of file article123456.txt is 123456), label is one the strings representing the three categories: reporting, opinion, satire. This is an example of a section of the gold file for the articles with ids 123456 - 123460:
```
123456    opinion
123457    opinion
123458    satire
123459    reporting
123460    satire
```						

#### Subtask 2

The format of a tab-separated line of the gold label and the submission files for subtask 2 is:
```
 article_id     label_1,label_2,...,label_N
```	    
where article_id is the numeric id in the name of the input article file (e.g. the id of file article123456.txt is 123456), label_x is one the strings representing the frames that are present in the articles: Economic,Capacity_and_resources,..., Other. This is an example of a section of the gold file for the articles with ids 123456 - 123460:
```
  123456    Crime_and_punishment,Policy_prescription_and_evaluation    
  123457    Legality,Constitutionality_and_jurisprudence,Security_and_defense
  123458    Health_and_safety,Quality_of_life,Cultural_identity
  123469    Public_opinion
```		

#### Subtask 3

```
111111111	1	
111111111	3	Doubt
111111111	5	Appeal_to_Authority
111111111	7	
111111111	9	
111111111	11	
111111111	13	Repetition
111111111	15	
111111111	17	Appeal_to_Fear-Prejudice
111111111	19	Appeal_to_Fear-Prejudice
111111111	21	
111111111	23	Appeal_to_Authority
111111111	25	Appeal_to_Fear-Prejudice
111111111	27	
111111111	29	
111111112	1	
111111112	3	
111111112	5	Slogans
111111112	7	
111111112	9	
111111112	11	False_Dilemma-No_Choice
111111112	13	
111111112	14	Slogans
111111112	15	Loaded_Language
```		

## Scorer and Official Evaluation Metrics

The scorer for the subtasks is located in the [scorers](scorers) folder.
The scorer will report official evaluation metric and other metrics of a prediction file.

You can install all prerequisites through,
> pip install -r requirements.txt

### Subtask 1
The **official evaluation metric** for the task is **macro-F1**. However, the scorer also reports micro-F1. 

To launch it, run the following command:
```python
cd scorers;
python3 scorer-subtask-1.py --gold_file_path <path_to_gold_labels> --pred_file_path <path_to_your_results_file> --classes_file_path=<path_to_techniques_categories_for_task>
```
For example:
```python
cd scorers;
python3 scorer-subtask-1.py --gold_file_path golds-subtask1.txt --pred_file_path predictions-subtask1.txt
```

### Subtask 2

```python
cd scorers;
python3 scorer-subtask-2.py --gold_file_path <path_to_gold_labels> --pred_file_path <path_to_your_results_file> --frame_file_path <path_to_frame_list_file>
```
For example:
```python
cd scorers;
python3 scorer-subtask-2.py --gold_file_path golds-subtask2.txt --pred_file_path predictions-subtask2.txt --frame_file_path frames_subtask2.txt
```

### Subtask 3

TBD

The scorer for subtask 3 is coded in another repository. In order to add it to the project type the following commands:
```
git submodule init
git submodule update
cd scorers/scorer-subtask-3 
git pull
```
Subtask 3 is a multi-label sequence tagging task. We modify the standard micro-averaged F1 to account for partial matching between the spans. 
In addition, an F1 value is computed for each persuasion technique.
```
cd scorer/task2; 
python3 task-2-semeval21_scorer.py -s prediction_file -r gold_labels_file -p ../../techniques_list_task1-2.txt 
```
To access the command line help of the scorer type
```
python3 task-2-semeval21_scorer.py -h
```
Note that the option -d prints additional debugging information.


## Baselines

TBD

### Task 1

 * Random baseline
 ```
cd baselines; python3 baseline_task1_random.py
 ```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of 0.04494.

### Task 2

The baseline for task 2 simply creates random spans and technique names for the development set. No learning is performed. 
Run as
```
cd baselines; python3 baseline_task2.py
```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of 0.00699.
If you score the baseline on the training set (uncomment lines 5-6 in baseline_task2.py), you should get a F1 score of 0.038112
```
python3 task-2-semeval21_scorer.py -s ../../baselines/baseline-output-task2-train.txt -r ../../data/training_set_task2.txt -p ../../techniques_list_task1-2.txt 
...
F1=0.00699
...
```

### Task 3

 * Random baseline
 ```
cd baselines; python3 baseline_task3_random.py
 ```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of 0.03376.


## Data Preparation for the Shared Task

the script collect_data_for_website.sh is supposed to create the compressed files containing the corpus, the scorers and the baselines for the shared task.
These files will then be placed in the ```data``` folder of the shared task website.
If you change the corpus, the scorers or the baselines, do update the ```collect_data_for_website.sh```, rerun it and upload the data to the website.


## Licensing

These datasets are free for research purposes, but they cannot be used for commercial purposes without the organisers authorisation.


## Citation

Besides the SemEval publication that describes the shared task (please cite it if you use the data in your work)

```bibtex
@InProceedings{SemEval2023:task3,
  author    = {TBA},
  title     = {TBA},
  booktitle = {TBA},
  series    = {TBA},
  year      = {TBA},
  url = {TBA},
}
```

the data for the task include the following previous datasets:

* PTC Dataset

```bibtex
@InProceedings{EMNLP19DaSanMartino,
	author = {Da San Martino, Giovanni and
	Yu, Seunghak and
	Barr\'{o}n-Cede\~no, Alberto and
	Petrov, Rostislav and
	Nakov, Preslav},
	title = {Fine-Grained Analysis of Propaganda in News Articles},
	booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019},
	series = {EMNLP-IJCNLP 2019},
	year = {2019},
	address = {Hong Kong, China},
	month = {November},
}
```



