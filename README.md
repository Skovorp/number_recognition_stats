# Description
This code generates analytics of car number recognition algorithm, that processes security camera photos. This is important to understand limitations and problematic cases of algorythm and estimate performance in defferent usecases.

# Input files
##### gt.csv
- File with actual car numbers. Sample file has real car numbers distorted to protect privacy.
- Doesn't have header
- Every line represents a sigle recognition attempt
- First column consists of filenames (or any other unuque strings) for number
- Second column consists of correct numbers

##### results.csv
- File with prediction system results
- Doesn't have header
- Every line represents a sigle recognition attempt
- First column consists of filenames (or any other unuque strings) for number. Should match with other info in gt
- Second column consists of prediction results
- ? means that character is not recognised

# Output files
##### ids.xlsx
For each pair of number and prediction, an optimal sequence of character inserts, deletes, substitutions (i,d,s) is calculated that will alter prediction to match a real number. Ids sequence is calculated as described in Levenstein distance calculation algorithm. You can set costs of ids manually to favor certain moves for others. Costs of operations with '?' can be set differently to favour having '?' instead of wrong char.
File has table with ids counts per each character (i,d) or pair (s) occured during processing of all numbers. 
##### ids_normal.xlsx
Writes table as in previous, but each number is normalised by occurance frequency. In s table each number is devided only once by true char frequency.
##### Nones_stats.txt
Writes stats of number detection, gt and results merge. Number is detected if at least one character is not '?'.
##### Type_stats.txt
There are 2 types of car numbers in Russia. In this file, there is info about freqency of type recogmision.
##### All_mistakes.txt
All numbers with mistakes. Each mistake comes with Levenstein distance calculated with provided weights.
##### ids_log.txt
Log of all ids operations made on all numbers.
##### Merged_data.csv
Table with merged gt and results data.

