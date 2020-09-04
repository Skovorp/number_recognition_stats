# Description
Code generates different stats from file with prediction results (results_secure.csv) and file with actual car numbers (gt_secure.csv)

# Input files
##### gt.csv
- File with actual car numbers
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


