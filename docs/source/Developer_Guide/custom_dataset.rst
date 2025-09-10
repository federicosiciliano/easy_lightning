Custom Dataset
==============
If you are planning to insert a new dataset, follow the next steps:

1. The `download_dataset` method is responsible of the download of the dataset from the web. What it does is a download via either `curl` or `wget`. What you have to do is to insert a new `if` with your `dataset_name` and then insert how to download it. Be careful, by default the data are in the folder `data\raw`. This means that if you want do download the dataset `abc` with name of the raw file is `abc_raw.csv` you should have a directory like `data\raw\abc\abc_raw.csv`.

2. You have to add in the dictionary `dataset_files` of the `map_dataset_name()` method the name of your data and the name of the raw file. For example, if the name of your dataset is `abc` and the name of the raw file is `abc_raw.csv` you should insert in the dictionary `abc : abc_raw.csv`.

3. You should implement how to preprocess it. You can take inspiration from all the ways to preprocess the available datasets. The output of this preprocessing should be `.csv` like file with the columns `'uid', 'sid', 'rating', 'timestamp'`. This is probably the most difficult step. However, lots of dataset nowadays are already available online in `.csv` format. 

4. Lastly, you should define how to load the preprocessed data. If you followed our suggestions in the previous step this will be no more than a single line of code where you load the `.csv` file using `df = pd.read_csv(preprocessed_file_name.csv)`