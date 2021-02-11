# mycelia-sdk
Mycelia SDKs

# Examples
Instanciating your base class
```python
mycelia = Mycelia(AUTH_KEY)
```

## Setting up your databases

Aplication using the model NLP FastText
```python
### fasttext implementation
# generate a random name for identification of the base, can be a user input
# save this if you wish to work in the same database later
name = mycelia.generate_name(20, prefix='sdk_', suffix='_fasttext')

# Data insertion, data can be a list of texts, pandas Series or DataFrame.
# if data is a list, then ids will be set with range(len(data_list))
# if data is a pandas type, then the ids will be the index values, index must not contain duplicated values
mycelia.insert_data(name, data)

# Train the unsupervised FastText model
mycelia.setup_database(name, db_type='FastText')
mycelia.wait_setup(10)  # wait for the train to finish
```

Aplication using the model NLP BERT
```python
### bert implementation
# Same initial steps
name = mycelia.generate_name(20, prefix='sdk_', suffix='_text')
mycelia.insert_data(name, data, batch_size=1024)

# this time we choose db_type="Text", applying the pre-trained BERT model
mycelia.setup_database(name, db_type='Text')
mycelia.wait_setup(10)
```

## Similarity
After you're done with setting up your database, you can find similarity:

- Using the indexes of the inputed data
```python
# Find the 100 most similar values for every inputed value
results = mycelia.similar_list(name, data.index, top_k=100, batch_size=1024)
```

- Using new data to be processed
```python
# Find the 100 most similar values for every new_data
results = mycelia.similar_data(name, new_data, top_k=100, batch_size=1024)
```
The output will be a list of dictionaries with ("query_id") the id of the value you want to find similars and ("results") a list with `top_k` dictionaries with the "id" and the "distance" between "query_id" and "id".
```
[{'query_id': 0,
  'results': [{'id': 0, 'distance': 0.0},
   {'id': 3836, 'distance': 2.298321008682251},
   {'id': 9193, 'distance': 2.545339584350586},
   {'id': 832, 'distance': 2.5819168090820312},
   {'id': 6162, 'distance': 2.638622283935547},
   ...]},
   ....,
{'query_id': 9,
  'results': [{'id': 9, 'distance': 0.0},
   {'id': 54, 'distance': 5.262974262237549},
   {'id': 101, 'distance': 5.634262561798096},
   ...},
   ....]
```

# Removing data

After you're done with the model setup, you can delete your raw data
```python
# Delete the raw data inputed as it won't be needed anymore
mycelia.delete_raw_data(name)
```

If you want to keep the environment clean
``` python
mycelia.delete_database(name)
```
