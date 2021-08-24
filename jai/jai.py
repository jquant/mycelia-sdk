import secrets
import json
import pandas as pd
import numpy as np
import requests
import time

from io import BytesIO
from .base import BaseJai
from .processing import (process_similar, process_resolution, process_predict)
from .functions.utils_funcs import data2json, pbar_steps
from .functions.classes import PossibleDtypes, Mode
from .functions import exceptions
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from pandas.api.types import is_integer_dtype
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import trange, tqdm

__all__ = ["Jai"]


class Jai(BaseJai):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """
    def __init__(self,
                 auth_key: str = None,
                 url: str = None,
                 var_env: str = "JAI_SECRET"):
        """
        Inicialize the Jai class.

        An authorization key is needed to use the Mycelia API.

        Parameters
        ----------
        auth_key : str
            Authorization key for the use of the API.
        url : str, optional
            Param used for development purposes. `Default is None`.

        Returns
        -------
            None

        """
        super(Jai, self).__init__(auth_key, url, var_env)

    @property
    def names(self):
        """
        Retrieves databases already created for the provided Auth Key.

        Return
        ------
            List with the sorted names of the databases created so far.

        Example
        -------
        >>> j.names
        ['jai_database', 'jai_selfsupervised', 'jai_supervised']

        """
        return sorted(self._info(mode="names"))

    @property
    def info(self):
        """
        Get name and type of each database in your environment.

        Return
        ------
        pandas.DataFrame
            Pandas dataframe with name, type, creation date and parent
            databases of each database in your environment.

        Example
        -------
        >>> j.info
                                db_name           db_type
        0                  jai_database              Text
        1            jai_selfsupervised    SelfSupervised
        2                jai_supervised        Supervised
        """
        df = pd.DataFrame(self._info()).rename(
            columns={
                "db_name": "name",
                "db_type": "type",
                "db_version": "last modified",
                "db_parents": "dependencies",
            })
        return df.sort_values(by="name")

    @property
    def status(self, max_tries=5, patience=25):
        """
        Get the status of your JAI environment when training.

        Return
        ------
        response : dict
            A `JSON` file with the current status of the training tasks.

        Example
        -------
        >>> j.status
        {
            "Task": "Training",
            "Status": "Completed",
            "Description": "Training of database YOUR_DATABASE has ended."
        }
        """
        tries = 0
        while tries < max_tries:
            try:
                return self._status()
            except:
                time.sleep(patience // max_tries)
                tries += 1
        return self._status()

    def generate_name(self,
                      length: int = 8,
                      prefix: str = "",
                      suffix: str = ""):
        """

        Generate a random string. You can pass a prefix and/or suffix. In this case,
        the generated string will be a concatenation of `prefix + random + suffix`.

        Args
        ----
        length : int
            Length for the desired string. `Default is 8`.
        prefix : str
            Prefix of your string. `Default is empty`.
        suffix  : str
            Suffix of your string. `Default is empty`.

        Returns
        -------
        str
            A random string.

        Example
        ----------
        >>> j.generate_name()
        13636a8b
        >>> j.generate_name(length=16, prefix="company")
        companyb8bbd445d

        """
        len_prefix = len(prefix)
        len_suffix = len(suffix)

        if length <= len_prefix + len_suffix:
            raise ValueError(
                f"length ({length}) should be larger than {len_prefix+len_suffix} for prefix and suffix inputed."
            )
        if length >= 32:
            raise ValueError(f"length ({length}) should be smaller than 32.")

        length -= len_prefix + len_suffix
        code = secrets.token_hex(length)[:length].lower()
        name = str(prefix) + str(code) + str(suffix)
        names = self.names

        while name in names:
            code = secrets.token_hex(length)[:length].lower()
            name = str(prefix) + str(code) + str(suffix)

        return name

    @staticmethod
    def get_auth_key(email: str,
                     firstName: str,
                     lastName: str,
                     company: str = ""):
        """
        Request an auth key to use JAI-SDK with.

        Args
        ----------
        `email`: str
            A valid email address where the auth key will be sent to.
        `firstName`: str
            User's first name.
        `lastName`: str
            User's last name.
        `company`: str
            User's company.

        Return
        ----------
        `response`: dict
            A Response object with whether or not the auth key was created.
        """
        url = "https://mycelia.azure-api.net/clone"
        body = {
            "email": email,
            "firstName": firstName,
            "lastName": lastName,
            "company": company
        }
        response = requests.put(url + "/auth", json=body)
        return response

    def download_vectors(self, name: str):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        vector : np.array
            Numpy array with all vectors.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> vectors = j.download_vectors(name=name)
        >>> print(vectors)
        [[ 0.03121682  0.2101511  -0.48933393 ...  0.05550333  0.21190546  0.19986008]
        [-0.03121682 -0.21015109  0.48933393 ...  0.2267401   0.11074653  0.15064166]
        ...
        [-0.03121682 -0.2101511   0.4893339  ...  0.00758727  0.15916921  0.1226602 ]]
        """
        r = requests.get(self._download_vectors(name))
        return np.load(BytesIO(r.content))

    def filters(self, name):
        """
        Gets the valid values of filters.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : list of strings
            List of valid filter values.
        """
        return self._filters(name)

    def similar(self,
                name: str,
                data,
                top_k: int = 5,
                filters=None,
                batch_size: int = 16384):
        """
        Query a database in search for the `top_k` most similar entries for each
        input data passed as argument.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : list, np.ndarray, pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        top_k : int
            Number of k similar items that we want to return. `Default is 5`.
        batch_size : int
            Size of batches to send the data. `Default is 16384`.

        Return
        ------
        results : list of dicts
            A list with a dictionary for each input value identified with
            'query_id' and 'result' which is a list with 'top_k' most similar
            items dictionaries, each dictionary has the 'id' from the database
            previously setup and 'distance' in between the correspondent 'id'
            and 'query_id'.

        Example
        -------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> TOP_K = 3
        >>> j = Jai(AUTH_KEY)
        >>> df_index_distance = j.similar(name, DATA_ITEM, TOP_K)
        >>> print(pd.DataFrame(df_index_distance['similarity']))
           id  distance
        10007       0.0
        45568    6995.6
         8382    7293.2
        """
        dtype = self.get_dtype(name)

        if isinstance(data, list):
            data = np.array(data)

        is_id = is_integer_dtype(data)

        results = []
        for i in trange(0, len(data), batch_size, desc="Similar"):
            if is_id:
                if isinstance(data, pd.Series):
                    _batch = data.iloc[i:i + batch_size].tolist()
                elif isinstance(data, pd.Index):
                    _batch = data[i:i + batch_size].tolist()
                else:
                    _batch = data[i:i + batch_size].tolist()
                res = self._similar_id(name,
                                       _batch,
                                       top_k=top_k,
                                       filters=filters)
            else:
                if isinstance(data, (pd.Series, pd.DataFrame)):
                    _batch = data.iloc[i:i + batch_size]
                else:
                    _batch = data[i:i + batch_size]
                res = self._similar_json(name,
                                         data2json(_batch,
                                                   dtype=dtype,
                                                   predict=True),
                                         top_k=top_k,
                                         filters=filters)
            results.extend(res["similarity"])
        return results

    def predict(self,
                name: str,
                data,
                predict_proba: bool = False,
                as_frame: bool = False,
                batch_size: int = 16384):
        """
        Predict the output of new data for a given database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : pd.Series or pd.DataFrame
            Data to be queried for similar inputs in your database.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction is
            it's a classification. `Default is False`.
        batch_size : int
            Size of batches to send the data. `Default is 16384`.

        Return
        ------
        results : list of dicts
            List of dictionaries with 'id' of the inputed data and 'predict'
            as predictions for the data passed as input.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> DATA_ITEM = # data in the format of the database
        >>> j = Jai(AUTH_KEY)
        >>> preds = j.predict(name, DATA_ITEM)
        >>> print(preds)
        [{"id":0, "predict": "class1"}, {"id":1, "predict": "class0"}]

        >>> preds = j.predict(name, DATA_ITEM, predict_proba=True)
        >>> print(preds)
        [{"id": 0 , "predict"; {"class0": 0.1, "class1": 0.6, "class2": 0.3}}]
        """
        dtype = self.get_dtype(name)
        if dtype != "Supervised":
            raise ValueError("predict is only available to dtype Supervised.")
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError(
                f"data must be a pandas Series or DataFrame. (data type `{data.__class__.__name__}`)"
            )

        # insert data
        from datetime import datetime
        annotations = {
            "prefix": "predict_",
            "suffix": "_" + datetime.now().strftime("%d%m%y%H%M%S%f")
        }
        db_type = PossibleDtypes.supervised
        self._insert_data(data=data,
                          name=name,
                          batch_size=batch_size,
                          db_type=db_type,
                          annotations=annotations)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        results = []
        for i in trange(0, len(data), batch_size, desc="Predict"):
            _batch = data.iloc[i:i + batch_size]
            res = self._predict(name,
                                data2json(_batch, dtype=dtype, predict=True),
                                predict_proba=predict_proba,
                                annotations=annotations)
            results.extend(res)

        return process_predict(results) if as_frame else results

    def ids(self, name: str, mode: Mode = "simple"):
        """
        Get id information of a given database.

        Args
        mode : str, optional

        Return
        -------
        response: list
            List with the actual ids (mode: 'complete') or a summary of ids
            ('simple'/'summarized') of the given database.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> ids = j.ids(name)
        >>> print(ids)
        ['891 items from 0 to 890']
        """
        return self._ids(name, mode)

    def is_valid(self, name: str):
        """
        Check if a given name is a valid database name (i.e., if it is in your environment).

        Args
        ----
        `name`: str
            String with the name of a database in your JAI environment.

        Return
        ------
        response: bool
            True if name is in your environment. False, otherwise.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> check_valid = j.is_valid(name)
        >>> print(check_valid)
        True
        """
        return self._is_valid(name)["value"]

    def setup(self,
              name: str,
              data,
              db_type: str,
              batch_size: int = 16384,
              frequency_seconds: int = 1,
              filter_name: str = None,
              verbose: int = 1,
              **kwargs):
        """
        Insert data and train model. This is JAI's crème de la crème.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Data to be inserted and used for training.
        db_type : str
            Database type {Supervised, SelfSupervised, Text, FastText, TextEdit, Image}
        batch_size : int
            Size of batch to insert the data.`Default is 16384 (2**14)`.
        frequency_seconds : int
            Time in between each check of status. `Default is 10`.
        **kwargs
            Parameters that should be passed as a dictionary in compliance with the
            API methods. In other words, every kwarg argument should be passed as if
            it were in the body of a POST method. **To check all possible kwargs in
            Jai.setup method, you can check the** `Setup kwargs`_ **section**.

        Return
        ------
        insert_response : dict
            Dictionary of responses for each data insertion.
        setup_response : dict
            Setup response telling if the model started training.

        Example
        -------
        >>> name = 'chosen_name'
        >>> data = # data in pandas.DataFrame format
        >>> j = Jai(AUTH_KEY)
        >>> _, setup_response = j.setup(name=name, data=data, db_type="Supervised", label={"task": "metric_classification", "label_name": "my_label"})
        >>> print(setup_response)
        {
            "Task": "Training",
            "Status": "Started",
            "Description": "Training of database chosen_name has started."
        }
        """
        overwrite = kwargs.get("overwrite", False)
        if overwrite and name in self.names:
            self.delete_database(name)
        elif name in self.names:
            raise KeyError(
                f"Database '{name}' already exists in your environment. Set overwrite=True to overwrite it."
            )
        else:
            # delete data reamains
            self.delete_raw_data(name)

        # make sure our data has the correct type and is free of NAs
        data = self._check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        insert_responses = self._insert_data(data=data,
                                             name=name,
                                             batch_size=batch_size,
                                             filter_name=filter_name,
                                             db_type=db_type)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        # train model
        body = self._check_kwargs(db_type=db_type, **kwargs)
        setup_response = self._setup(name, body, overwrite)
        if "kwargs" in setup_response:
            print("\nRecognized setup args:")
            for key, value in setup_response["kwargs"].items():
                kwarg_input = kwargs.get(key, None)
                if isinstance(kwarg_input, dict):
                    value = json.loads(value)
                    intersection = kwarg_input.keys() & value.keys()
                    m = max([len(s) for s in intersection] + [0])
                    value = "".join(
                        [f"\n  * {k:{m}s}: {value[k]}" for k in intersection])
                print(f"- {key}: {value}")

        if frequency_seconds >= 1:
            if db_type in ["Supervised", "SelfSupervised"]:
                print(
                    f"\nTraining might finish early due to early stopping criteria."
                )
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

        if db_type in [
                PossibleDtypes.selfsupervised, PossibleDtypes.supervised
        ]:
            self.report(name, verbose)

        return insert_responses, setup_response

    def fit(self, *args, **kwargs):
        """
        Another name for setup.
        """
        return self.setup(*args, **kwargs)

    def add_data(self,
                 name: str,
                 data,
                 batch_size: int = 16384,
                 frequency_seconds: int = 1,
                 filter_name: str = None):
        """
        Insert raw data and extract their latent representation.

        This method should be used when we already setup up a database using `setup()`
        and want to create the vector representations of new data
        using the model we already trained for the given database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data : pandas.DataFrame or pandas.Series
            Data to be inserted and used for training.
        batch_size : int
            Size of batch to send the data. `Default is 16384`.
        frequency_seconds : int
            Time in between each check of status. `Default is 10`.

        Return
        -------
        insert_responses: dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        # delete data reamains
        self.delete_raw_data(name)

        # get the db_type
        db_type = self.get_dtype(name)

        # make sure our data has the correct type and is free of NAs
        data = self._check_dtype_and_clean(data=data, db_type=db_type)

        # insert data
        insert_responses = self._insert_data(data=data,
                                             name=name,
                                             batch_size=batch_size,
                                             db_type=db_type,
                                             filter_name=filter_name,
                                             predict=True)

        # check if we inserted everything we were supposed to
        self._check_ids_consistency(name=name, data=data)

        # add data per se
        add_data_response = self._append(name=name)

        if frequency_seconds >= 1:
            self.wait_setup(name=name, frequency_seconds=frequency_seconds)

        return insert_responses, add_data_response

    def append(self,
               name: str,
               data,
               batch_size: int = 16384,
               frequency_seconds: int = 1,
               filter_name: str = None):
        """
        Another name for add_data
        """
        return self.add_data(name=name,
                             data=data,
                             batch_size=batch_size,
                             frequency_seconds=frequency_seconds,
                             filter_name=filter_name)

    def _insert_data(self,
                     data,
                     name,
                     db_type,
                     batch_size,
                     filter_name: str = None,
                     predict: bool = False,
                     annotations=None):
        """
        Insert raw data for training. This is a protected method.

        Args
        ----------
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, SelSupervised, Text...)
        batch_size : int
            Size of batch to send the data.
        predict : bool
            Allows table type data to have only one column for predictions,
            if False, then tables must have at least 2 columns. `Default is False`.

        Return
        ------
        insert_responses : dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        insert_responses = {}
        for i, b in enumerate(
                trange(0, len(data), batch_size, desc="Insert Data")):
            _batch = data.iloc[b:b + batch_size]
            data_json = data2json(_batch,
                                  dtype=db_type,
                                  filter_name=filter_name,
                                  predict=predict)
            insert_responses[i] = self._insert_json(name, data_json,
                                                    filter_name, annotations)
        return insert_responses

    def _check_kwargs(self, db_type, **kwargs):
        """
        Sanity checks in the keyword arguments.
        This is a protected method.

        Args
        ----
        db_type : str
            Database type (Supervised, SelfSupervised, Text...)

        Return
        ------
        body: dict
            Body to be sent in the POST request to the API.
        """
        possible = ["hyperparams", "callback_url"]
        must = []
        if db_type == PossibleDtypes.selfsupervised:
            possible.extend([
                'num_process', 'cat_process', 'datetime_process',
                'mycelia_bases', "features"
            ])
        elif db_type == PossibleDtypes.supervised:
            possible.extend([
                'num_process', 'cat_process', 'datetime_process',
                'mycelia_bases', "features", 'label', 'split'
            ])
            must.extend(['label'])

        missing = [key for key in must if kwargs.get(key, None) is None]
        if len(missing) > 0:
            raise ValueError(f"missing arguments {missing}")

        body = {}
        for key in possible:
            val = kwargs.get(key, None)
            if val is not None:
                body[key] = val

        body["db_type"] = db_type
        return body

    def report(self, name, verbose: int = 2, return_report: bool = False):
        """
        Get a report about the training model.

        Parameters
        ----------
        name : str
            String with the name of a database in your JAI environment.
        verbose : int, optional
            Level of description. The default is 2.
            Use verbose 2 to get the loss graph, verbose 1 to get only the
            metrics result.

        Returns
        -------
        dict
            Dictionary with the information.

        """
        dtype = self.get_dtype(name)
        if dtype not in [
                PossibleDtypes.selfsupervised, PossibleDtypes.supervised
        ]:
            return None

        result = self._report(name, verbose)
        if return_report:
            return result

        if 'Model Training' in result.keys():
            plots = result['Model Training']

            plt.plot(*plots['train'], label="train loss")
            plt.plot(*plots['val'], label="val loss")
            plt.title("Training Losses")
            plt.legend()
            plt.xlabel("epoch")
            plt.show()

        print("\nSetup Report:")
        print(result['Model Evaluation']) if 'Model Evaluation' in result.keys(
        ) else None
        print()
        print(result["Loading from checkpoint"].split("\n")
              [1]) if 'Loading from checkpoint' in result.keys() else None

    def get_dtype(self, name):
        """
        Return the database type.

        Parameters
        ----------
        name : str
            String with the name of a database in your JAI environment.

        Raises
        ------
        ValueError
            If the name is not valid.

        Returns
        -------
        db_type : str
            The name of the type of the database.

        """
        if self.is_valid(name):
            dtypes = self.info
            return dtypes.loc[dtypes["name"] == name, "type"].values[0]
        else:
            raise ValueError(f"{name} is not a valid name.")

    def _check_ids_consistency(self, name, data):
        """
        Check if inserted data is consistent with what we expect.
        This is mainly to assert that all data was properly inserted.

        Args
        ----
        name : str
            Database name.
        data : pandas.DataFrame or pandas.Series
            Inserted data.

        Return
        ------
        None or Exception
            If an inconsistency is found, an error is raised.
        """
        inserted_ids = self._temp_ids(name)
        if len(data) != int(inserted_ids[0].split()[0]):
            print(f"Found invalid ids: {inserted_ids[0]}")
            print(self.delete_raw_data(name))
            raise Exception(
                "Something went wrong on data insertion. Please try again.")

    def _check_dtype_and_clean(self, data, db_type):
        """
        Check data type and remove NAs from the data.
        This is a protected method.

        Args
        ----
        data : pandas.DataFrame or pandas.Series
            Data to be checked and cleaned.

        db_type : str
            Database type (Supervised, SelfSupervised, Text...)

        Return
        ------
        data : pandas.DataFrame or pandas.Series
            Data without NAs
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        elif not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"Inserted data is of type `{data.__class__.__name__}`,\
 but supported types are list, np.ndarray, pandas.Series or pandas.DataFrame")
        if db_type in [
                PossibleDtypes.text,
                PossibleDtypes.fasttext,
                PossibleDtypes.edit,
        ]:
            data = data.dropna()
        else:
            cols_to_drop = []
            for col in data.select_dtypes(include=["category", "O"]).columns:
                if data[col].nunique() > 1024:
                    cols_to_drop.append(col)
            data = data.dropna(subset=cols_to_drop)
        return data

    def fields(self, name: str):
        """
        Get the table fields for a Supervised/SelfSupervised database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with table fields.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> fields = j.fields(name=name)
        >>> print(fields)
        {'id': 0, 'feature1': 0.01, 'feature2': 'string', 'feature3': 0}
        """
        dtype = self.get_dtype(name)
        if dtype != PossibleDtypes.selfsupervised and dtype != PossibleDtypes.supervised:
            raise ValueError(
                "'fields' method is only available to dtype SelSupervised and Supervised."
            )
        return self._fields(name)

    def describe(self, name: str):
        """
        Get the database hyperparameters and parameters of a specific database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with database description.
        """
        return self._describe(name)

    def wait_setup(self, name: str, frequency_seconds: int = 1):
        """
        Wait for the setup (model training) to finish

        Placeholder method for scripts.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        frequency_seconds : int, optional
            Number of seconds apart from each status check. `Default is 5`.

        Return
        ------
        None.
        """
        def get_numbers(sts):
            curr_step, max_iterations = sts["Description"].split(
                "Iteration: ")[1].strip().split(" / ")
            return int(curr_step), int(max_iterations)

        max_steps = None
        while max_steps is None:
            status = self.status[name]
            starts_at, max_steps = pbar_steps(status=status)
            time.sleep(1)

        step = starts_at
        aux = 0
        sleep_time = frequency_seconds
        try:
            with tqdm(total=max_steps,
                      desc="JAI is working",
                      bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}]'
                      ) as pbar:
                while status['Status'] != 'Task ended successfully.':
                    if status['Status'] == 'Something went wrong.':
                        raise BaseException(status['Description'])
                    elif fnmatch(status["Description"], "*Iteration:*"):
                        # create a second progress bar to track
                        # training progress
                        _, max_iterations = get_numbers(status)
                        with tqdm(total=max_iterations,
                                  desc=f"[{name}] Training",
                                  leave=False) as iteration_bar:
                            while fnmatch(status["Description"],
                                          "*Iteration:*"):
                                curr_step, _ = get_numbers(status)
                                step_update = curr_step - iteration_bar.n
                                if step_update > 0:
                                    iteration_bar.update(step_update)
                                    sleep_time = frequency_seconds
                                else:
                                    sleep_time += frequency_seconds
                                time.sleep(sleep_time)
                                status = self.status[name]
                            # training might stop early, so we make the progress bar appear
                            # full when early stopping is reached -- peace of mind
                            iteration_bar.update(max_iterations -
                                                 iteration_bar.n)

                    if (step == starts_at) and (aux == 0):
                        pbar.update(starts_at)
                    else:
                        diff = step - starts_at
                        pbar.update(diff)
                        starts_at = step

                    step, _ = pbar_steps(status=status, step=step)
                    time.sleep(frequency_seconds)
                    status = self.status[name]
                    aux += 1

                if (starts_at != max_steps) and aux != 0:
                    diff = max_steps - starts_at
                    pbar.update(diff)
                elif (starts_at != max_steps) and aux == 0:
                    pbar.update(max_steps)

        except KeyboardInterrupt:
            print("\n\nInterruption caught!\n\n")
            raise KeyboardInterrupt(self._cancel_setup(name))

        self._delete_status(name)
        return status

    def delete_ids(self, name, ids):
        """
        Delete the specified ids from database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        ids : list
            List of ids to be removed from database.

        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        return self._delete_ids(name, ids)

    def delete_raw_data(self, name: str):
        """
        Delete raw data. It is good practice to do this after training a model.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        -------
        response : dict
            Dictionary with the API response.

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        return self._delete_raw_data(name)

    def delete_database(self, name: str):
        """
        Delete a database and everything that goes with it (I thank you all).

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with the API response.

        Example
        -------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_database(name=name)
        'Bombs away! We nuked database chosen_name!'
        """
        return self._delete_database(name)

    # Helper function to decide which kind of text model to use
    def _resolve_db_type(self, db_type, col):
        if isinstance(db_type, str):
            return db_type
        elif isinstance(db_type, dict) and col in db_type:
            return db_type[col]
        else:
            return "TextEdit"

    # Helper function to validate name lengths before training
    def _check_name_lengths(self, name, cols):
        invalid_cols = []
        for col in cols:
            if len(name + "_" + col) > 32:
                invalid_cols.append(col)

        if len(invalid_cols):
            raise ValueError(
                f"The following column names are too large to concatenate\
                with database '{name}':\n{invalid_cols}\nPlease enter a shorter database name or\
                shorter column names; 'name_column' string must be at most 32 characters long."
            )

    # Helper function to build the database names of columns that
    # are automatically processed during 'sanity' and 'fill' methods
    def _build_name(self, name, col):
        origin = name + "_" + col
        return origin.lower().replace("-", "_").replace(" ", "_")

    # Helper function to delete the whole tree of databases related with
    # database 'name'
    def _delete_tree(self, name):
        df = self.info
        bases_to_del = df.loc[df["name"] == name, "dependencies"].values[0]
        bases_to_del.append(name)
        total = len(bases_to_del)
        for i, base in enumerate(bases_to_del):
            try:
                msg = self.delete_database(base)
            except:
                msg = f"Database '{base}' does not exist in your environment."
            print(f"({i+1} out of {total}) {msg}")

    def embedding(self,
                  name: str,
                  data,
                  db_type="TextEdit",
                  batch_size: int = 16384,
                  frequency_seconds: int = 1,
                  hyperparams=None,
                  overwrite=False):
        """
        Quick embedding for high numbers of categories in columns.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.Series
            Data for your text based model.
        db_type : str, optional
            type of model to be trained. The default is 'TextEdit'.
        hyperparams: optional
            See setup documentation for the db_type used.

        Returns
        -------
        name : str
            name of the base where the data was embedded.

        """
        if isinstance(data, pd.Series):
            data = data.copy()
        else:
            raise ValueError(
                f"data must be a Series. data is `{data.__class__.__name__}`")

        ids = data.index

        if db_type == "TextEdit":
            hyperparams = {
                "nt": np.clip(np.round(len(data) / 10, -3), 1000, 10000)
            }

        if name not in self.names or overwrite:
            self.setup(
                name,
                data,
                db_type=db_type,
                batch_size=batch_size,
                overwrite=overwrite,
                frequency_seconds=frequency_seconds,
                hyperparams=hyperparams,
            )
        else:
            missing = ids[~np.isin(ids, self.ids(name, "complete"))]
            if len(missing) > 0:
                self.add_data(name,
                              data.loc[missing],
                              batch_size=batch_size,
                              frequency_seconds=frequency_seconds)
        return ids

    def match(self,
              name: str,
              data_left,
              data_right,
              top_k: int = 100,
              batch_size: int = 16384,
              threshold: float = None,
              original_data: bool = False,
              db_type="TextEdit",
              hyperparams=None,
              overwrite: bool = False):
        """
        Match two datasets with their possible equal values.

        Queries the data right to get the similar results in data left.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data_left, data_right : pd.Series
            data to be matched.
        top_k : int, optional
            Number of similars to query. Default is 100.
        threshold : float, optional
            Distance threshold to decide if the result is the same item or not.
            Smaller distances give more strict results. Default is None.
            The threshold is automatically set by default, but may need manual
            setting for more accurate results.
        original_data : bool, optional
            If True, returns the values of the original data along with the ids.
            Default is False.
        db_type : str, optional
            type of model to be trained. The default is 'TextEdit'.
        hyperparams: dict, optional
            See setup documentation for the db_type used.
        overwrite : bool, optional
            If True, then the model is always retrained. Default is False.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with the matching ids of data_left and data_right.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_similar
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> match = j.match(name, data1, data2)
        >>> match
                  id_left     id_right     distance
           0            1            2         0.11
           1            2            1         0.11
           2            3          NaN          NaN
           3            4          NaN          NaN
           4            5            5         0.15
        """
        self.embedding(name,
                       data_left,
                       db_type=db_type,
                       batch_size=batch_size,
                       hyperparams=hyperparams,
                       overwrite=overwrite)
        similar = self.similar(name,
                               data_right,
                               top_k=top_k,
                               batch_size=batch_size)
        processed = process_similar(similar,
                                    threshold=threshold,
                                    return_self=True)
        match = pd.DataFrame(processed).sort_values('query_id')
        match = match.rename(columns={"id": "id_left", "query_id": "id_right"})
        if original_data:
            match['data_letf'] = data_left.loc[match['id_left']].to_numpy(
                copy=True)
            match['data_rigth'] = data_right.loc[match['id_right']].to_numpy(
                copy=True)

        return match

    def resolution(self,
                   name: str,
                   data,
                   top_k: int = 20,
                   batch_size: int = 16384,
                   threshold: float = None,
                   return_self: bool = True,
                   original_data: bool = False,
                   db_type="TextEdit",
                   hyperparams=None,
                   overwrite=False):
        """
        Experimental

        Find possible duplicated values within the data.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.Series
            data to find duplicates.
        top_k : int, optional
            Number of similars to query. Default is 100.
        threshold : float, optional
            Distance threshold to decide if the result is the same item or not.
            Smaller distances give more strict results. Default is None.
            The threshold is automatically set by default, but may need manual
            setting for more accurate results.
        original_data : bool, optional
            If True, returns the ids when resolution_id is the same as id.
            Default is True.
        original_data : bool, optional
            If True, returns the values of the original data along with the ids.
            Default is False.
        db_type : str, optional
            type of model to be trained. The default is 'TextEdit'.
        hyperparams: dict, optional
            See setup documentation for the db_type used.
        overwrite : bool, optional
            If True, then the model is always retrained. Default is False.

        Returns
        -------
        pd.DataFrame
            Each id with its resolution id. More columns depending on parameters.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_similar
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.resolution(name, data)
        >>> results
          id  resolution_id
           0              0
           1              0
           2              0
           3              3
           4              3
           5              5
        """

        ids = self.embedding(name,
                             data,
                             db_type=db_type,
                             batch_size=batch_size,
                             hyperparams=hyperparams,
                             overwrite=overwrite)
        simliar = self.similar(name, ids, top_k=top_k, batch_size=batch_size)
        connect = process_resolution(simliar,
                                     threshold=threshold,
                                     return_self=return_self)
        r = pd.DataFrame(connect).set_index('id').sort_index()

        if original_data:
            r['Original'] = data.loc[r.index.values].to_numpy(copy=True)
            r['Resolution'] = data.loc[r["resolution_id"].values].to_numpy(
                copy=True)
        return r

    def fill(self,
             name: str,
             data,
             column: str,
             batch_size: int = 16384,
             db_type="TextEdit",
             **kwargs):
        """
        Experimental

        Fills the column in data with the most likely value given the other columns.

        Only works with categorical columns. Can not fill missing values for
        numerical columns.

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.DataFrame
            data to fill NaN.
        column : str
            name of the column to be filled.
        db_type : str or dict
            which db_type to use for embedding high dimensional categorical columns.
            If a string is provided, we assume that all columns will be embedded using that db_type;
            if a dict-like structure {"col1": "TextEdit", "col2": "FastText", ...} is provided, we embed the
            specified columns with their respective db_types, and columns not in dict are by default embedded
            with "TextEdit"
        **kwargs : TYPE
            Extra args for supervised model. See setup method.

        Returns
        -------
        list of dicts
            List of dicts with possible filling values for each id with column NaN.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_predict
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.fill(name, data, COL_TO_FILL)
        >>> processed = process_predict(results)
        >>> pd.DataFrame(processed).sort_values('id')
                  id   sanity_prediction    confidence_level (%)
           0       1             value_1                    70.9
           1       4             value_1                    67.3
           2       7             value_1                    80.2
        """
        if "id" in data.columns:
            data = data.set_index("id")
        data = data.copy()
        cat_threshold = kwargs.get("cat_threshold", 512)
        overwrite = kwargs.get("overwrite", False)
        as_frame = kwargs.get("as_frame", False)

        # delete tree of databases derived from 'name',
        # including 'name' itself
        if overwrite:
            self._delete_tree(name)

        if column in data.columns:
            vals = data.loc[:, column].value_counts() < 2
            if vals.sum() > 0:
                eliminate = vals[vals].index.tolist()
                print(
                    f"values {eliminate} from column {column} were removed for having less than 2 examples."
                )
                data.loc[data[column].isin(eliminate), column] = None
        else:
            data.loc[:, column] = None

        cat = data.select_dtypes(exclude="number")

        if name not in self.names:
            mask = data.loc[:, column].isna()
            train = data.loc[~mask].copy()
            test = data.loc[mask].drop(columns=[column])

            # first columns to include are the ones that satisfy
            # the cat_threshold
            pre = cat.columns[cat.nunique() > cat_threshold].tolist()

            # check if db_type is a dict and has some keys in it
            # that do not satisfy the cat_threshold, but must be
            # processed anyway
            if isinstance(db_type, dict):
                pre.extend(
                    [item for item in db_type.keys() if item in cat.columns])

            # we make `pre` a set to ensure it has
            # unique column names
            pre = set(pre)

            prep_bases = []

            # check if database and column names will not overflow the 32-character
            # concatenation limit
            self._check_name_lengths(name, pre)

            for col in pre:
                id_col = "id_" + col
                origin = self._build_name(name, col)

                # find out which db_type to use for this particular column
                curr_db_type = self._resolve_db_type(db_type, col)

                train[id_col] = self.embedding(origin,
                                               train[col],
                                               db_type=curr_db_type)
                test[id_col] = self.embedding(origin,
                                              test[col],
                                              db_type=curr_db_type)
                prep_bases.append({"id_name": id_col, "db_parent": origin})
            train = train.drop(columns=pre)
            test = test.drop(columns=pre)

            label = {"task": "metric_classification", "label_name": column}
            split = {
                "type": "stratified",
                "split_column": column,
                "test_size": 0.2
            }
            mycelia_bases = kwargs.get("mycelia_bases", [])
            mycelia_bases.extend(prep_bases)
            kwargs['mycelia_bases'] = mycelia_bases

            self.setup(
                name,
                train,
                db_type="Supervised",
                batch_size=batch_size,
                label=label,
                split=split,
                **kwargs,
            )
        else:
            drop_cols = []
            for col in cat.columns:
                id_col = "id_" + col
                origin = self._build_name(name, col)

                if origin in self.names:
                    data[id_col] = self.embedding(origin, data[col])
                    drop_cols.append(col)
            if column in data.columns:
                drop_cols.append(column)
            test = data.drop(columns=drop_cols)

        ids_test = test.index
        missing_test = ids_test[~np.isin(ids_test, self.ids(name, "complete"))]
        if len(missing_test) > 0:
            self.add_data(name, test.loc[missing_test], batch_size=batch_size)

        return self.predict(name,
                            test,
                            predict_proba=True,
                            batch_size=batch_size,
                            as_frame=as_frame)

    def sanity(self,
               name: str,
               data,
               batch_size: int = 16384,
               columns_ref: list = None,
               db_type="TextEdit",
               **kwargs):
        """
        Experimental

        Validates consistency in the columns (columns_ref).

        Parameters
        ----------
        name: str
            String with the name of a database in your JAI environment.
        data : pd.DataFrame
            Data reference of sound data.
        columns_ref : list, optional
            Columns that can have inconsistencies. As default we use all non numeric columns.
        db_type : str or dict
            which db_type to use for embedding high dimensional categorical columns.
            If a string is provided, we assume that all columns will be embedded using that db_type;
            if a dict-like structure {"col1": "TextEdit", "col2": "FastText", "col3": "Text", ...} is provided,
            we embed the specified columns with their respective db_types,
            and columns not in dict are by default embedded with "TextEdit"
        kwargs :
            Extra args for supervised model except label and split. See setup method. Also:

            * **frac** (float):
                Percentage of the orignal dataframe to be shuffled to create
                invalid samples for each column in columns_ref. `Default is 0.1`.
            * **random_seed** (int):
                random seed. `Default is 42`.
            * **cat_threshold** (int):
                threshold for processing categorical columns with fasttext model.
                `Default is 512`.
            * **target** (str):
                target validation column. If target is already in data, shuffling is skipped.
                `Default is "is_valid"`.


        Returns
        -------
        list of dicts
            Result of data is valid or not.

        Example
        -------
        >>> import pandas as pd
        >>> from jai.processing import process_predict
        >>>
        >>> j = Jai(AUTH_KEY)
        >>> results = j.sanity(name, data)
        >>> processed = process_predict(results)
        >>> pd.DataFrame(processed).sort_values('id')
                  id   sanity_prediction    confidence_level (%)
           0       1               Valid                    70.9
           1       4             Invalid                    67.3
           2       7             Invalid                    80.6
           3      13               Valid                    74.2
        """
        if "id" in data.columns:
            data = data.set_index("id")
        data = data.copy()

        frac = kwargs.get("frac", 0.1)
        random_seed = kwargs.get("random_seed", 42)
        cat_threshold = kwargs.get("cat_threshold", 512)
        target = kwargs.get("target", "is_valid")
        overwrite = kwargs.get("overwrite", False)
        as_frame = kwargs.get("as_frame", False)

        # delete tree of databases derived from 'name',
        # including 'name' itself
        if overwrite:
            self._delete_tree(name)

        SKIP_SHUFFLING = target in data.columns

        np.random.seed(random_seed)

        cat = data.select_dtypes(exclude="number")

        if name not in self.names:
            # first columns to include are the ones that satisfy
            # the cat_threshold
            pre = cat.columns[cat.nunique() > cat_threshold].tolist()

            # check if db_type is a dict and has some keys in it
            # that do not satisfy the cat_threshold, but must be
            # processed anyway
            if isinstance(db_type, dict):
                pre.extend(
                    [item for item in db_type.keys() if item in cat.columns])

            # we make `pre` a set to ensure it has
            # unique column names
            pre = set(pre)

            if columns_ref is None:
                columns_ref = cat.columns.tolist()
            elif not isinstance(columns_ref, list):
                columns_ref = columns_ref.tolist()

            prep_bases = []

            # check if database and column names will not overflow the 32-character
            # concatenation limit
            self._check_name_lengths(name, pre)

            for col in pre:
                id_col = "id_" + col
                origin = self._build_name(name, col)

                # find out which db_type to use for this particular column
                curr_db_type = self._resolve_db_type(db_type, col)

                data[id_col] = self.embedding(origin,
                                              data[col],
                                              db_type=curr_db_type)
                prep_bases.append({"id_name": id_col, "db_parent": origin})

                if col in columns_ref:
                    columns_ref.remove(col)
                    columns_ref.append(id_col)

            data = data.drop(columns=pre)

            if not SKIP_SHUFFLING:

                def change(options, original):
                    return np.random.choice(options[options != original])

                # get a sample of the data and shuffle it
                sample = []
                strat_split = StratifiedShuffleSplit(n_splits=1,
                                                     test_size=frac,
                                                     random_state=0)
                for c in columns_ref:
                    indexes = []
                    # We try to get a stratified sample on each column.
                    # However, stratified does not work with NaN values, so
                    # we need to drop them before getting the samples
                    try:
                        _, indexes = next(
                            strat_split.split(data.dropna(subset=[c]),
                                              data.dropna(subset=[c])[c]))
                        s = data.dropna(subset=[c]).iloc[indexes].copy()
                    except:
                        pass

                    # due to dropping NaN values, the number of samples might
                    # fall short the desired amount we want;
                    # in this case, we give up the stratified strategy and simply
                    # sample our database randomly.
                    # This 'if' statement below will also hold true if
                    # stratified sampling could not work for whatever
                    # reason (for instance, all samples in a given column are different)
                    if len(indexes) < int(np.floor(data.shape[0] * frac)):
                        s = data.sample(frac=frac)

                    uniques = s[c].unique()
                    s.loc[:, c] = [change(uniques, v) for v in s[c]]
                    sample.append(s)
                sample = pd.concat(sample)

                # set target column values
                sample[target] = "Invalid"

                # set index of samples with different values as data

                sample.index = 10**int(np.log10(data.shape[0]) +
                                       2) + np.arange(len(sample))
                data[target] = "Valid"
                train = pd.concat([data, sample])
            else:
                train = data.copy()

            label = {"task": "metric_classification", "label_name": target}
            split = {
                "type": "stratified",
                "split_column": target,
                "test_size": 0.2
            }

            mycelia_bases = kwargs.get("mycelia_bases", [])
            mycelia_bases.extend(prep_bases)
            kwargs['mycelia_bases'] = mycelia_bases

            self.setup(
                name,
                train,
                db_type="Supervised",
                batch_size=batch_size,
                label=label,
                split=split,
                **kwargs,
            )
        else:

            drop_cols = []
            for col in cat.columns:
                id_col = "id_" + col
                origin = self._build_name(name, col)

                if origin in self.names:
                    data[id_col] = self.embedding(origin, data[col])
                    drop_cols.append(col)

            data = data.drop(columns=drop_cols)

            ids = data.index
            missing = ids[~np.isin(ids, self.ids(name, "complete"))]

            if len(missing) > 0:
                self.add_data(name, data.loc[missing], batch_size=batch_size)

        return self.predict(name,
                            data,
                            predict_proba=True,
                            batch_size=batch_size,
                            as_frame=as_frame)
