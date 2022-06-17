import time
from fnmatch import fnmatch

import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any

from .base import TaskBase
from .query import Query
from ..core.utils_funcs import check_filters, print_args
from ..core.validations import (
    check_response,
    check_dtype_and_clean,
)

from ..types.generic import PossibleDtypes
from ..types.hyperparams import InsertParams

from ..types.responses import (
    Report1Response,
    Report2Response,
    AddDataResponse,
    SetupResponse,
    StatusResponse,
)

from typing import Dict
import pandas as pd
import numpy as np
import json
from collections.abc import Iterable

__all__ = ["Trainer"]


def get_numbers(status):
    if fnmatch(status["Description"], "*Iteration:*"):
        curr_step, max_iterations = (
            status["Description"].split("Iteration: ")[1].strip().split(" / "))
        return True, int(curr_step), int(max_iterations)
    return False, 0, 0


def flatten_sample(sample):
    for el in sample:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten_sample(el)
        else:
            yield el


class Trainer(TaskBase):
    """
    Base class for communication with the Mycelia API.

    Used as foundation for more complex applications for data validation such
    as matching tables, resolution of duplicated values, filling missing values
    and more.

    """
    def __init__(
        self,
        name: str,
        environment: str = "default",
        env_var: str = "JAI_AUTH",
        verbose: int = 1,
        safe_mode: bool = False,
    ):
        """
        Initialize the Jai Trainer class.

        An authorization key is needed to use the Jai API.

        Parameters
        ----------
            name (str): The name of the database.
            environment (str): The environment to use. Defaults to `default`.
            env_var (str): The environment variable that contains the JAI authentication token. Defaults to
            JAI_AUTH
            verbose (int): The level of verbosity. Defaults to 1
            safe_mode (bool): If True, the trainer will not send any data to the server. Defaults to False

        """
        super(Trainer, self).__init__(
            name=name,
            environment=environment,
            env_var=env_var,
            verbose=verbose,
            safe_mode=safe_mode,
        )

        self._verbose = verbose
        self._setup_params = None

        self._insert_params = {"batch_size": 16384, "max_insert_workers": None}

    @property
    def insert_params(self):
        """
        Parameters used for insert data.
        """
        return self._insert_params

    @insert_params.setter
    def insert_params(self, value: InsertParams):
        """
        The method takes a dictionary of parameters

        Returns:
          A dictionary of the InsertParams class.
        """
        self._insert_params = InsertParams(**value).dict()

    @property
    def setup_params(self):
        if self._setup_params is None:
            raise ValueError("Generic error message."
                             )  # TODO: run set_params first message.
        return self._setup_params

    def set_params(
        self,
        db_type: str,
        hyperparams=None,
        features=None,
        num_process: dict = None,
        cat_process: dict = None,
        datetime_process: dict = None,
        pretrained_bases: list = None,
        label: dict = None,
        split: dict = None,
    ):
        """
        It checks the input parameters and sets the `setup_params` attribute for setup.

        TODO: complete args
        Args:
        db_type (str): str
        hyperparams: dict
        features: list of dictionary features to use
        num_process (dict): dict = None,
        cat_process (dict): dict = None,
        datetime_process (dict): dict = None,
        pretrained_bases (list): list = None,
        label (dict): dict = None,
        split (dict): dict = None,
        """

        self._input_kwargs = dict(
            db_type=db_type,
            hyperparams=hyperparams,
            features=features,
            num_process=num_process,
            cat_process=cat_process,
            datetime_process=datetime_process,
            pretrained_bases=pretrained_bases,
            label=label,
            split=split,
        )

        # I figure we don't need a safe_mode validation here
        # because this is already a validation method.
        self._setup_params = self._check_params(
            db_type=db_type,
            hyperparams=hyperparams,
            features=features,
            num_process=num_process,
            cat_process=cat_process,
            datetime_process=datetime_process,
            pretrained_bases=pretrained_bases,
            label=label,
            split=split,
        )

        print_args(self.setup_params,
                   self._input_kwargs,
                   verbose=self._verbose)

    def _check_pretrained_bases(self, data, pretrained_bases):
        for base in pretrained_bases:
            parent_name = base["db_parent"]
            column = base["id_name"]

            if isinstance(data, pd.DataFrame):
                flat_ids = np.unique(list(flatten_sample(data[column])))

                ids = self._ids(parent_name, mode="complete")
                if self.safe_mode:
                    ids = check_response(List[Any], ids)
            elif parent_name in data.keys():
                data_parent = data[parent_name]
                df = data.get(self.name, data["main"])
                flat_ids = np.unique(list(flatten_sample(df[column])))
                ids = (data_parent["id"]
                       if "id" in data_parent.columns else data_parent.index)
            else:
                for df in data.values():
                    if column in df.columns:
                        flat_ids = np.unique(list(flatten_sample(df[column])))
                        ids = self._ids(parent_name, mode="complete")
                        if self.safe_mode:
                            ids = check_response(List[Any], ids)

            inverted_in = np.isin(flat_ids, ids, invert=True)
            if inverted_in.sum() > 0:
                missing = flat_ids[inverted_in].tolist()
                raise KeyError(
                    f"Id values on column `{column}` must belong to the set of Ids from database {parent_name}.\nMissing: {missing}"
                )
            print(f"YES: {column}->{parent_name}")

    def fit(self,
            data,
            *,
            overwrite: bool = False,
            frequency_seconds: int = 1):
        """
        Takes in a dataframe or dictionary of dataframes, and inserts the data into the database.

        It then calls the `_setup` function to train the model.



        Otherwise, it calls the `wait_setup` function to wait for the model to finish training, and then
        calls the `report` function to print out the model's performance metrics.

        Finally, it returns the `get_query` function, which returns the model's predictions.

        Let's take a look at the `_setup` function.

        Args:
            data: The data to be inserted into the database. Can be an pandas.Dataframe or dictionary of pandas.DataFrame.
            overwrite (bool): If overwrite is True, then deletes previous database with the same name if
            exists. Defaults to False.
            frequency_seconds (int): How often to check the status of the model. If `frequency_seconds` is
            less than 1, it returns the `insert_responses` and `setup_response` and it won't wait for
            training to finish, allowing to perform other actions, but could cause errors on some scripts
            if the model is expected to be ready for consuming. Defaults to 1.

        Returns:
        The return value is a tuple of two elements. The first element is a list of responses from the
        insert_data function. The second element is a dictionary of the response from the setup function.
        """
        if self.is_valid():
            if overwrite:
                self.delete_database()
            else:
                raise KeyError(
                    f"Database '{self.name}' already exists in your environment.\
                        Set overwrite=True to overwrite it.")
        self._check_pretrained_bases(
            data, self.setup_params.get("pretrained_bases", []))

        if isinstance(data, (pd.Series, pd.DataFrame)):
            # make sure our data has the correct type and is free of NAs
            data = check_dtype_and_clean(data=data, db_type=self.db_type)

            # insert data
            self._delete_raw_data(self.name)
            insert_responses = self._insert_data(
                data=data,
                name=self.name,
                db_type=self.setup_params["db_type"],
                batch_size=self.insert_params["batch_size"],
                has_filter=check_filters(data,
                                         self.setup_params.get("features",
                                                               {})),
                max_insert_workers=self.insert_params["max_insert_workers"],
                predict=False,
            )

        elif isinstance(data, dict):
            # TODO: check keys

            # loop insert
            for name, value in data.items():

                # make sure our data has the correct type and is free of NAs
                value = check_dtype_and_clean(data=value, db_type=self.db_type)

                # TODO: filter_name fix
                if name == "main":
                    name = self.name

                # insert data
                self._delete_raw_data(name)
                insert_responses = self._insert_data(
                    data=value,
                    name=name,
                    db_type=self.setup_params["db_type"],
                    batch_size=self.insert_params["batch_size"],
                    has_filter=check_filters(
                        value, self.setup_params.get("features", {})),
                    max_insert_workers=self.
                    insert_params["max_insert_workers"],
                    predict=False,
                )
        else:
            raise ValueError(
                "Generic Data Error Message")  # TODO: change message

        # train model
        setup_response = self._setup(self.name,
                                     self.setup_params,
                                     overwrite=overwrite)
        if self.safe_mode:
            setup_response = check_response(SetupResponse,
                                            setup_response).dict()

        print_args(
            {k: json.loads(v)
             for k, v in setup_response["kwargs"].items()},
            self._input_kwargs,
            verbose=self._verbose,
        )

        if frequency_seconds < 1:
            return insert_responses, setup_response

        self.wait_setup(frequency_seconds=frequency_seconds)
        self.report(self._verbose)

        if self.setup_params[
                "db_type"] == PossibleDtypes.recommendation_system:
            towers = list(data.keys())
            return {
                towers[0]: self.get_query(name=towers[0]),
                towers[1]: self.get_query(name=towers[1])
            }

        return self.get_query()  # TODO: maybe fix this for recommendation

    def append(self, data, *, frequency_seconds: int = 1):
        """
        Insert raw data and extract their latent representation.

        This method should be used when we already setup up a database using `fit()`
        and want to create the vector representations of new data
        using the model we already trained for the given database.

        Args
        ----
        data : pandas.DataFrame
            Data to be inserted and used for training.
        frequency_seconds : int
            Time in between each check of status. If less than 1, it won't wait for setup
            to finish, allowing to perform other actions, but could cause errors on some
            scripts. `Default is 1`.

        Return
        -------
        insert_responses: dict
            Dictionary of responses for each batch. Each response contains
            information of whether or not that particular batch was successfully inserted.
        """
        if not self.is_valid():
            raise KeyError(
                f"Database '{self.name}' does not exist in your environment.\n"
                "Run a `setup` set your database up first.")

        # delete data reamains
        self.delete_raw_data()

        # make sure our data has the correct type and is free of NAs
        data = check_dtype_and_clean(data=data, db_type=self.db_type)

        # insert data
        self._delete_raw_data(self.name)
        insert_responses = self._insert_data(
            data=data,
            name=self.name,
            db_type=self.db_type,
            batch_size=self.insert_params["batch_size"],
            has_filter=self.describe()["has_filter"],
            max_insert_workers=self.insert_params["max_insert_workers"],
            predict=True,
        )

        # add data per se
        add_data_response = self._append(name=self.name)
        if self.safe_mode:
            add_data_response = check_response(AddDataResponse,
                                               add_data_response)

        if frequency_seconds >= 1:
            self.wait_setup(frequency_seconds=frequency_seconds)

        return insert_responses, add_data_response

    def status(self):
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
        status = self._status()[self.name]
        if self.safe_mode:
            return check_response(StatusResponse, status).dict()
        return status

    def report(self, verbose: int = 2, return_report: bool = False):
        """
        Get a report about the training model.

        Parameters
        ----------
        verbose : int, optional
            Level of description. The default is 2.
            Use verbose 2 to get the loss graph, verbose 1 to get only the
            metrics result.
        return_report : bool, optional
            Returns the report dictionary and does not print or plot anything. The default is False.


        Returns
        -------
        dict
            Dictionary with the information.

        """
        if self.db_type not in [
                PossibleDtypes.selfsupervised,
                PossibleDtypes.supervised,
                PossibleDtypes.recommendation_system,
        ]:
            return None

        report = self._report(self.name, verbose)

        if self.safe_mode:
            if verbose >= 2:
                report = check_response(Report2Response,
                                        report).dict(by_alias=True)
            elif verbose == 1:
                report = check_response(Report1Response,
                                        report).dict(by_alias=True)
            else:
                report = check_response(Report1Response,
                                        report).dict(by_alias=True)

        if return_report:
            return report

        if "Model Training" in report.keys():
            plots = report["Model Training"]

            plt.plot(*plots["train"], label="train loss")
            plt.plot(*plots["val"], label="val loss")
            plt.title("Training Losses")
            plt.legend()
            plt.xlabel("epoch")
            plt.show()

        print("\nSetup Report:")
        print(report["Model Evaluation"]) if "Model Evaluation" in report.keys(
        ) else None
        print()
        print(report["Loading from checkpoint"].split("\n")
              [1]) if "Loading from checkpoint" in report.keys() else None

    def wait_setup(self, frequency_seconds: int = 1):
        """
        Wait for the setup (model training) to finish

        Args
        ----
        frequency_seconds : int, optional
            Number of seconds apart from each status check. `Default is 5`.

        Return
        ------
        None.
        """

        end_message = "Task ended successfully."
        error_message = "Something went wrong."

        status = self.status()
        current, max_steps = status["CurrentStep"], status["TotalSteps"]

        step = current
        is_init = True
        sleep_time = frequency_seconds
        try:
            with tqdm(
                    total=max_steps,
                    desc="JAI is working",
                    bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}]",
            ) as pbar:
                while status["Status"] != end_message:
                    if status["Status"] == error_message:
                        raise BaseException(status["Description"])

                    iteration, _, max_iterations = get_numbers(status)
                    if iteration:
                        with tqdm(
                                total=max_iterations,
                                desc=f"[{self.name}] Training",
                                leave=False,
                        ) as iteration_bar:
                            while iteration:
                                iteration, curr_step, _ = get_numbers(status)
                                step_update = curr_step - iteration_bar.n
                                if step_update > 0:
                                    iteration_bar.update(step_update)
                                    sleep_time = frequency_seconds
                                else:
                                    sleep_time += frequency_seconds
                                time.sleep(sleep_time)
                                status = self.status()

                            # training might stop early, so we make the progress bar appear
                            # full when early stopping is reached -- peace of mind
                            iteration_bar.update(max_iterations -
                                                 iteration_bar.n)

                    if (step == current) and is_init:
                        pbar.update(current)
                    else:
                        pbar.update(step - current)
                        current = step

                    step = status["CurrentStep"]
                    time.sleep(frequency_seconds)
                    status = self.status()
                    is_init = False

                if (current != max_steps) and not is_init:
                    pbar.update(max_steps - current)
                elif (current != max_steps) and is_init:
                    pbar.update(max_steps)

        except KeyboardInterrupt:
            print("\n\nInterruption caught!\n\n")
            response = self._cancel_setup(self.name)
            if self.safe_mode:
                response = check_response(str, response)
            raise KeyboardInterrupt(response)

        response = self._delete_status(self.name)
        if self.safe_mode:
            check_response(str, response)
        return status

    def delete_ids(self, ids):
        """
        Delete the specified ids from database.

        Args
        ----
        ids : list
            List of ids to be removed from database.

        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        response = self._delete_ids(self.name, ids)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def delete_raw_data(self):
        """
        Delete raw data. It is good practice to do this after training a model.


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
        response = self._delete_raw_data(self.name)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def delete_database(self):
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
        response = self._delete_database(self.name)
        if self.safe_mode:
            return check_response(str, response)
        return response

    def get_query(self, name: str = None):
        """
        The function returns a new `Query` object with the same initial values as the current `Trainer`
        object

        Args:
          name (str): The name of the query. Defaults to the same name as the current `Trainer` object.

        Returns:
          A Query object with the name and init values.
        """
        if name is None:
            return Query(name=self.name, **self._init_values)
        return Query(name=name, **self._init_values)
