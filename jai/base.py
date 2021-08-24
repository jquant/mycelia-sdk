import os
import json
import requests
import functools

from copy import copy

from .functions.classes import Mode
from .functions import exceptions

__all__ = ["BaseJai"]


def raise_status_error(code):
    """
    Decorator to process responses with unexpected response codes.

    Args
    ----
    code: int
        Expected Code.

    """
    def decorator(function):
        @functools.wraps(function)
        def new_function(*args, **kwargs):
            response = function(*args, **kwargs)
            if response.status_code == code:
                return response.json()
            # find a way to process this
            # what errors to raise, etc.
            message = f"Something went wrong.\n\nSTATUS: {response.status_code}\n"
            try:
                res_json = response.json()
                print(res_json)
                if isinstance(res_json, dict):
                    detail = res_json.get(
                        'message', res_json.get('detail', response.text))
                else:
                    detail = response.text
            except:
                detail = response.text

            detail = str(detail)

            if "Error: " in detail:
                error, msg = detail.split(": ", 1)
                try:
                    raise eval(error)(message + msg)
                except NameError:
                    raise eval("exceptions." + error)(message + msg)
                except:
                    raise ValueError(message + response.text)
            else:
                raise ValueError(message + detail)

        return new_function

    return decorator


class BaseJai(object):
    """
    Base class for requests with the Mycelia API.
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
        if auth_key is None:
            auth_key = os.environ.get(var_env, "")
        if url is None:
            self.__url = "https://mycelia.azure-api.net"
            self.header = {"Auth": auth_key}
        else:
            self.__url = url[:-1] if url.endswith("/") else url
            self.header = {"company-key": auth_key}

    @property
    def url(self):
        """
        Get name and type of each database in your environment.
        """
        return self.__url

    @raise_status_error(200)
    def _info(self, mode="complete", get_size=True):
        """
        Get name and type of each database in your environment.
        """
        get_size = json.dumps(get_size)
        return requests.get(url=self.url +
                            f"/info?mode={mode}&get_size={get_size}",
                            headers=self.header)

    @raise_status_error(200)
    def _status(self):
        """
        Get the status of your JAI environment when training.
        """
        return requests.get(self.url + "/status", headers=self.header)

    @raise_status_error(200)
    def _delete_status(self, name):
        return requests.delete(self.url + f"/status?db_name={name}",
                               headers=self.header)

    @raise_status_error(200)
    def _download_vectors(self, name: str):
        """
        Download vectors from a particular database.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        return requests.get(self.url + f"/key/{name}", headers=self.header)

    @raise_status_error(200)
    def _filters(self, name):
        """
        Gets the valid values of filters.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        """
        return requests.get(self.url + f"/filters/{name}", headers=self.header)

    @raise_status_error(200)
    def _similar_id(self,
                    name: str,
                    id_item: list,
                    top_k: int = 5,
                    filters=None):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given an id.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        id_item : list
            List of ids of the item the user is looking for.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar items`.
        """

        if not isinstance(id_item, list):
            raise TypeError(
                f"id_item param must be int or list, `{id_item.__class__.__name__}` found."
            )

        filtering = "" if filters is None else "".join(
            ["&filters=" + s for s in filters])
        url = self.url + f"/similar/id/{name}?top_k={top_k}" + filtering
        return requests.put(
            url,
            headers=self.header,
            json=id_item,
        )

    @raise_status_error(200)
    def _similar_json(self,
                      name: str,
                      data_json,
                      top_k: int = 5,
                      filters=None):
        """
        Creates a list of dicts, with the index and distance of the k items most similars given a JSON data entry.
        This is a protected method

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        data_json : dict (JSON)
            Data in JSON format. Each input in the dictionary will be used to search for the `top_k` most
            similar entries in the database.

        top_k : int
            Number of k similar items we want to return. `Default is 5`.

        Return
        ------
        response : dict
            Dictionary with the index and distance of `the k most similar items`.
        """
        filtering = "" if filters is None else "".join(
            ["&filters=" + s for s in filters])
        url = self.url + f"/similar/data/{name}?top_k={top_k}" + filtering
        header = copy(self.header)
        header['Content-Type'] = "application/json"
        return requests.put(url, headers=header, data=data_json)

    @raise_status_error(200)
    def _predict(self,
                 name: str,
                 data_json,
                 predict_proba: bool = False,
                 annotations=None):
        """
        Predict the output of new data for a given database by calling its
        respecive API method. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_json : JSON file (dict)
            Data to be inferred by the previosly trained model.
        predict_proba : bool
            Whether or not to return the probabilities of each prediction. `Default is False`.

        Return
        -------
        results : dict
            Dictionary of predctions for the data passed as parameter.
        """
        url = self.url + \
            f"/predict/{name}?predict_proba={predict_proba}"

        header = copy(self.header)
        header['Content-Type'] = "application/json"
        data = {'data': json.loads(data_json)}
        if annotations is not None:
            data = {**data, **annotations}
        data = json.dumps(data)
        return requests.put(url, headers=header, data=data)

    @raise_status_error(200)
    def _ids(self, name: str, mode: Mode = "simple"):
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
        return requests.get(self.url + f"/id/{name}?mode={mode}",
                            headers=self.header)

    @raise_status_error(200)
    def _is_valid(self, name: str):
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
        """
        return requests.get(self.url + f"/validation/{name}",
                            headers=self.header)

    @raise_status_error(202)
    def _append(self, name: str):
        """
        Add data to a database that has been previously trained.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        return requests.patch(self.url + f"/data/{name}", headers=self.header)

    @raise_status_error(200)
    def _insert_json(self,
                     name: str,
                     data_json,
                     filter_name: str = None,
                     annotations=None):
        """
        Insert data in JSON format. This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        data_json : dict
            Data in JSON format.

        Return
        ------
        response : dict
            Dictionary with the API response.
        """
        filtering = "" if filter_name is None else f"?filter_name={filter_name}"
        url = self.url + f"/data/{name}" + filtering

        header = copy(self.header)
        header['Content-Type'] = "application/json"
        data = {'table': json.loads(data_json)}
        if annotations is not None:
            data = {**data, **annotations}
        data = json.dumps(data)
        return requests.post(url, headers=header, data=data)

    @raise_status_error(201)
    def _setup(self, name: str, body, overwrite=False):
        """
        Call the API method for database setup.
        This is a protected method.

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        db_type : str
            Database type (Supervised, SelfSupervised, Text...)
        overwrite : bool
            [Optional] Whether of not to overwrite the given database. `Default is False`.
        **kwargs:
            Any parameters the user wants to (or needs to) set for the given datase. Please
            refer to the API methods to see the possible arguments.

        Return
        -------
        response : dict
            Dictionary with the API response.
        """
        overwrite = json.dumps(overwrite)
        return requests.post(
            self.url + f"/setup/{name}?overwrite={overwrite}",
            headers=self.header,
            json=body,
        )

    @raise_status_error(200)
    def _report(self, name, verbose: int = 2):
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
        return requests.get(self.url + f"/report/{name}?verbose={verbose}",
                            headers=self.header)

    @raise_status_error(200)
    def _temp_ids(self, name: str, mode: Mode = "simple"):
        """
        Get id information of a RAW database (i.e., before training). This is a protected method

        Args
        ----
        name : str
            String with the name of a database in your JAI environment.
        mode : str, optional
            Level of detail to return. Possible values are 'simple', 'summarized' or 'complete'.

        Return
        -------
        response: list
            List with the actual ids (mode: 'complete') or a summary of ids
            ('simple'/'summarized') of the given database.
        """
        return requests.get(self.url + f"/setup/ids/{name}?mode={mode}",
                            headers=self.header)

    @raise_status_error(200)
    def _fields(self, name: str):
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
        """
        return requests.get(self.url + f"/fields/{name}", headers=self.header)

    @raise_status_error(200)
    def _describe(self, name: str):
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
        return requests.get(self.url + f"/describe/{name}",
                            headers=self.header)

    @raise_status_error(200)
    def _cancel_setup(self, name: str):
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
        return requests.post(self.url + f'/cancel/{name}', headers=self.header)

    @raise_status_error(200)
    def _delete_ids(self, name, ids):
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

        Example
        ----------
        >>> name = 'chosen_name'
        >>> j = Jai(AUTH_KEY)
        >>> j.delete_raw_data(name=name)
        'All raw data from database 'chosen_name' was deleted!'
        """
        return requests.delete(self.url + f"/entity/{name}",
                               headers=self.header,
                               json=ids)

    @raise_status_error(200)
    def _delete_raw_data(self, name: str):
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
        return requests.delete(self.url + f"/data/{name}", headers=self.header)

    @raise_status_error(200)
    def _delete_database(self, name: str):
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
        return requests.delete(self.url + f"/database/{name}",
                               headers=self.header)
