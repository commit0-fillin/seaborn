"""
Components for parsing variable assignments and internally representing plot data.
"""
from __future__ import annotations
from collections.abc import Mapping, Sized
from typing import cast
import warnings
import pandas as pd
from pandas import DataFrame
from seaborn._core.typing import DataSource, VariableSpec, ColumnName
from seaborn.utils import _version_predates

class PlotData:
    """
    Data table with plot variable schema and mapping to original names.

    Contains logic for parsing variable specification arguments and updating
    the table with layer-specific data and/or mappings.

    Parameters
    ----------
    data
        Input data where variable names map to vector values.
    variables
        Keys are names of plot variables (x, y, ...) each value is one of:

        - name of a column (or index level, or dictionary entry) in `data`
        - vector in any format that can construct a :class:`pandas.DataFrame`

    Attributes
    ----------
    frame
        Data table with column names having defined plot variables.
    names
        Dictionary mapping plot variable names to names in source data structure(s).
    ids
        Dictionary mapping plot variable names to unique data source identifiers.

    """
    frame: DataFrame
    frames: dict[tuple, DataFrame]
    names: dict[str, str | None]
    ids: dict[str, str | int]
    source_data: DataSource
    source_vars: dict[str, VariableSpec]

    def __init__(self, data: DataSource, variables: dict[str, VariableSpec]):
        data = handle_data_source(data)
        frame, names, ids = self._assign_variables(data, variables)
        self.frame = frame
        self.names = names
        self.ids = ids
        self.frames = {}
        self.source_data = data
        self.source_vars = variables

    def __contains__(self, key: str) -> bool:
        """Boolean check on whether a variable is defined in this dataset."""
        if self.frame is None:
            return any((key in df for df in self.frames.values()))
        return key in self.frame

    def join(self, data: DataSource, variables: dict[str, VariableSpec] | None) -> PlotData:
        """Add, replace, or drop variables and return as a new dataset."""
        new_data = handle_data_source(data)
        new_frame, new_names, new_ids = self._assign_variables(new_data, variables or {})
        
        # Combine the existing and new data
        combined_frame = pd.concat([self.frame, new_frame], axis=1)
        combined_names = {**self.names, **new_names}
        combined_ids = {**self.ids, **new_ids}
        
        # Create a new PlotData instance
        new_plot_data = PlotData(combined_frame, {})
        new_plot_data.frame = combined_frame
        new_plot_data.names = combined_names
        new_plot_data.ids = combined_ids
        new_plot_data.source_data = {**self.source_data, **new_data} if isinstance(self.source_data, dict) else new_data
        new_plot_data.source_vars = {**self.source_vars, **(variables or {})}
        
        return new_plot_data

    def _assign_variables(self, data: DataFrame | Mapping | None, variables: dict[str, VariableSpec]) -> tuple[DataFrame, dict[str, str | None], dict[str, str | int]]:
        """
        Assign values for plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data
            Input data where variable names map to vector values.
        variables
            Keys are names of plot variables (x, y, ...) each value is one of:

            - name of a column (or index level, or dictionary entry) in `data`
            - vector in any format that can construct a :class:`pandas.DataFrame`

        Returns
        -------
        frame
            Table mapping seaborn variables (x, y, color, ...) to data vectors.
        names
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).
        ids
            Like the `names` dict, but `None` values are replaced by the `id()`
            of the data object that defined the variable.

        Raises
        ------
        TypeError
            When data source is not a DataFrame or Mapping.
        ValueError
            When variables are strings that don't appear in `data`, or when they are
            non-indexed vector datatypes that have a different length from `data`.

        """
        if data is None and not variables:
            return pd.DataFrame(), {}, {}

        if not isinstance(data, (pd.DataFrame, Mapping)) and data is not None:
            raise TypeError("Data must be a DataFrame or Mapping")

        frame = pd.DataFrame()
        names = {}
        ids = {}

        for var_name, var_spec in variables.items():
            if isinstance(var_spec, str):
                if data is None or var_spec not in data:
                    raise ValueError(f"Variable '{var_spec}' not found in data")
                frame[var_name] = data[var_spec]
                names[var_name] = var_spec
                ids[var_name] = id(data[var_spec])
            else:
                try:
                    series = pd.Series(var_spec, name=var_name)
                    if data is not None and len(series) != len(data):
                        raise ValueError(f"Length of {var_name} does not match length of data")
                    frame[var_name] = series
                    names[var_name] = getattr(var_spec, 'name', None)
                    ids[var_name] = id(var_spec)
                except Exception as e:
                    raise ValueError(f"Could not convert {var_name} to a Series: {str(e)}")

        return frame, names, ids

def handle_data_source(data: object) -> pd.DataFrame | Mapping | None:
    """Convert the data source object to a common union representation."""
    if data is None:
        return None
    elif isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, Mapping):
        return data
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    elif hasattr(data, '__dataframe__'):  # Check for DataFrame interchange protocol
        return convert_dataframe_to_pandas(data)
    else:
        try:
            return pd.DataFrame(data)
        except Exception:
            raise TypeError(f"Could not convert data of type {type(data)} to DataFrame or Mapping")

def convert_dataframe_to_pandas(data: object) -> pd.DataFrame:
    """Use the DataFrame exchange protocol, or fail gracefully."""
    try:
        df_protocol = data.__dataframe__()
        if hasattr(df_protocol, 'to_pandas'):
            return df_protocol.to_pandas()
        else:
            # Fallback to manual conversion if to_pandas() is not available
            columns = [col.name for col in df_protocol.columns()]
            data_dict = {col: df_protocol.get_column(col).to_numpy() for col in columns}
            return pd.DataFrame(data_dict)
    except Exception as e:
        raise ValueError(f"Failed to convert data using DataFrame exchange protocol: {str(e)}")
