.. highlight:: shell

=======================
Adding your own dataset
=======================
Transforms in :code:`DIRECT` currently support only gridded data (data acquired on an equispaced grid).
Any compatible dataset should inherit from PyTorch's dataset class :code:`torch.utils.data.Dataset`.
Follow the steps below:

- Implement your custom dataset under :code:`direct/data/datasets.py` following the template:

.. code-block:: python

    import pathlib

    from torch.utils.data import Dataset

    logger = logging.getLogger(__name__)

    class MyNewDataset(Dataset):
        """
        Information about the Dataset.
        """

        def __init__(
            self,
            root: pathlib.Path,
            transform: Optional[Callable] = None,
            filenames_filter: Optional[List[PathOrString]] = None,
            text_description: Optional[str] = None,
            ...
        ) -> None:
            """
            Initialize the dataset.

            Parameters
            ----------
            root : pathlib.Path
                Root directory to saved data.
            transform : Optional[Callable]
                Callable function that transforms the loaded data.
            filenames_filter : List
                List of filenames to include in the dataset.
            text_description : str
                Description of dataset, can be useful for logging.
            ...
            ...
            """
            super().__init__()

            self.logger = logging.getLogger(type(self).__name__)
            self.root = root
            self.transform = transform
            if filenames_filter:
                self.logger.info(f"Attempting to load {len(filenames_filter)} filenames from list.")
                filenames = filenames_filter
            else:
                self.logger.info(f"Parsing directory {self.root} for <data_type> files.")
            filenames = list(self.root.glob("*.<data_type>"))
            self.filenames_filter = filenames_filter

            self.text_description = text_description

            ...

        def self.get_dataset_len(self):
            ...

        def __len__(self):
            return self.get_dataset_len()

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            ...
            sample = ...
            ...
            if self.transform:
                sample = self.transform(sample)
            return sample


Note that the :code:`__getitem__` method should output dictionaries which contain keys with values either torch.Tensors or
other metadata. Current implemented models and transforms can work with multi-coil two-dimensional data. Therefore, new datasets
should split three-dimensional data to slices of two-dimensional data.


- Register the new dataset in :code:`direct/data/datasets_config.py`

.. code-block:: python

    @dataclass
    class MyDatasetConfig(BaseConfig):
        ...
        name: str = "MyNew"
        lists: List[str] = field(default_factory=lambda: [])
        transforms: BaseConfig = TransformsConfig()
        text_description: Optional[str] = None
        ...


- To use your dataset, you have to request it in the :code:`config.yaml` file. The following shows an example for training.


.. code-block:: yaml

    training:
        datasets:
        -   name: MyNew
            lists:
                - <list_1_name>.lst
                - <list_2_name>.lst
                - ...
            transforms:
                ...
                masking:
                    ...
            ...

