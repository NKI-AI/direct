# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict


def fix_state_dict_module_prefix(state_dict):
    """If models are saved after being wrapped in e.g. DataParallel, the keys of the state dict are prefixed with
    `module.`. This function removes this prefix.

    Parameters
    ----------
    state_dict: dict
        state_dict of a network module
    Returns
    -------
    dict
    """
    if list(state_dict.keys())[0].startswith("module."):
        new_ordered_dict = OrderedDict()
        for _, (k, v) in enumerate(state_dict.items()):
            name = k[7:]
            new_ordered_dict[name] = v
        state_dict = new_ordered_dict

    return state_dict
