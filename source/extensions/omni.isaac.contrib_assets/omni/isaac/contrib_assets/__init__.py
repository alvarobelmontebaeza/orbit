# Copyright (c) 2024 Álvaro Belmonte Baeza
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations added from external contributors."""

import os
import toml

# Conveniences to other module directories via relative paths
CONTRIB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

CONTRIB_ASSETS_DATA_DIR = os.path.join(CONTRIB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

CONTRIB_ASSETS_METADATA = toml.load(os.path.join(CONTRIB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = CONTRIB_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .tako import *
