# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
from os import path as osp
from typing import List

TIME_CONVERSION_FROM_SECONDS = {
    'seconds': 1.0,
    'minutes': 60.0,
    'hours': 3600.0,
    'days': 3600.0*24.0
}

class GridIndexTracker:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row = None
        self.col = None


    def __call__(self):
        if self.n_rows == 1:
            return (self.col)
        if self.n_cols == 1:
            return (self.row)
        return (self.row, self.col)


    def next(self):
        if self.row is None and self.col is None:
            self.row = 0
            self.col = 0
        else:
            self.col += 1
            if self.col >= self.n_cols:
                self.col = 0
                self.row += 1
                if self.row >= self.n_rows:
                    raise IndexError("Exceeded maximum number of rows")

def get_existing_extension(location: str,
                           fname: str,
                           possible_extensions: List[str]
                           ) -> str:
    """Returns the found extension with a leading dot"""
    for ext in possible_extensions:
        if osp.isfile(osp.join(location, fname+'.'+ext)):
            return '.'+ext
    return None