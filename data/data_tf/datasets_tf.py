# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Small library that points to the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.data_tf.dataset_tf import Dataset_TFRecords


class CatsDogsData(Dataset_TFRecords):

    def __init__(self, subset, file_pattern):
        super(CatsDogsData, self).__init__('CatsDogs', subset, file_pattern)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 2

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 20000

        if self.subset == 'validation':
            return 5000

class IrisWarsawData(Dataset_TFRecords):

    def __init__(self, subset, file_pattern):
        super(IrisWarsawData, self).__init__('IrisWarsaw', subset, file_pattern)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 2

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 431

        if self.subset == 'validation':
            return 1236

class IrisATVSData(Dataset_TFRecords):

    def __init__(self, subset, file_pattern):
        super(IrisATVSData, self).__init__('IrisATVS', subset, file_pattern)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 2

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 400

        if self.subset == 'validation':
            return 1200

class IrisMobBioFakeData(Dataset_TFRecords):

    def __init__(self, subset, file_pattern):
        super(IrisMobBioFakeData, self).__init__('IrisMobBioFake', subset, file_pattern)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 2

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 800

        if self.subset == 'validation':
            return 800

class ReplayAttackData(Dataset_TFRecords):

    def __init__(self, subset, file_pattern):
        super(ReplayAttackData, self).__init__('ReplayAttack', subset, file_pattern)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 2

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return None

        if self.subset == 'validation':
            return None

        if self.subset == 'test':
            return None