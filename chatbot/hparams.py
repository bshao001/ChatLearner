# Copyright 2017 Bo Shao. All Rights Reserved.
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
import codecs
import json
import os
import tensorflow as tf


class HParams:
    def __init__(self, model_dir):
        """
        Args:
            model_dir: Name of the folder storing the hparams.json file.
        """
        self.hparams = self.load_hparams(model_dir)

    @staticmethod
    def load_hparams(model_dir):
        """Load hparams from an existing directory."""
        hparams_file = os.path.join(model_dir, "hparams.json")
        if tf.gfile.Exists(hparams_file):
            print("# Loading hparams from {} ...".format(hparams_file))
            with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
                try:
                    hparams_values = json.load(f)
                    hparams = tf.contrib.training.HParams(**hparams_values)
                except ValueError:
                    print("Error loading hparams file.")
                    return None
            return hparams
        else:
            return None
