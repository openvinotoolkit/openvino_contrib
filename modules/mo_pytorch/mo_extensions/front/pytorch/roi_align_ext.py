"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.front.extractor import FrontExtractorOp
from extensions.ops.roialign import ROIAlign


class ROIAlignExtractor(FrontExtractorOp):
    op = 'ROIAlign'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'pooled_h': node.module.pooled_h,
            'pooled_w': node.module.pooled_w,
            'sampling_ratio': node.module.sampling_ratio,
            'mode': node.module.mode,
            'spatial_scale': node.module.spatial_scale,
        }
        ROIAlign.update_node_stat(node, attrs)
        return cls.enabled
