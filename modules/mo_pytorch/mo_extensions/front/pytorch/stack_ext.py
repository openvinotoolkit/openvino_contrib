from extensions.ops.pack import PackOp
from mo.front.extractor import FrontExtractorOp


class StackFrontExtractor(FrontExtractorOp):
    op = 'Stack'
    enabled = True

    @classmethod
    def extract(cls, node):
        update_attrs = {
            'axis': 0
        }

        # update the attributes of the node
        PackOp.update_node_stat(node, update_attrs)

        return cls.enabled
