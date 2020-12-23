from anytree import NodeMixin, RenderTree

"""
Ampliation of the Node from the Anytree library
"""
class TreeNode(NodeMixin):  # Add Node feature

    """
    @attr   name
    @attr   target_result
    @attr   values
    @attr   branches
    @attr   continious
    @attr   threshold
    @attr   parent
    @attr   children
    """

    def __init__(self, name, feature=None, target_result=None, values=None,
                 continious=0, threshold=None, parent=None, children=None):
        self.name = name
        self.feature = feature
        self.target_result = target_result
        self.values = values
        self.continious = continious
        self.threshold = threshold
        self.parent = parent
        if children:  # set children only if given
            self.children = children
