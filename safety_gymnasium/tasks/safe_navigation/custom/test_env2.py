"""Test Environment 2."""

from safety_gymnasium.assets.mocaps import Gremlins
# from safety_gymnasium.tasks.safe_navigation.custom.test_env0 import TestTaskLevel0
from safety_gymnasium.tasks.safe_navigation.custom.test_env1 import TestTaskLevel1

class TestTaskLevel2(TestTaskLevel1):
    """An agent must navigate to a goal while avoiding gremlins.

    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        print("TestTaskLevel2")
        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        self._add_mocaps(Gremlins(num=2, keepout=0.18))