from safety_gymnasium.assets.mocaps import Gremlins_Old
from safety_gymnasium.tasks.safe_navigation.custom.test_env0 import TestTaskLevel0


class TestTaskLevel4(TestTaskLevel0):
    """An agent must navigate to a goal while avoiding gremlins."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self._add_mocaps(Gremlins_Old(num=8, keepout=0.18))
