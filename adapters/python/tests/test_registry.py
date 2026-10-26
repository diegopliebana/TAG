"""Game discovery and generic consumers use Rust's construction registry."""
import importlib.util
import unittest
from unittest.mock import patch
import tag_rust as tag


class GameRegistry(unittest.TestCase):
    def test_discovered_games_construct_and_unknown_names_report_the_registry(self):
        names = tag.registered_games()
        self.assertTrue(names)
        self.assertEqual(len(names),len(set(names)))
        for name in names:
            with tag.create(name) as game:
                self.assertEqual(game.spec().identity.game,name)
        with self.assertRaises(ValueError) as failure:
            tag.create("unregistered-test-game")
        self.assertIn("unregistered-test-game",str(failure.exception))
        for name in names:
            self.assertIn(name,str(failure.exception))

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch not installed")
    def test_training_settings_accept_a_game_advertised_by_the_registry(self):
        from tag_rust.learning.training import TrainingSettings
        with patch("tag_rust.learning.training.registered_games",return_value=("registered-fixture",)):
            # This checks configuration discovery, without claiming an unimplemented
            # fixture has episode, encoder or training support.
            self.assertEqual(TrainingSettings(game="registered-fixture").game,"registered-fixture")
            with self.assertRaisesRegex(ValueError,"registered-fixture"):
                TrainingSettings(game="unregistered-test-game")


if __name__ == "__main__":
    unittest.main()
