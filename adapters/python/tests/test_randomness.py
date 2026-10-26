"""Automatic setup resolves once, records the seed and preserves exact resumes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tag_rust.randomness import resolve_seed, seed_argument


class RandomnessTests(unittest.TestCase):
    def test_os_entropy_explicit_seeds_and_errors(self):
        with patch("tag_rust.randomness.secrets.randbits", return_value=2**64-1) as entropy:
            self.assertEqual(resolve_seed(),2**64-1)
            entropy.assert_called_once_with(64)
            self.assertEqual(resolve_seed(0),0)
            self.assertEqual(resolve_seed(42),42)
            entropy.assert_called_once()
        with patch("tag_rust.randomness.secrets.randbits", side_effect=OSError("no entropy")):
            with self.assertRaisesRegex(OSError,"no entropy"):
                resolve_seed()
        for invalid in (-1,True,1.5,"auto"):
            with self.assertRaises(ValueError):resolve_seed(invalid)
        self.assertIsNone(seed_argument("auto"))
        self.assertIsNone(seed_argument("null"))
        self.assertEqual(seed_argument("42"),42)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch not installed")
    def test_automatic_training_matches_explicit_seed_and_checkpoint_never_reseeds(self):
        import torch
        from dataclasses import replace
        from tag_rust.learning.training import TrainingSettings,TrainingSession
        from tag_rust.learning.artifacts import save_checkpoint,load_checkpoint
        settings=TrainingSettings(teacher_episodes=1,actor_episodes=0,group_size=1,hidden_sizes=(4,))
        with patch("tag_rust.randomness.secrets.randbits", return_value=73) as entropy:
            automatic=TrainingSession(settings)
            explicit=TrainingSession(replace(settings,master_seed=73))
            self.assertEqual(automatic.settings.master_seed,73)
            self.assertEqual(automatic.step(),explicit.step())
            entropy.assert_called_once_with(64)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.pt"
            save_checkpoint(path,automatic)
            with patch("tag_rust.randomness.secrets.randbits",side_effect=AssertionError("resume reseeded")):
                restored=load_checkpoint(path)
            self.assertEqual(restored.settings.master_seed,73)
            self.assertEqual(restored.history,automatic.history)
            for key,value in automatic.model.state_dict().items():
                self.assertTrue(torch.equal(value,restored.model.state_dict()[key]))
            data=torch.load(path,weights_only=True)
            data["settings"]["master_seed"]=None
            torch.save(data,path)
            with self.assertRaisesRegex(ValueError,"resolved master seed"):
                load_checkpoint(path)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch not installed")
    def test_automatic_evaluation_reports_reproducible_seed(self):
        from tag_rust.learning.training import TrainingSettings,TrainingSession
        from tag_rust.learning.evaluation import evaluate_model
        session=TrainingSession(TrainingSettings(master_seed=5,teacher_episodes=1,actor_episodes=0,hidden_sizes=(4,)))
        with patch("tag_rust.randomness.secrets.randbits",return_value=91) as entropy:
            automatic=evaluate_model(session.model,episodes=2)
            explicit=evaluate_model(session.model,episodes=2,seed=91)
            self.assertEqual(automatic,explicit)
            self.assertEqual(automatic["seed"],91)
            entropy.assert_called_once_with(64)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch not installed")
    def test_automatic_collection_retains_its_master_seed(self):
        from tag_rust.learning.collection import collect_episode_group
        with patch("tag_rust.randomness.secrets.randbits",return_value=31) as entropy:
            group=collect_episode_group("tictactoe",episode_count=1,phase="teacher",payloads=("cells",))
            replay=collect_episode_group("tictactoe",episode_count=1,phase="teacher",payloads=("cells",),master_seed=31)
            self.assertEqual(group.master_seed,31)
            self.assertEqual(group.outcomes,replay.outcomes)
            self.assertEqual(group.episodes[0].reset_seed,replay.episodes[0].reset_seed)
            self.assertEqual(group.episodes[0].teacher_seed,replay.episodes[0].teacher_seed)
            entropy.assert_called_once_with(64)
