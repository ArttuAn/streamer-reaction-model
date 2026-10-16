import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from checkpoint_io import load_tensor_data

class CheckpointTests(unittest.TestCase):
    def test_tensor_checkpoint_and_retrieval_index_roundtrip(self):
        data = {"model_state": {"head.weight": torch.arange(6).reshape(2, 3)}, "paths": ["clips/sample"], "emotion_classes": ["happy"], "epoch": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(data, path)
            loaded = load_tensor_data(path)
            self.assertTrue(torch.equal(loaded["model_state"]["head.weight"], data["model_state"]["head.weight"]))
            self.assertEqual(loaded["paths"], data["paths"])
            self.assertEqual(loaded["epoch"], 2)

    def test_older_torch_is_rejected_before_deserialization(self):
        with patch.object(torch, "__version__", "2.5.1"), patch.object(torch, "load") as load:
            with self.assertRaises(RuntimeError):
                load_tensor_data("untrusted.pt")
            load.assert_not_called()

    def test_checkpoint_cannot_execute_pickle_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            class Payload:
                def __reduce__(self):
                    return (Path.touch, (marker,))
            path = Path(directory) / "untrusted.pt"
            torch.save({"payload": Payload()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_tensor_data(path)
            self.assertFalse(marker.exists())

if __name__ == "__main__":
    unittest.main()
