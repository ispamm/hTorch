import unittest
import torch
from qtorch.quaternion import QuaternionTensor

class TestQuaternionTensor(unittest.TestCase):

    def test_torch_casting(self):
        # Test casting to standard PyTorch tensor
        x = QuaternionTensor([0.1, 0.2, 0.3, 0.4])
        print(x.torch())
        self.assertTrue(torch.equal(x.torch(), torch.tensor([0.1, 0.2, 0.3, 0.4])))

if __name__ == '__main__':
    unittest.main()