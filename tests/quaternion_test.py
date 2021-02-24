import unittest
import sys  # DOPO LEVALO GIOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
sys.path.append("..")
import torch
from htorch.quaternion import QuaternionTensor
import pyquaternion


class TestQuaternionTensor(unittest.TestCase):

    def test_torch_casting(self):
        # Test casting to standard PyTorch tensor
        x = QuaternionTensor([0.1, 0.2, 0.3, 0.4])
        print(x.torch())
        self.assertTrue(torch.equal(x.torch(), torch.tensor([0.1, 0.2, 0.3, 0.4])))

    def test_add(self):
        # Test adding two quaternions
        q = [0.1, 0.2, 0.3, 0.4]
        x = QuaternionTensor(q)
        y = pyquaternion.Quaternion(*q)
        self.assertTrue(torch.allclose(x + x, torch.Tensor((y + y).q))) 
    
    def test_scalar_add(self):
        # Test adding a quaternion and a scalar
        q = [0.1, 0.2, 0.3, 0.4]
        x = QuaternionTensor(q)
        y = pyquaternion.Quaternion(*q)
        self.assertTrue(torch.allclose(x + 1, torch.Tensor((y + 1).q)))

    def test_mul(self):
        # Test multiplying two quaternions
        q1 = [0.1, 0.2, 0.3, 0.4]
        q2 = [1, 2, 3, 4]

        x1 = QuaternionTensor(q1)
        x2 = QuaternionTensor(q2)

        y1 = pyquaternion.Quaternion(*q1)
        y2 = pyquaternion.Quaternion(*q2)
        self.assertTrue(torch.allclose(x1*x2, torch.Tensor((y1*y2).q))) 

    def test_div(self):
        # Test dividing two quaternions
        q1 = [0.1, 0.2, 0.3, 0.4]
        q2 = [1, 2, 3, 4]

        x1 = QuaternionTensor(q1)
        x2 = QuaternionTensor(q2)

        y1 = pyquaternion.Quaternion(*q1)
        y2 = pyquaternion.Quaternion(*q2)
        self.assertTrue(torch.allclose(x1/x2, torch.Tensor((y1/y2).q)))

    def test_norm(self):
        # Test calculating notm of a quaternion
        q = [0.1, 0.2, 0.3, 0.4]
        x = QuaternionTensor(q)
        y = pyquaternion.Quaternion(*q)
        
        self.assertTrue(torch.allclose(x.norm(), torch.Tensor([y.norm]))) 

    def test_pow(self):
        # Test raising quaternion to power
        q = [0.1, 0.2, 0.3, 0.4]
        x = QuaternionTensor(q)
        y = pyquaternion.Quaternion(*q)
        
        self.assertTrue(torch.allclose(x**2, torch.Tensor((y**2).q))) 

    def test_exp(self):
        # Test quaternion exponential
        q = [0.1, 0.2, 0.3, 0.4]
        x = QuaternionTensor(q)
        y = pyquaternion.Quaternion(*q)
        
        self.assertTrue(torch.allclose(x.exp(), torch.Tensor((pyquaternion.Quaternion.exp(y).q)))) 

if __name__ == '__main__':
    unittest.main()