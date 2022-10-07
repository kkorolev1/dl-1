import numpy as np
import scipy.special

from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        out = np.zeros(input.shape)
        out[input > 0] = 1
        return grad_output * out


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * self.compute_output(input) * (1 - self.compute_output(input))


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return np.exp(input) / np.exp(input).sum(axis=1).reshape(-1, 1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = self.compute_output(input)

        # Решение через подсчет тензоров
        #jacobian1 = np.einsum('ij,jk->ijk', softmax, np.eye(softmax.shape[1]))
        #jacobian2 = np.einsum('ij,ik->ijk', softmax, softmax)
        #return np.einsum("ij,ijk->ik", grad_output, jacobian1 - jacobian2)

        # Заметим, что можно проще
        # y * (D - v^t * v) = y * D - (y * v^t) * v
        return grad_output * softmax - softmax * (grad_output * softmax).sum(axis=1).reshape(-1, 1)

class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = Softmax().compute_output(input)

        #jacobian2 = np.einsum('ij,ik->ijk', np.ones(softmax.shape), softmax)
        #return np.einsum("ij,ijk->ik", grad_output, np.eye(softmax.shape[1]) - jacobian2)

        return grad_output - softmax * grad_output.sum(axis=1).reshape(-1, 1)

