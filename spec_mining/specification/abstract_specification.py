from abc import ABCMeta, abstractmethod


class Specification(object, metaclass=ABCMeta):

    @abstractmethod
    def merge(self, spec, Xable):
        """

        :param Xable:
        :param spec:
        :return:
        """

    @abstractmethod
    def to_CTL(self):
        """

        :return:
        """

    @abstractmethod
    def to_LTL(self):
        """

        :return:
        """