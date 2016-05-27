'''
Created on May 27, 2016

@author: isvoboda
'''

from Queue import Full, Empty

class RoundBuffer(object):
    """
    Round buffer with random access memory.
    """

    def __init__(self, max_size=5):
        self.max_size = max_size
        self.i_index = 0
        self.size = 0
        self.buffer = [None] * self.max_size
        self.it_id = 0

    def append(self, obj):
        """
        Append the obj to the RoundBuffer.
        """
        if self.size == self.max_size:
            raise Full  # "RoundBuffer full, invalid attempt to assign a value."

        self.buffer[self.i_index % self.max_size] = obj
        self.i_index += 1
        self.size += 1

    def append_round(self, obj):
        """
        Append the item obj and if necessary pop the first in item.
        """
        ret = None
        if self.size == self.max_size:
            ret = self.pop()
        self.append(obj)
        return ret

    def pop(self):
        """
        Pop the first in obj.
        """
        if self.size == 0:
            raise Empty  # "RoundBuffer empty, invalid attempt to pop a value."
        ret = self[0]
        self.size -= 1
        return ret

    def __getitem__(self, key):
        if 0 > key or self.size < key:
            raise IndexError  #"RoundBuffer attempt to access key out of bounds."
        i = (self.i_index - self.size + key) % self.max_size
        return self.buffer[i]

    def __setitem__(self, key, value):
        if 0 > key or self.size < key:
            raise IndexError  #"RoundBuffer attempt to set key out of bounds."
        i = (self.i_index - self.size + key) % self.max_size
        self.buffer[i] = value

    def __iter__(self):
        self.it_id = 0
        return self

    def next(self):
        """
        Return next obj.
        """
        if self.it_id == self.size:
            raise StopIteration
        key = self.it_id
        self.it_id += 1
        return self.__getitem__(key)
