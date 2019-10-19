from pymba import Vimba
from pymba import VimbaException
from pymba import Frame

import cv2
import sys
import traitlets
import getpass
import atexit

class Foo(traitlets.HasTraits):
    value = traitlets.Any()
    bar = traitlets.Int()
    baz = traitlets.Unicode()
    class_val = 0

    def __init__(self, init_value):
        self.class_val = init_value

    def set_value(self, input):
        self.value = input

    @traitlets.observe('value')
    def _observe_value(self, change):
        print("class value change")
        #print("value = ", value)
        print(change['old'])
        print(change['new'])
        print("self.value = ", self.value)

    @traitlets.observe('baz')
    def _observe_bar(self, change):
        #pass
        print("class func")
        #self.value = self.baz
        print(change['old'])
        print(change['new'])

def func(change):
    print("callback")
    print(change['old'])
    print(change['new'])  # as of traitlets 4.3, one should be able to
    # write print(change.new) instead

def exit_func1():
    print("exit_func1")
def exit_func2():
    print("exit_func2")

def main():
    foo = Foo(2)
    foo.observe(func, names=['bar'])
    atexit.register(exit_func1)
    atexit.register(exit_func2)

    #raise RuntimeError('Could not read image from camera.')
    #raise OSError("No camera present.")

    foo.bar = 1  # prints '0\n 1'
    foo.baz = 'abc'  # prints nothing
    foo.bar = 4
    foo.set_value(100)
    foo.value = 111
    print(foo.class_val)
    print(Foo.class_val)

    AA = Foo(20)
    AA.set_value(300)
    AA.value = 400
    print(AA.class_val)
    print(Foo.class_val)

class A(traitlets.HasTraits):
    value = traitlets.Any()


if __name__ == '__main__':
    main()
