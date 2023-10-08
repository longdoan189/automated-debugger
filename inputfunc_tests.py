from inputfunc import *

def test_1():
    """+"""
    assert inputfunc("hello","y","a") == "fcjjm"

def test_2():
    """+"""
    assert inputfunc("hola","k","b") == "ryvk"

def test_3():
    """+"""
    assert inputfunc("bbb","a","c") == "bbb"

def test_4():
    """-"""
    assert inputfunc("hola","o","v") == "vzo"

def test_5():
    """-"""
    assert inputfunc("hello","l","v") == "spz"

def test_6():
    """-"""
    assert inputfunc("bbb","c","b") == ""
