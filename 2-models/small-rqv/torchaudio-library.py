import ctypes

path = r"C:\Users\julio\anaconda3\envs\lyrics2music\Lib\site-packages\torchaudio\lib\libtorchaudio.pyd"

try:
    ctypes.CDLL(path)
    print("library load successfully!")
except Exception as e:
    print("error: ", e)