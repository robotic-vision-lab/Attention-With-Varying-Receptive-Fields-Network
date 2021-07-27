import os
def cd(directory):
  os.chdir(directory)
  os.listdir()
def ls(directory):
  files = os.listdir(directory)
  print(files)
def ls():
  files = os.listdir()
  print(files)
  
