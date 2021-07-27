from tkinter import *
import os
from tkinter import ttk
from tkinter import filedialog
import json
import pdb
class SelectDirectory(Frame):
  def __init__(self, parent = None, folderName = "", data = None,**kw):
    Frame.__init__(self, master = parent, **kw)
    self.folderPath = StringVar()
    self.labelName = Label(self, text = folderName)
    self.labelName.grid(row = 0, column = 0)
    self.entPath = Entry(self, textvariable = self.folderPath)
    self.entPath.grid(row = 0, column = 1)
    self.button = ttk.Button(self, text = "Set and Reset", command =self.set_dir)
    self.button.grid(row = 0, column = 2)
    self.addButton = ttk.Button(self, text = "Add Directory", command =self.add_dir)
    self.addButton.grid(row = 0, column = 3)
    if data != None: self.folderPath.set(','.join(data))
    self.counter = 0
  def set_dir(self):
    selected_directory = filedialog.askdirectory()
    self.folderPath.set(selected_directory)
  def add_dir(self):
    selected_directory = filedialog.askdirectory()
    old_dir = self.folderPath.get()
    if self.counter != 0:self.folderPath.set(old_dir + "," + selected_directory)
    else: self.folderPath.set( selected_directory)
    self.counter +=1
  def get_folder_path(self):
    return self.folderPath.get() 

def save_folders_to_json(dataset_dirs,data_names, dir_names, filename, gui):

  json_to_save = {}
  for dataset_dir, data_name in zip(dataset_dirs,data_names):
    json_to_save[data_name] = {}
    for folder_path, dir_name in zip(dataset_dir, dir_names):
      json_to_save[data_name][dir_name] = folder_path.get_folder_path().split(",")
    
  json.dump(json_to_save,open(filename, 'w'))
  gui.destroy()
  


def select_dirs(data_names,dir_names,  filename):
  """
  Example:
  select_dirs(['thermal', 'div2k'],['test_val', 'val_dir'] ,'myjson.json') 
  """
  gui = Tk()
  gui.geometry("400x600")
  dataset_dirs = []
  row = 0
  if os.path.exists(filename):
      data = json.load(open(filename,'r'))
  else: data = None
  for j,data_name in enumerate(data_names):
    directories = []
    labelName = Label(gui ,text = data_name)
    labelName.grid(row = row, column = 0)
    row +=1
    for i,dir_name in enumerate(dir_names):
      thisdata = data[data_name][dir_name] if data is not None else None
      dir_select = SelectDirectory(gui, dir_name, data = thisdata)
      dir_select.grid(row = row)
      directories.append(dir_select)
      row += 1
    dataset_dirs.append(directories)
  save = ttk.Button(gui, text = "save", command = lambda:
save_folders_to_json(dataset_dirs,data_names, dir_names, filename, gui))
 
  save.grid(row = row, column = 0)
  gui.mainloop()
  

  
  
                        
