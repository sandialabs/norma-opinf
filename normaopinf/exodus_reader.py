### MUST HAVE EXODUS LOADED AS A MODULE
### script developed and tested with sparc-tools/exodus/2021.11.26
## Known issue: import h5py can conflict with import exodus3 as exodus
import os
import sys
import numpy as np
import exodus
import copy
import scipy.interpolate

class Suppressor(object):
    """
    Suppresses stdout.  Useful to avoid exopy printing hundereds of copyright notices
    """
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            raise

    def write(self, x): pass



def getProcNumber(fileName):
    procNumber = int( fileName.split('.')[-1] )
    return procNumber 

def load_exodus_file(filename,steady=False):
  if len(filename.split('/')) == 1:
    solution_directory = './'
  else:
    solution_directory = filename.split('/')
    solution_directory = '/'.join(solution_directory[0:-1]) + '/'
  filename = filename.split('/')[-1]
  filesList = [solution_directory +'/' + f for f in os.listdir(solution_directory) if filename in f]
  if len(filesList) > 1:
    filesList.sort(key=getProcNumber)
    nProcs = np.size(filesList)
  else:
    nProcs = 1
  dataDict = {}
  i = 0
  for fileName in filesList:
    if (steady):
      dataDict = loadSingleExodusFileAtFinalStep(fileName,dataDict)
    else:
      dataDictLocal = loadUnsteadyExodusFile(fileName)
      for key in dataDictLocal.keys():
          if key in dataDict.keys():
              dataDict[key] = np.append(dataDict[key],dataDictLocal[key],axis=1)

          else:
              dataDict[key] = copy.deepcopy(dataDictLocal[key])

      i += 1
  return dataDict



def interpolate_file(exo_file,new_times):
    new_dict = {}
    for key in list(exo_file.keys()):
       interpolator = scipy.interpolate.interp1d(exo_file['time'],exo_file[key].transpose())
       interpolated_val = interpolator(new_times)
       new_dict[key] = interpolated_val.transpose()
    return new_dict


def compute_errors(exo_file_a,exo_file_b):
   '''
   Method to compute errors for elements between two different exodus files
   Inputs:
       exo_file_a: exodus file loaded via load_exodus_file
       exo_file_b: exodus file loaded via load_exodus_file
   Returns:
       errors: dictionary containing absolute and relative errors for shared keys
               Relative error is computed w.r.p. to the second input

   '''
   ## get common elements
   #assert np.allclose(exo_file_a['time'], exo_file_b['time']), " Currently only support equal time grids" 
   common_keys = []
   keys = list(exo_file_a.keys())
   for key in list(exo_file_b.keys()):
     if key in keys:
       common_keys.append(key) 
   
   errors = {}
   ## If on the same time grid
   if len(exo_file_a['time']) == len(exo_file_b['time']):
     if np.allclose(exo_file_a['time'], exo_file_b['time']):
       for key in common_keys:
           errors[key] = np.linalg.norm(exo_file_a[key] - exo_file_b[key])
           errors['rel-' + key] = errors[key] / np.linalg.norm(exo_file_b[key])

   if len(exo_file_a['time']) != len(exo_file_b['time']) or np.allclose(exo_file_a['time'], exo_file_b['time']) == False:
       ## If on different time grids, restrict to the coarser grid
       average_dt_a = np.mean(np.diff(exo_file_a['time']))
       average_dt_b = np.mean(np.diff(exo_file_b['time']))
       if average_dt_a < average_dt_b:
           coarse_file = exo_file_b
           fine_file = exo_file_a
       else:
           fine_file = exo_file_b
           coarse_file = exo_file_a

       for key in common_keys:
           print(key,fine_file[key].shape,fine_file['time'].shape)
           interpolator = scipy.interpolate.interp1d(fine_file['time'],fine_file[key].transpose())
           interpolated_val = interpolator(coarse_file['time'])
           errors[key] = np.linalg.norm(coarse_file[key] - interpolated_val.transpose())
           if average_dt_a < average_dt_b:
               errors['rel-' + key] = errors[key] / np.linalg.norm(coarse_file[key])
           else:
               errors['rel-' + key] = errors[key] / np.linalg.norm(interpolated_val) 
   return errors

def compute_list_of_errors(exo_files,exo_file_b):

  for i,exo_file in enumerate(exo_files):
    errors = compute_errors(exo_file,exo_file_b)

    if i == 0:
      all_errors = {}
      for key in errors.keys():
        all_errors[key] = np.zeros(0)

    for key in list(errors.keys()):
      all_errors[key] = np.append(all_errors[key],errors[key])

  return all_errors


def computeMean(x,y,z):
  X = np.zeros((np.size(x),3))
  X[:,0] = x
  X[:,1] = y
  X[:,2] = z
  return np.mean(X,axis=0)


def loadSingleExodusFileAtFinalStep(filename,dataDict={}):
    # create exodus object to read exodus files
    with Suppressor():
        exo = exodus.exodus(filename,array_type='numpy')

    # get list of block ids and variable names 
    nSteps = exo.num_times() 
    dataDict = loadSingleExodusFileAtStep(exo,nSteps)

    with Suppressor():
        exo.close()
    return dataDict 



def loadSingleExodusFileAtStep(exo,step,dataDict={}):
    element_var_names = exo.get_element_variable_names()
    nodal_var_names = exo.get_node_variable_names()
    num_var=len(nodal_var_names)

    snapshots=None

    x,y,z = exo.get_coords()

    # Check number of elements to see if this file has any for the current block
    for variable in nodal_var_names:
      loadedData = exo.get_node_variable_values(variable,step).flatten('F')
      if variable in dataDict.keys():
        dataDict[variable] = np.append(dataDict[variable],loadedData)
      else:
        dataDict[variable] = loadedData
    return dataDict 


def loadUnsteadyExodusFile(filename):
    # create exodus object to read exodus files
    with Suppressor():
      exo = exodus.exodus(filename,array_type='numpy')
    nSteps = exo.num_times() 
    tmp = {}
    dataDict = loadSingleExodusFileAtStep(exo,1,tmp)
    nSols = 1
    for step in range(2,nSteps+1):
      tmp = {}
      dataDictTmp = loadSingleExodusFileAtStep(exo,step,tmp)
      nSols += 1
      for key in list(dataDict.keys()):
        if step == 2:
          dataDict[key] = dataDict[key][None]
        dataDict[key] = np.append(dataDict[key],dataDictTmp[key][None],axis=0)

    times = exo.get_times()[:]
    dataDict['time'] = times
    with Suppressor():
      exo.close()
    
    return dataDict

