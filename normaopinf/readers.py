import os
import numpy as np
def get_timestamp_str(filename):
  # Naming convenction is 01-time-xxx.csv
  #Split at the ".csv" and grab bit before the . 
  tmp = filename.split('.')[-2]
  # split at the dash and get final string
  tmp = tmp.split('-')
  time = tmp[-1]
  return time

def get_timestamp(filename):
  # Naming convenction is 01-time-xxx.csv
  #Split at the ".csv" and grab bit before the . 
  tmp = filename.split('.')[-2]
  # split at the dash and get final string
  tmp = tmp.split('-')
  tmp = tmp[-1]
  time = int(tmp)
  return time

def _load_csv_files(field: str, solution_directory : str, base_name : str, skip_files=1,verbose=True, return_time = True) -> (np.ndarray,np.ndarray):
    '''
    Function to load displacement,velocity or acceleration values and put into snapshot tensor

    Args:
        field (string): disp, velo, or acce 
        solution_directory (string): path to the directory where the .csv files are stored 
        solution_id (int): domain ID for which to collect csv files 
        skip_files (int): frequency at which to load the snapshots
        verbose (bool): bool on print statements 
        return_time (bool): bool on if to return associated time stamps  

   Returns:
        snapshots (np.ndarray): snapshot tensor of shape (3,n_nodes,n_snapshots)
    '''
    string_to_match = base_name + "-" + field + "-"
    list_of_files = [solution_directory +'/' + f for f in os.listdir(solution_directory) if (string_to_match in f)]
    list_of_files.sort(key=get_timestamp)

    string_to_match = base_name + "-time-"
    #string_to_match = "-" + str(solution_id) + "-time-"
    list_of_files.sort(key=get_timestamp)
    if verbose:
      print(r"Found " + str(len(list_of_files)) + " files for field: ", field) 
    list_of_files = list_of_files[::skip_files] 
    snapshots = None
    time_snapshots = np.zeros(0) 
    for i,file in enumerate(list_of_files):
        sol = np.genfromtxt(file,delimiter=',')
        if snapshots is None:
            snapshots = sol.transpose()[...,None] 
        else:
            snapshots = np.append(snapshots,sol.transpose()[...,None],axis=2)

        # if return associated time stamps 
        if return_time:
          timestamp = get_timestamp_str(file) 
          string_to_match =  base_name + "-time-" + str(timestamp) + '.csv'
          #string_to_match =  str(solution_id) + "-time-" + str(timestamp) + '.csv'
          list_of_files = [solution_directory +'/' + f for f in os.listdir(solution_directory) if string_to_match in f]
          assert len(list_of_files) == 1
          #file_start = file.split('/')[-1].split('-')[0]
          #time_filename = solution_directory +'/' + file_start + '-' + str(solution_id) + '-time-' + timestamp + '.' + file.split('.')[-1] 
          time_filename = list_of_files[0]
          time = np.genfromtxt(time_filename)
          time_snapshots = np.append(time_snapshots,time)
    if verbose:
      print(r"Returning snapshots tensor of shape " + str(snapshots.shape)) 

      
    return snapshots,time_snapshots

def load_velocity_csv_files(solution_directory : str, base_name : str, skip_files=1,verbose=True, return_time = True) -> (np.ndarray,np.ndarray):
    '''
    Function to load displacement values and put into snapshot tensor

    Args:
        solution_directory (string): path to the directory where the .csv files are stored 
        solution_id (int): domain ID for which to collect csv files 
        skip_files (int): frequency at which to load the snapshots
        verbose (bool): bool on print statements 
        return_time (bool): bool on if to return associated time stamps  

   Returns:
        snapshots (np.ndarray): snapshot tensor of shape (3,n_nodes,n_snapshots)
    '''
    snapshots,time_snapshots = _load_csv_files('velo',solution_directory,base_name, skip_files,verbose,return_time)
    return snapshots,time_snapshots


def load_displacement_csv_files(solution_directory : str, base_name : str, skip_files=1,verbose=True, return_time = True) -> (np.ndarray,np.ndarray):
    '''
    Function to load displacement values and put into snapshot tensor

    Args:
        solution_directory (string): path to the directory where the .csv files are stored 
        solution_id (int): domain ID for which to collect csv files 
        skip_files (int): frequency at which to load the snapshots
        verbose (bool): bool on print statements 
        return_time (bool): bool on if to return associated time stamps  

   Returns:
        snapshots (np.ndarray): snapshot tensor of shape (3,n_nodes,n_snapshots)
    '''
    snapshots,time_snapshots = _load_csv_files('disp',solution_directory,base_name , skip_files,verbose,return_time)
    return snapshots,time_snapshots

def load_acceleration_csv_files(solution_directory : str, base_name : str, skip_files=1,verbose=True, return_time = True) -> (np.ndarray,np.ndarray):
    '''
    Function to load displacement values and put into snapshot tensor

    Args:
        solution_directory (string): path to the directory where the .csv files are stored 
        solution_id (int): domain ID for which to collect csv files 
        skip_files (int): frequency at which to load the snapshots
        verbose (bool): bool on print statements 
        return_time (bool): bool on if to return associated time stamps  

   Returns:
        snapshots (np.ndarray): snapshot tensor of shape (3,n_nodes,n_snapshots)
    '''
    snapshots,time_snapshots = _load_csv_files('acce',solution_directory,base_name , skip_files,verbose,return_time)
    return snapshots,time_snapshots


def load_sideset_displacement_csv_files(solution_directory: str, sidesets: list, base_name : str, skip_files=1,verbose=True) -> dict:
    '''
    Function to load displacement values on sidesets and put into snapshot tensor

    Args:
        solution_directory (string): path to the directory where the .csv files are stored 
        sidesets (list): list containing names of sidesets (should be the same as in the yaml file)
        solution_id (int): domain ID for which to collect csv files 
        skip_files (int): frequency at which to load the snapshots
        verbose (bool): bool on print statements 
    Returns:
        sideset_snapshots (dict): dictionary containing sideset snapshots for each sideset 
    '''
    sideset_snapshots = {}
    for sideset in sidesets:
        sideset_snapshots[sideset] = None
        string_to_match = base_name + "-" + sideset + "-disp-"
        #string_to_match = str(solution_id) + "-" + sideset + "-disp-"
        list_of_files = [solution_directory +'/' + f for f in os.listdir(solution_directory) if string_to_match in f]
        list_of_files.sort(key=get_timestamp)
        if verbose:
           print(r"Found " + str(len(list_of_files)) + " displacement files for sideset " + sideset) 
        list_of_files = list_of_files[::skip_files] 
        for file in list_of_files:
            sol = np.genfromtxt(file,delimiter=',')

            # Add axis if one dimensional 
            if sol.ndim == 1:
              sol = sol[:,None]

            if sideset_snapshots[sideset] is None:
                sideset_snapshots[sideset] = sol.transpose()[...,None]
            else:
                sideset_snapshots[sideset] = np.append(sideset_snapshots[sideset],sol.transpose()[...,None],axis=2)
    if verbose:
        for sideset in sidesets:
            print(r"sideset " + sideset + " snapshots are of size " + str(sideset_snapshots[sideset].shape)) 

    return sideset_snapshots


def get_free_dofs(solution_directory: str, base_name : str) -> np.ndarray:
    '''
    Function to load bool output stating if DOFs are free or fixed

    Args:
        snapshot_tensor (np.ndarray): snapshot tensor of shape (3,n_nodes,n_snapshots)
        solution_directory (string): path to the directory where the .csv files are stored 
        solution_id (int): domain ID for which to collect csv files 
    Returns:
        free_dofs (np.ndarray): matrix of shape (3,n_nodes). True means the dof is free
    '''
    #old_string_to_match = str(solution_id).zfill(2) + "-free_dofs-"
    #string_to_match = "-" + str(solution_id) + "-free_dofs-"
    string_to_match = base_name + "-free_dofs-"
    list_of_files = [solution_directory +'/' + f for f in os.listdir(solution_directory) if (string_to_match in f)]
    list_of_files.sort(key=get_timestamp)
    free_dofs = np.genfromtxt(list_of_files[0],dtype=bool)
    N = free_dofs.size
    N_by_three = int(N/3)
    free_dofs = np.reshape(free_dofs,(3,N_by_three),'F') 
    return free_dofs
