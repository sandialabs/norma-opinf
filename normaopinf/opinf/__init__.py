import numpy as np
import opinf
import sys
import os
from matplotlib import pyplot as plt
import normaopinf
import normaopinf.readers
import normaopinf.calculus
import normaopinf.parser
import romtools
import scipy.optimize
import argparse
import normaopinf.opinf.models as opinf_models
import nnopinf
import nnopinf.operators
import nnopinf.models
import nnopinf.training

# ANSI escape codes for bold and blue text
BOLD_BLUE ='\033[0;34m'  # Bold blue
RESET = '\033[0m'         # Reset to default

def reshape_snapshots(snapshots):
    snapshots = np.reshape(snapshots,(snapshots.shape[0],np.prod(snapshots.shape[1::])))
    return snapshots 

def train_model(opinf_settings,output_dir,ensemble_id,uhat,uhat_ddots,uhat_sidesets,initialization_model=None):
    rom_dim = uhat.shape[0]
    n_hidden_layers = 3
    n_neurons_per_layer = rom_dim
    n_inputs = rom_dim
    n_outputs = rom_dim

    ## Design operators for the state
    NpdMlp = nnopinf.operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)

    #SkewMlp = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    #MatrixMlp = operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,(n_outputs,n_outputs))
    #CompositeOperator = operators.CompositeOperator([NpdMlp])

    # Create models for sidesets and wrap for OpInf
    my_operators = []
    sidesets = list(uhat_sidesets.keys())
    inputs = {}
    for sideset in sidesets:
      inputs["u-" + sideset] = reshape_snapshots(uhat_sidesets[sideset]).transpose() 
      #print(uhat_sidesets[sideset].shape)
      n_sideset_inputs = uhat_sidesets[sideset].shape[0]
      op = nnopinf.operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,n_sideset_inputs,(n_outputs,n_sideset_inputs))
      my_operators.append( nnopinf.models.WrappedOperatorForModel(operator=op,inputs=("u-" + sideset,),name="BC-" + sideset + '-' + str(ensemble_id))  )

    stiffness_operator =  nnopinf.models.WrappedOperatorForModel(operator=NpdMlp,inputs=("x",),name="stiffness-" + str(ensemble_id))
    my_operators.append(stiffness_operator)

    my_model = nnopinf.models.OpInfModel( my_operators )
    if initialization_model is not None:
        my_model.hierarchical_update(initialization_model)

    #Construct training data
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir)

    training_settings = opinf_settings['neural-network-training-settings']

    print('hi',uhat.shape)
    uhat = reshape_snapshots(uhat) 
    inputs['x'] = uhat.transpose()
    nnopinf.training.train(my_model,input_dict=inputs,y=reshape_snapshots(uhat_ddots).transpose(),training_settings=training_settings)














def formatRed(skk): 
  return "\033[91m {}\033[00m" .format(skk)

def get_acceptable_opinf_settings():
  acceptable_settings = {}
  #OpInf model type 
  acceptable_settings['model-type'] = ['linear','quadratic','cubic','symmetric linear','neural-network']

  #If OpInf model has forcing
  acceptable_settings['forcing'] = [True,False]

  # How we truncate the bases for the main domain
  acceptable_settings['truncation-type'] = ['size','energy']

  # How we truncate the bases for the boundary
  acceptable_settings['boundary-truncation-type'] = ['size','energy']

  # If we make per DOF bases, or if we split. This is applied to both sideset and
  # volume snapshots
  acceptable_settings['trial-space-splitting-type'] = ['split','combined']

  # How we compute the acceleration (either from snapshots or finite difference of displacement)
  acceptable_settings['acceleration-computation-type'] = ['acceleration-snapshots','finite-difference']
  return acceptable_settings



def check_valid_settings_for_getting_snapshots(opinf_settings):
  required_keys = ['fom-yaml-file','training-data-directories','training-skip-steps','stop-training-time']

  # Check that keys exist
  opinf_keys = list(opinf_settings.keys())
  for key in required_keys:
      assert key in opinf_keys, formatRed("Error processing OpInf settings: required key " + str(key) + " was not found")

  # Check that keys types are correct
  assert isinstance(opinf_settings['fom-yaml-file'],str) ,"fom-yaml-file must be a string"
  assert isinstance(opinf_settings['training-data-directories'],list), "training-data-directories must be a list"
  assert isinstance(opinf_settings['training-skip-steps'],int), "training-skip-steps must be an int"
  assert (isinstance(opinf_settings['stop-training-time'],float) or opinf_settings['stop-training-time'] == 'end'), "stop-training-time must be a float or 'end'"



def check_valid_settings(opinf_settings):
  required_keys = ['fom-yaml-file','training-data-directories','training-skip-steps','model-type','forcing','truncation-type','truncation-value','boundary-truncation-type',
                   'boundary-truncation-value','regularization-parameter','model-name','trial-space-splitting-type','acceleration-computation-type']

  opinf_keys = list(opinf_settings.keys())
  for key in required_keys:
      assert key in opinf_keys, formatRed("Error processing OpInf settings: required key " + str(key) + " was not found")

  acceptable_settings = get_acceptable_opinf_settings()

  for key in list(acceptable_settings.keys()):
      assert opinf_settings[key] in acceptable_settings[key], formatRed("Error processing OpInf settings: key " + str(opinf_settings[key]) + " is not valid \n" +  "Acceptable options are: " + str(acceptable_settings[key]))
    
                                                             



def get_example_opinf_settings():
  settings = {}
  # Name of FOM yaml file from which to create OpInf ROM
  settings['fom-yaml-file'] = "torsion-2.yaml"
  # List of paths to where the training data is
  settings['training-data-directories'] = [os.getcwd()]

  # how many files to skip when loading training data, e.g., [files = files[::skip]] 
  settings['training-skip-steps'] = 1

  # Type of ROM to create
  settings['model-type'] = 'linear' 

  # If ROM has forcing
  settings['forcing'] =  False

  # How to truncate the main domain
  settings['truncation-type'] = 'energy' 
  settings['truncation-value'] = 0.9999                      

  # How to truncate the boundary
  settings['boundary-truncation-type'] =  'energy' 
  settings['boundary-truncation-value'] = 0.9999

  # How to regularize ('automatic' or float for regularization parameter)
  settings['regularization-parameter'] =  'automatic'

  # How to split the trial space 
  settings['trial-space-splitting-type'] = 'split'

  # How to compute the acceleration target (acceleration-snapshots or finite-difference)
  settings['acceleration-computation-type'] = 'acceleration-snapshots'

  # Name of the operator
  settings['model-name'] = 'opinf-operator'
  return settings



def get_default_neural_network_settings():
  settings = {}


###In development
def non_parametric_fit_with_grid_search_beta(opinf_model: opinf.models.ContinuousModel, 
                                        x: np.ndarray, xdot: np.ndarray, xddot: np.ndarray,
                                        bcs : np.ndarray, times: np.ndarray , regularization_parameters_to_try:np.ndarray = np.logspace(-5,5,5)):

   print(f"Performing grid search")
   ## Store errors for different regularization parameters
   errors = np.zeros(regularization_parameters_to_try.size**3)

   # Create an extension of the time window to run ROMs into the future
   extend_window_ratio = 1
   times_extended = times*1
   dt = times[1] - times[0]
   n_steps = times.size
   for i in range(1,extend_window_ratio):
     time_window = times_extended[-1] + (times - times[0]) + dt 
     times_extended = np.append(times_extended,time_window)
   assert(np.allclose(times_extended[0:times.size],times))

   # Loop over regularization parameters, fit, and test
   for counter,regularization_parameter in enumerate(regularization_parameters_to_try):
     for counter2,regularization_parameter2 in enumerate(regularization_parameters_to_try):
       for counter3,regularization_parameter3 in enumerate(regularization_parameters_to_try):
         # Create regularization solver
         if opinf_model.H_ is None: 
           solver = opinf.lstsq.L2Solver(regularizer=regularization_parameter)
         else:
           solver = opinf.lstsq.L2DecoupledSolver(regularizer=[regularization_parameter,regularization_parameter2,regularization_parameter3])
         opinf_model.solver = solver
    
         # Fit     
         opinf_model.fit(states=x, ddts=xddot,inputs=bcs)
         u0 = x[:,0]
    
         # Create wrapper for forward evaluation of the model
         if opinf_model.H_ is None: 
             opInfForwardModel = normaopinf.opinf.models.LinearOpInfRom( opinf_model.A_.entries, opinf_model.B_.entries)
         elif opinf_model.G_ is None:
             opInfForwardModel = normaopinf.opinf.models.QuadraticOpInfRom( opinf_model.A_.entries, opinf_model.B_.entries,opinf_model.H_.expand_entries(opinf_model.H_.entries),opinf_model)
         else:
             opInfForwardModel = normaopinf.opinf.models.CubicOpInfRom( opinf_model.A_.entries, opinf_model.B_.entries,opinf_model.H_.expand_entries(opinf_model.H_.entries),opinf_model.G_.expand_entries(opinf_model.G_.entries), opinf_model)

         def bc_hook(step):
           step_to_get = min(step,bcs.shape[1] - 1)
           return bcs[:,step_to_get]
    
         # Test forward simulation
         test_states = opInfForwardModel.advance_n_steps_newmark( x[:,0], xdot[:,0], xddot[:,0], dt,int( n_steps * extend_window_ratio) ,bc_hook)
    
         # Check if we blew up
         if (np.any(np.isnan(test_states)) or  np.any(np.abs(test_states) > 1e5)):
           print('Dedected NaN in solution')
           errors[counter] = 1e10
         else:
           errors[counter] = np.linalg.norm(test_states[:,0:times.size] - x) / np.linalg.norm(x)
    
         print(f"Error: {errors[counter]:.4e}, Regularization Parameter 1: {regularization_parameter:.4e}, Regularization Parameter 2: {regularization_parameter2:.4e}, Regularization Parameter 3: {regularization_parameter3:.4e}")
         counter += 1
   optimal_case = np.nanargmin(errors)
   optimal_regularization_parameter = regularization_parameters_to_try[optimal_case]
   best_error = np.nanmin(errors)
   print('Best regularization parameter = ' + str(optimal_regularization_parameter))
   print('Error = ' + str(best_error))
   if opinf_model.H_ is None: 
     solver = opinf.lstsq.L2Solver(regularizer=optimal_regularization_parameter)
   else:
     solver = opinf.lstsq.L2Solver(regularizer=optimal_regularization_parameter)
     #solver = opinf.lstsq.L2DecoupledSolver(regularizer=[optimal_regularization_parameter,1000.*optimal_regularization_parameter])

   opinf_model.solver = solver
   opinf_model.fit(states=x, ddts=xddot,inputs=bcs)
   return opinf_model




def non_parametric_fit_with_grid_search(opinf_model: opinf.models.ContinuousModel, 
                                        x: np.ndarray, xdot: np.ndarray, xddot: np.ndarray,
                                        bcs : np.ndarray, times: np.ndarray , regularization_parameters_to_try:np.ndarray = np.logspace(-4,0,40)):

   print(f"Performing grid search")
   ## Store errors for different regularization parameters
   errors = np.zeros(len(regularization_parameters_to_try))

   # Create an extension of the time window to run ROMs into the future
   extend_window_ratio = 1
   times_extended = times*1
   dt = times[1] - times[0]
   n_steps = times.shape[0]
   for i in range(1,extend_window_ratio):
     time_window = times_extended[-1] + (times - times[0]) + dt 
     times_extended = np.append(times_extended,time_window)
   assert(np.allclose(times_extended[0:times.size],times))

   # Loop over regularization parameters, fit, and test
   for counter,regularization_parameter in enumerate(regularization_parameters_to_try):
     opinf_model.set_solver(regularization_parameter)

     # Fit     
     opinf_model.fit(states=x, ddts=xddot,inputs=bcs)

     n_cases = x.shape[-2]
     errors[counter] = 0.
     for i in range(0,n_cases):

         # Create wrapper for forward evaluation of the model
         if isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricOpInfModel):
           opInfForwardModel = normaopinf.opinf.models.LinearOpInfRom(   -opinf_model.get_stiffness_matrix(), opinf_model.get_exogenous_input_matrix())

         elif isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricQuadraticOpInfModel): 
           opInfForwardModel = normaopinf.opinf.models.QuadraticOpInfRom( -opinf_model.get_stiffness_matrix(), opinf_model.get_exogenous_input_matrix(),-opinf_model.get_quadratic_stiffness_matrix(),opinf_model)

         elif isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricCubicOpInfModel): 
           opInfForwardModel = normaopinf.opinf.models.CubicOpInfRom( -opinf_model.get_stiffness_matrix(), opinf_model.get_exogenous_input_matrix(),-opinf_model.get_quadratic_stiffness_matrix(),-opinf_model.get_cubic_stiffness_matrix(),opinf_model)
         else:
           print('Model type not found, exiting')
           sys.exit()

         def bc_hook(step):
           step_to_get = min(step,bcs.shape[-1] - 1)
           return bcs[...,i,step_to_get]

         # Test forward simulation
         test_states = opInfForwardModel.advance_n_steps_newmark( x[...,i,0], xdot[...,i,0], xddot[...,i,0], dt[i],int( n_steps * extend_window_ratio) ,bc_hook)

         # Check if we blew up
         if (np.any(np.isnan(test_states)) or  np.any(np.abs(test_states) > 1e5)):
             print('Dedected NaN in solution')
             errors[counter] += 1e10
         else:
             local_error = np.linalg.norm(test_states[:,0:times.shape[0]] - x[:,i,:]) / np.linalg.norm(x[:,i,:])
             errors[counter] += local_error

     print(f"Error: {errors[counter]/n_cases:.4e}, Regularization Parameter: {regularization_parameter:.4e}")
     counter += 1
   optimal_case = np.nanargmin(errors)
   optimal_regularization_parameter = regularization_parameters_to_try[optimal_case]
   best_error = np.nanmin(errors)
   print('Best regularization parameter = ' + str(optimal_regularization_parameter))
   print('Error = ' + str(best_error))
   #   if isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricOpInfModel) or isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricQuadraticOpInfModel): 
   opinf_model.set_solver(optimal_regularization_parameter)
   opinf_model.fit(states=x, ddts=xddot,inputs=bcs)
   return opinf_model,optimal_regularization_parameter



def create_linear_opinf_rom(states,acceleration,times,velocity=None,bcs=None):
    if velocity is None:
      velocity = states*0.
    if bcs is None:
      bcs = np.zeros(states.shape[1])[None]

    opinf_model = opinf.models.ContinuousModel("AB",solver=l2solver)
    opinf_model = non_parametric_fit_with_grid_search(states,velocity,acceleration,bcs,times)
    return opinf_model 


def get_bc_sidesets(input_yaml):
  sidesets = []
  if 'boundary conditions' in input_yaml:
    bc_block = input_yaml['boundary conditions']
    acceptable_bc_types = ['Schwarz overlap','Dirichlet']
    for bc in bc_block:
      if 'Dirichlet' in bc:
        n_bcs = len(bc_block['Dirichlet'])
        for i in range(0,n_bcs):
          nodeset = bc_block['Dirichlet'][i]['node set'] 
          component = bc_block['Dirichlet'][i]['component']
          name = nodeset + '-' + component
          sidesets.append(name)

      if 'Schwarz overlap' in bc:
        n_bcs = len(bc_block['Schwarz overlap'])
        for i in range(0,n_bcs):
          name = bc_block['Schwarz overlap'][i]['side set'] 
          sidesets.append(name)
    print('Found sidesets: ' + str(sidesets))
  return sidesets


 
def get_processed_snapshots(opinf_settings):
    # Check that settings are valid
    check_valid_settings_for_getting_snapshots(opinf_settings)

    n_training_cases = len(opinf_settings['training-data-directories'])

    input_yaml = normaopinf.parser.open_yaml(opinf_settings['fom-yaml-file'])
    base_name = opinf_settings['fom-yaml-file'].split('/')[-1].split('.')[0] 
    # Load in snapshots

    for i in range(0,n_training_cases):    
      cur_dir = opinf_settings['training-data-directories'][i]
      if i ==0:
        displacement_snapshots,times = normaopinf.readers.load_displacement_csv_files(solution_directory=cur_dir,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
        velocity_snapshots,_ = normaopinf.readers.load_velocity_csv_files(solution_directory=cur_dir,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
        acceleration_snapshots,_ = normaopinf.readers.load_acceleration_csv_files(solution_directory=cur_dir,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
        # Identify which DOFs are free
        free_dofs = normaopinf.readers.get_free_dofs(solution_directory=cur_dir,base_name=base_name)
        if opinf_settings['stop-training-time'] != 'end':
          stop_index = np.argmin(np.abs(times - float(opinf_settings['stop-training-time']) ) )
          print('Only training on the first ' + str(stop_index) + ' snapshots')
          displacement_snapshots = displacement_snapshots[...,0:stop_index]
          velocity_snapshots = velocity_snapshots[...,0:stop_index]
          acceleration_snapshots = acceleration_snapshots[...,0:stop_index]
          times = times[0:stop_index]
        # Add extra axis to handle case where we have multiple training directories
        displacement_snapshots = displacement_snapshots[:,:,None,:]
        velocity_snapshots = velocity_snapshots[:,:,None,:]
        acceleration_snapshots = acceleration_snapshots[:,:,None,:]
        times = times[...,None]

      else:
        displacement_snapshots_tmp,times_tmp = normaopinf.readers.load_displacement_csv_files(solution_directory=cur_dir,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
        velocity_snapshots_tmp,_ = normaopinf.readers.load_velocity_csv_files(solution_directory=cur_dir,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
        acceleration_snapshots_tmp,_ = normaopinf.readers.load_acceleration_csv_files(solution_directory=cur_dir,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
        if opinf_settings['stop-training-time'] != 'end':
          displacement_snapshots_tmp = displacement_snapshots_tmp[...,0:stop_index]
          velocity_snapshots_tmp = velocity_snapshots_tmp[...,0:stop_index]
          acceleration_snapshots_tmp = acceleration_snapshots_tmp[...,0:stop_index]
          times_tmp = times_tmp[0:stop_index]

        # Add extra axis to handle case where we have multiple training directories
        displacement_snapshots_tmp = displacement_snapshots_tmp[:,:,None,:]
        velocity_snapshots_tmp = velocity_snapshots_tmp[:,:,None,:]
        acceleration_snapshots_tmp = acceleration_snapshots_tmp[:,:,None,:]
        times_tmp = times_tmp[...,None]

        displacement_snapshots = np.append(displacement_snapshots,displacement_snapshots_tmp,axis=2)
        velocity_snapshots = np.append(velocity_snapshots,velocity_snapshots_tmp,axis=2)
        acceleration_snapshots = np.append(acceleration_snapshots,acceleration_snapshots_tmp,axis=2)
        times = np.append(times,times_tmp,axis=1)

        _ = normaopinf.readers.get_free_dofs(solution_directory=cur_dir,base_name=base_name)
        assert np.allclose(free_dofs,_), "Error, different solution directories have different free DOFs"


    #Get sideset snapshots
    sidesets = get_bc_sidesets(input_yaml)
    if len(sidesets) > 0:
        for i in range(0,n_training_cases):    
            cur_dir = opinf_settings['training-data-directories'][i]
            if i ==0:
                sideset_snapshots = normaopinf.readers.load_sideset_displacement_csv_files(solution_directory=cur_dir,sidesets=sidesets,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
                if opinf_settings['stop-training-time'] != 'end':
                    for sideset in sidesets:
                        sideset_snapshots[sideset] = sideset_snapshots[sideset][...,0:stop_index] 

                # Add extra axis to handle case where we have multiple training directories
                for sideset in sidesets:
                    sideset_snapshots[sideset] = sideset_snapshots[sideset][:,:,None,:] 
            else: 
                sideset_snapshots_tmp = normaopinf.readers.load_sideset_displacement_csv_files(solution_directory=cur_dir,sidesets=sidesets,base_name=base_name,skip_files=opinf_settings['training-skip-steps'])
                if opinf_settings['stop-training-time'] != 'end':
                    for sideset in sidesets:
                        sideset_snapshots_tmp[sideset] = sideset_snapshots_tmp[sideset][...,0:stop_index] 

                # Add extra axis to handle case where we have multiple training directories
                for sideset in sidesets:
                    sideset_snapshots_tmp[sideset] = sideset_snapshots_tmp[sideset][:,:,None,:] 

                ## Append
                for sideset in sidesets:
                    sideset_snapshots[sideset] = np.append(sideset_snapshots[sideset],sideset_snapshots_tmp[sideset],axis=2)
    else:
        sideset_snapshots = {}
         
    # Set values = 0 if DOFs are fixed
    displacement_snapshots[free_dofs[:,:]==False] = 0.
    velocity_snapshots[free_dofs[:,:]==False] = 0.
    acceleration_snapshots[free_dofs[:,:]==False] = 0.

    snapshots_dict = {}
    snapshots_dict['displacement'] = displacement_snapshots
    snapshots_dict['velocity'] = velocity_snapshots
    snapshots_dict['acceleration'] = acceleration_snapshots
    snapshots_dict['sidesets'] = sideset_snapshots 
    snapshots_dict['free_dofs'] = free_dofs
    snapshots_dict['times'] = times
    return snapshots_dict


def convert_nodesets_to_sidesets(snapshots_dict,nodesets,sidesets):
    for i,nodeset in enumerate(nodesets):
      sideset = sidesets[i]
      data = np.append(snapshots_dict['sidesets'][nodeset + '-x'],snapshots_dict['sidesets'][nodeset + '-y'],axis=0)
      data = np.append(data,snapshots_dict['sidesets'][nodeset + '-z'],axis=0)
      snapshots_dict['sidesets'][sideset] = data
      snapshots_dict['sidesets'].pop(nodeset + '-x',None)
      snapshots_dict['sidesets'].pop(nodeset + '-y',None)
      snapshots_dict['sidesets'].pop(nodeset + '-z',None)
      print('Succesfully converted ' + sideset + ' from nodesets to a sideset')
    return snapshots_dict

def make_opinf_model(opinf_settings):
    # Check that settings are valid
    check_valid_settings(opinf_settings)
    # Load snapshots
    snapshots_dict = get_processed_snapshots(opinf_settings)
    # Create the model
    make_opinf_model_from_snapshots_dict(snapshots_dict,opinf_settings)


def make_opinf_model_from_snapshots_dict(snapshots_dict,opinf_settings):
    print(BOLD_BLUE + '===== OpInf configuration ===')
    for key in list(opinf_settings.keys()):
      print(key + ':',opinf_settings[key])
    print('=============================' + RESET)
    #print(opinf_settings)
    # Check that settings are valid
    check_valid_settings(opinf_settings)

    n_training_cases = len(opinf_settings['training-data-directories'])

    displacement_snapshots = snapshots_dict['displacement']
    velocity_snapshots = snapshots_dict['velocity']
    acceleration_snapshots = snapshots_dict['acceleration']
    free_dofs = snapshots_dict['free_dofs']
    times = snapshots_dict['times']

    #Get sideset snapshots
    input_yaml = normaopinf.parser.open_yaml(opinf_settings['fom-yaml-file'])
    sideset_snapshots = snapshots_dict['sidesets']
    sidesets = list(sideset_snapshots.keys())#get_bc_sidesets(input_yaml)
    
    reduced_sideset_snapshots = {}
    if len(sidesets) > 0:
        # Create bases for sidesets 
        if opinf_settings['boundary-truncation-type'] == "energy":
            my_boundary_truncater = romtools.vector_space.utils.EnergyBasedTruncater(opinf_settings['boundary-truncation-value'])
        elif opinf_settings['boundary-truncation-type'] == "size":
            my_boundary_truncater = romtools.vector_space.utils.BasisSizeTruncater(opinf_settings['boundary-truncation-value'])
        else:
            print("Truncater " + opinf_settings['boundary-truncation-type'] + " not supported")

        ss_tspace = {}
        ss_tspace_energy = {}
        for sideset in sidesets:
            ## Flatten last axis
            snapshot_shape = np.shape(sideset_snapshots[sideset])
            reshaped_snapshots = np.reshape(sideset_snapshots[sideset],(snapshot_shape[0],snapshot_shape[1],snapshot_shape[2]*snapshot_shape[3]) )
            ss_tspace_energy[sideset] = np.zeros(0) 

            if sideset_snapshots[sideset].shape[0] ==  1 or opinf_settings['trial-space-splitting-type'] == 'combined':
                ss_tspace[sideset] = romtools.VectorSpaceFromPOD(snapshots=reshaped_snapshots,
                                                  truncater=my_boundary_truncater,
                                                  shifter = None,
                                                  orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                                  scaler = romtools.vector_space.utils.NoOpScaler())
                ss_tspace_energy[sideset] = np.append( ss_tspace_energy[sideset] , truncater.get_energy())

            else:
                comp_trial_space = []
                for i in range(0,3):
                    tspace = romtools.VectorSpaceFromPOD(snapshots=reshaped_snapshots[i:i+1],
                                              truncater=my_boundary_truncater,
                                              shifter = None,
                                              orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                              scaler = romtools.vector_space.utils.NoOpScaler())

                    comp_trial_space.append(tspace)
                    ss_tspace_energy[sideset] = np.append( ss_tspace_energy[sideset] , truncater.get_energy())

                ss_tspace[sideset] = romtools.CompositeVectorSpace(comp_trial_space)

              
            reduced_sideset_snapshots[sideset] = romtools.rom.optimal_l2_projection(reshaped_snapshots,ss_tspace[sideset]) 
            reduced_sideset_snapshots[sideset] = np.reshape( reduced_sideset_snapshots[sideset], (reduced_sideset_snapshots[sideset].shape[0],snapshot_shape[2],snapshot_shape[3]) )
 
        reduced_stacked_sideset_snapshots = None
        for sideset in sidesets:
            if reduced_stacked_sideset_snapshots is None:
                reduced_stacked_sideset_snapshots = reduced_sideset_snapshots[sideset]*1.
            else: 
                reduced_stacked_sideset_snapshots = np.append(reduced_stacked_sideset_snapshots,reduced_sideset_snapshots[sideset],axis=0)

    else:
        reduced_stacked_sideset_snapshots = np.zeros((1,displacement_snapshots.shape[-2],displacement_snapshots.shape[-1]))

    # Create trial space for main DOFs
    if opinf_settings['truncation-type'] == "energy":
        my_truncater = romtools.vector_space.utils.EnergyBasedTruncater(opinf_settings['truncation-value'])
    elif opinf_settings['truncation-type'] == "size":
        my_truncater = romtools.vector_space.utils.BasisSizeTruncater(opinf_settings['truncation-value'])
    else:
        print("Truncater " + opinf_settings['boundary-truncation-type'] + " not supported")

    snapshot_shape = np.shape(displacement_snapshots)
    reshaped_snapshots = np.reshape(displacement_snapshots,(snapshot_shape[0],snapshot_shape[1],snapshot_shape[2]*snapshot_shape[3]) )

    tspace_energy = np.zeros(0)
    if opinf_settings['trial-space-splitting-type'] == "combined":
        trial_space = romtools.VectorSpaceFromPOD(snapshots=reshaped_snapshots,
                                              truncater=my_truncater,
                                              shifter = None,
                                              orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                              scaler = romtools.vector_space.utils.NoOpScaler())
        # Get associted energy criteria for reporting
        energy = my_truncater.get_energy()
        tspace_energy = np.append(tspace_energy,energy)

    elif opinf_settings['trial-space-splitting-type'] == 'split':
        trial_spaces = []
        for i in range(0,3):
            trial_space = romtools.VectorSpaceFromPOD(snapshots=reshaped_snapshots[i:i+1],
                                            truncater=my_truncater,
                                            shifter = None,
                                            orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                            scaler = romtools.vector_space.utils.NoOpScaler())
            trial_spaces.append(trial_space)
            # Get associted energy criteria if we truncated based on size
            if opinf_settings['truncation-type'] == "size":
                energy = my_truncater.get_energy()
                tspace_energy = np.append(tspace_energy,energy)

            # Get associted energy criteria for reporting
            energy = my_truncater.get_energy()
            tspace_energy = np.append(tspace_energy,energy)
        trial_space = romtools.CompositeVectorSpace(trial_spaces)
    else:
        print('trial-space-splitting-type not found') 



    # Project snapshots to ROM space
    snapshot_shape = np.shape(displacement_snapshots)
    reshaped_snapshots = np.reshape(displacement_snapshots,(snapshot_shape[0],snapshot_shape[1],snapshot_shape[2]*snapshot_shape[3]) )
    uhat = romtools.rom.optimal_l2_projection(reshaped_snapshots,trial_space)
    uhat = np.reshape(uhat,(uhat.shape[0],snapshot_shape[2],snapshot_shape[3]))

    snapshot_shape = np.shape(velocity_snapshots)
    reshaped_snapshots = np.reshape(velocity_snapshots,(snapshot_shape[0],snapshot_shape[1],snapshot_shape[2]*snapshot_shape[3]) )
    uhat_dots = romtools.rom.optimal_l2_projection(reshaped_snapshots,trial_space)
    uhat_dots = np.reshape(uhat_dots,(uhat_dots.shape[0],snapshot_shape[2],snapshot_shape[3]))


    if opinf_settings['acceleration-computation-type'] == 'acceleration-snapshots':
      snapshot_shape = np.shape(acceleration_snapshots)
      reshaped_snapshots = np.reshape(acceleration_snapshots,(snapshot_shape[0],snapshot_shape[1],snapshot_shape[2]*snapshot_shape[3]) )
      uhat_ddots = romtools.rom.optimal_l2_projection(reshaped_snapshots,trial_space)
      uhat_ddots = np.reshape(uhat_ddots,(uhat_ddots.shape[0],snapshot_shape[2],snapshot_shape[3]))

    elif opinf_settings['acceleration-computation-type'] == 'finite-difference':
      u_ddots = normaopinf.calculus.d2dx2(displacement_snapshots,times,method='2nd-order-uniform-dx')
      snapshot_shape = np.shape(u_ddots)
      reshaped_snapshots = np.reshape(u_ddots,(snapshot_shape[0],snapshot_shape[1],snapshot_shape[2]*snapshot_shape[3]) )
      uhat_ddots = romtools.rom.optimal_l2_projection(reshaped_snapshots,trial_space)
      uhat_ddots = np.reshape(uhat_ddots,(uhat_ddots.shape[0],snapshot_shape[2],snapshot_shape[3]))


    ####### POLYNOMIAL OPINF MODELS
    opinf_polynomial_models = ['linear','linear-symmetric','quadratic','cubic']
    if opinf_settings['model-type'] in opinf_polynomial_models:
        if opinf_settings['forcing'] == False:
          if opinf_settings['model-type'] == 'linear':
            #opinf_model = opinf.models.ContinuousModel("AB")
            opinf_model = opinf_models.ShaneNonParametricOpInfModel("AB")
          elif opinf_settings['model-type'] == 'linear-symmetric':
            #opinf_model = opinf.models.ContinuousModel("AB")
            opinf_model = opinf_models.AnthonyNonParametricOpInfModel()
          elif opinf_settings['model-type'] == 'quadratic':
            opinf_model = opinf_models.ShaneNonParametricQuadraticOpInfModel("AHB")
          elif opinf_settings['model-type'] == 'cubic':
            opinf_model = opinf_models.ShaneNonParametricCubicOpInfModel("AHGB")
   
          else:
            print("model type not found")
            sys.exit()
        else:
          if opinf_settings['model-type'] == 'linear':
            opinf_model = opinf_models.ShaneNonParametricOpInfModel("cAB")
          elif opinf_settings['model-type'] == 'linear-symmetric':
            opinf_model = opinf_models.AnthonyNonParametricOpInfModel()
          elif opinf_settings['model-type'] == 'quadratic':
            opinf_model = opinf_models.ShaneNonParametricOpInfModel("cAHB")
          elif opinf_settings['model-type'] == 'cubic':
            opinf_model = opinf_models.ShaneNonParametricCubicOpInfModel("cAHGB")
   
          else:
            print("model type not found")
            sys.exit()
    
        if isinstance(opinf_settings['regularization-parameter'],np.ndarray) or isinstance(opinf_settings['regularization-parameter'],list):
            opinf_model,regularization_parameter = non_parametric_fit_with_grid_search(opinf_model=opinf_model,x=uhat,xdot=uhat_dots,xddot=uhat_ddots,bcs = reduced_stacked_sideset_snapshots,times=times, regularization_parameters_to_try=opinf_settings['regularization-parameter'])
        elif opinf_settings['regularization-parameter'] == "automatic":
            opinf_model,regularization_parameter = non_parametric_fit_with_grid_search(opinf_model=opinf_model,x=uhat,xdot=uhat_dots,xddot=uhat_ddots,bcs = reduced_stacked_sideset_snapshots,times=times)
        else:
           #l2solver = opinf.lstsq.L2Solver(regularizer=opinf_settings['regularization-parameter'])
           #opinf_model.fit(states=uhat, ddts=uhat_ddots,inputs=reduced_stacked_sideset_snapshots)
           opinf_model.set_solver(opinf_settings['regularization-parameter'])
           regularization_parameter = opinf_settings['regularization-parameter']
           print('here',np.shape(uhat_ddots))
           opinf_model.fit(states=uhat, ddts=uhat_ddots,inputs=reduced_stacked_sideset_snapshots)
    
        ## Now extract boundary operators and create dictionary to save
        Phi = trial_space.get_basis()
        nvars,n,k = np.shape(Phi)
        Phi = np.reshape(Phi,(nvars*n,k))
        print(BOLD_BLUE + '=========================')
        print('SUMMARY   ' )
        print(f'ROM basis size: {k}')
        print(f'Sideset info:')
        for sideset in sidesets:
          print(f'    Sideset {sideset} basis size: {np.shape( ss_tspace[sideset].get_basis() )[-1]}') 
        print('=========================' + RESET)
        K = opinf_model.get_stiffness_matrix()
        B = opinf_model.get_exogenous_input_matrix()
    
    
        col_start = 0
        sideset_operators = {}
        for sideset in sidesets:
          num_dofs = reduced_sideset_snapshots[sideset].shape[0]
          val = np.einsum('kr,vnr->vkn',B[:,col_start:col_start + num_dofs] , ss_tspace[sideset].get_basis() )
          shape2 = B[:,col_start:col_start + num_dofs] @ ss_tspace[sideset].get_basis()[0].transpose()
          sideset_operators["B_" + sideset] = val#
          col_start += num_dofs
        
    
        vals_to_save = sideset_operators 
        for sideset in sidesets:
          vals_to_save[sideset + '-energy'] = ss_tspace_energy[sideset]

        vals_to_save["regularization-parameter"] = regularization_parameter 
        vals_to_save["basis"] = trial_space.get_basis() 
        vals_to_save["K"] = K
        vals_to_save["energy-cutoff"] = tspace_energy 

        if isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricQuadraticOpInfModel):
          H = opinf_model.get_quadratic_stiffness_matrix()
          vals_to_save['H'] = H
 
        if isinstance(opinf_model,normaopinf.opinf.models.ShaneNonParametricCubicOpInfModel):
          H = opinf_model.get_quadratic_stiffness_matrix()
          vals_to_save['H'] = H
          G = opinf_model.get_cubic_stiffness_matrix()
          vals_to_save['G'] = G

        if opinf_settings['forcing'] == True:
            f = -opinf_model.c_.entries
        else:
            f = np.zeros(K.shape[0])
    
        vals_to_save["f"] = f
        np.savez(opinf_settings['model-name'],**vals_to_save)


    ####### NEURAL NETWORK OPINF MODELS
    opinf_neural_network_models = ['neural-network']

    # make output directory
    if opinf_settings['model-type'] in opinf_neural_network_models:
        output_dir = opinf_settings['neural-network-training-settings']['output-path']

        if os.path.isdir(output_dir):
          pass
        else:
          os.makedirs(output_dir)

        vals_to_save = {}
        vals_to_save["basis"] = trial_space.get_basis()
        np.savez(output_dir + '/nn-opinf-basis',**vals_to_save)
        for sideset in sidesets:
          basis = ss_tspace[sideset].get_basis()
          np.savez(output_dir + '/nn-opinf-basis-' + sideset,basis=basis)

        for ensemble_id in range(opinf_settings['ensemble-size']):
            train_model(opinf_settings,output_dir,ensemble_id,uhat,uhat_ddots,reduced_sideset_snapshots,initialization_model=None)

