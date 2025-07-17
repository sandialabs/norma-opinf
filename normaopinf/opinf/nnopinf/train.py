import torch
import nnopinf.operators as operators
import nnopinf.models as models
import nnopinf.training

def train_model(output_dir,ensemble_id,uhat,uhat_ddots,reduced_stacked_sideset_snapshots,initialization_model=None):
    rom_dim = uhat.shape[0]

    n_hidden_layers = 4
    n_neurons_per_layer = rom_dim
    n_inputs = rom_dim
    n_outputs = rom_dim

    ## Design operators for the state
    NpdMlp = operators.NpdOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    SkewMlp = operators.SkewOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,n_outputs)
    MatrixMlp = operators.MatrixOperator(n_hidden_layers,n_neurons_per_layer,n_inputs,(n_outputs,n_outputs))
    CompositeOperator = operators.CompositeOperator([NpdMlp])

    my_operators = []
    for sideset in sidesets:
      n_sideset_inputs = reduced_sideset_snapshots[sideset].shape[0]
      op = operators.StandardOperator(n_hidden_layers,n_neurons_per_layer,n_sideset_inputs,n_outputs)
      my_operators.append( models.WrappedOperatorForModel(operator=op,inputs=("u-" + sideset,),name="BC-" + sideset + "-" + str(ensemble_id) )  )
    stiffness_operator =  models.WrappedOperatorForModel(operator=CompositeOperator,inputs=("x",),name="stiffness-" + str(ensemble_id))
    my_operators.append(stiffness_operator)

    my_model = models.OpInfModel( my_operators )

    if initialization_model is not None:
        my_model.hierarchical_update(initialization_model)

    #Construct training data
    if os.path.isdir(output_path):
        pass
    else:
        os.makedirs(output_path)

    training_settings = nnopinf.training.get_default_settings()
    training_settings['num-epochs'] = 10000
    training_settings['learning-rate'] = 1.e-3
    training_settings['lr-decay'] = 0.9998
    training_settings['weight-decay'] = 1.0e-10
    training_settings['batch-size'] = 200
    training_settings['output-path'] = output_dir
    training_settings['print-training-output'] = True

    inputs = {}
    inputs['x'] = uhat.transpose()
    for sideset in sidesets:
      inputs['u-' + sideset] = reduced_sideset_snapshots[sideset].transpose()

    nnopinf.training.train(my_model,input_dict=inputs,y=uhat_ddots.transpose(),training_settings=training_settings)

