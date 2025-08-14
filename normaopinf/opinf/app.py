import tkinter as tk
import nnopinf
import nnopinf.training
from tkinter import ttk, messagebox
import os
import subprocess  # For running the generated Python file

# Tooltip class to create tooltips for widgets
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window is not None:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 275
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        #self.tooltip_window.wm_attributes('-topmost', True)  # Ensure tooltip is on top
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, background="lightyellow", relief="solid", borderwidth=1)
        label.pack()
        #root.attributes('-topmost',False)

    def hide_tooltip(self, event=None):
        if self.tooltip_window is not None:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# Function to generate the Python file based on selected settings
def generate_python_file(settings, output_filename):
    # Create the content for the Python file
    content = """import normaopinf
import normaopinf.opinf
import nnopinf
import nnopinf.training
import os
import numpy as np

if __name__ == '__main__':
    settings = {}
"""
    for key, value in settings.items():
        if isinstance(value, list):
            value_str = f"[{', '.join(repr(v) for v in value)}]"
        else:
            value_str = repr(value)
        content += f"    settings['{key}'] = {value_str}\n"

    content += """    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
"""

    # Write to the specified Python file
    with open(output_filename, "w") as f:
        f.write(content)

    messagebox.showinfo("Success", f"Python file '{output_filename}' generated successfully!")

# Function to collect settings from the GUI
def collect_settings():
    settings = {}
    settings['fom-yaml-file'] = fom_yaml_var.get()
    settings['training-data-directories'] = [x for x in training_data_var.get().split(',')]
    settings['model-type'] = model_type_var.get()
    settings['stop-training-time'] = float(stop_training_time_var.get())
    settings['training-skip-steps'] = int(training_skip_steps_var.get())
    settings['forcing'] = forcing_var.get() == 'True'
    settings['truncation-type'] = truncation_type_var.get()
    settings['boundary-truncation-type'] = boundary_truncation_type_var.get()
    
    # Parse the regularization parameter as a list of floats
    regularization_param_str = regularization_parameter_var.get()
    if regularization_param_str == 'automatic':
        settings['regularization-parameter'] = 'automatic'
    else:
        settings['regularization-parameter'] = [float(x) for x in regularization_param_str.split(',')]
    
    # Collect additional settings
    settings['model-name'] = model_name_var.get()
    if settings['truncation-type'] == 'size':
      settings['truncation-value'] = int(truncation_value_var.get())
    else:
      settings['truncation-value'] = float(truncation_value_var.get())

    if settings['boundary-truncation-type'] == 'size':
      settings['boundary-truncation-value'] = int(boundary_truncation_value_var.get())
    else:
      settings['boundary-truncation-value'] = float(boundary_truncation_value_var.get())
   
    settings['trial-space-splitting-type'] = trial_space_splitting_type_var.get()
    settings['acceleration-computation-type'] = acceleration_computation_type_var.get()

    if settings['model-type'] == 'neural-network':
      settings['neural-network-training-settings'] = nnopinf.training.get_default_settings()
      settings['neural-network-training-settings']['model-name'] = model_name_var.get() 
      settings['neural-network-training-settings']['output-path'] = output_path_var.get() 
      settings['neural-network-training-settings']['epoch'] = int(num_epochs_var.get())
      settings['neural-network-training-settings']['batch-size'] = int(batch_size_var.get())
      settings['neural-network-training-settings']['learning-rate'] = float(learning_rate_var.get())
      settings['neural-network-training-settings']['weight-decay'] = float(weight_decay_var.get())
      settings['neural-network-training-settings']['lr-decay'] = float(learning_rate_decay_var.get())
      if resume_var.get() == 'True' or resume_var.get() == 'true':
        resume = True
      else:
        resume = False
      settings['neural-network-training-settings']['resume'] = resume 
      settings['ensemble-size'] = int(ensemble_size_var.get())
    return settings

# Function to run the generated Python file
def run_python_file():
    settings = collect_settings()
    output_filename = output_filename_var.get()
    generate_python_file(settings, output_filename)
    print('Executing ' + str(output_filename)) 
    # Run the generated Python file
    try:
        subprocess.run(["python", output_filename], check=True)
        messagebox.showinfo("Success", f"Python file '{output_filename}' executed successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error running the Python file: {e}")

# Create the main window
root = tk.Tk()
root.title("OpInf Generator")

root.lift()
root.attributes('-topmost',True)

# Create a frame for the dropdowns
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Define variables for the settings
fom_yaml_var = tk.StringVar(value="cuboid.yaml")
training_data_var = tk.StringVar(value=os.getcwd())
model_type_var = tk.StringVar(value='linear')
stop_training_time_var = tk.StringVar(value='1.e5')
training_skip_steps_var = tk.StringVar(value='1')
forcing_var = tk.StringVar(value='False')
truncation_type_var = tk.StringVar(value='energy')
boundary_truncation_type_var = tk.StringVar(value='energy')
regularization_parameter_var = tk.StringVar(value='5.e-4, 5.e-3, 5.e-2')  # Example default value
model_name_var = tk.StringVar(value='opinf-operator')  # Default model name
truncation_value_var = tk.StringVar(value='0.999999')  # Default truncation value
boundary_truncation_value_var = tk.StringVar(value='0.999999')  # Default boundary truncation value
trial_space_splitting_type_var = tk.StringVar(value='split')
acceleration_computation_type_var = tk.StringVar(value='finite-difference')

# Create input fields for each setting
ttk.Label(frame, text="Output File Name:").grid(row=0, column=0, sticky=tk.W)
output_filename_var = tk.StringVar(value="generated_script.py")  # Default output filename
output_entry = ttk.Entry(frame, textvariable=output_filename_var)
output_entry.grid(row=0, column=1, padx=5, pady=5)
#ToolTip(output_entry, "Enter the name of the output Python file.")
ttk.Label(frame, text="FOM YAML File:").grid(row=1, column=0, sticky=tk.W)
fom_yaml_entry = ttk.Entry(frame, textvariable=fom_yaml_var)
fom_yaml_entry.grid(row=1, column=1, padx=5, pady=5)
#ToolTip(fom_yaml_entry, "Specify the YAML file for FOM settings.")

ttk.Label(frame, text="Training Data Directories (comma-separated):").grid(row=2, column=0, sticky=tk.W)
training_data_entry = ttk.Entry(frame, textvariable=training_data_var)
training_data_entry.grid(row=2, column=1, padx=5, pady=5)
#ToolTip(training_data_entry, "Enter the directories for training data.")

ttk.Label(frame, text="Model Type:").grid(row=3, column=0, sticky=tk.W)
model_type_combo = ttk.Combobox(frame, textvariable=model_type_var, values=['linear', 'quadratic','cubic','neural-network'])
model_type_combo.grid(row=3, column=1, padx=5, pady=5)
#ToolTip(model_type_combo, "Select the type of model to use.")

ttk.Label(frame, text="Stop Training Time:").grid(row=4, column=0, sticky=tk.W)
stop_training_time_entry = ttk.Entry(frame, textvariable=stop_training_time_var)
stop_training_time_entry.grid(row=4, column=1, padx=5, pady=5)
#ToolTip(stop_training_time_entry, "Specify the stop training time.")

ttk.Label(frame, text="Training Skip Steps:").grid(row=5, column=0, sticky=tk.W)
training_skip_steps_entry = ttk.Entry(frame, textvariable=training_skip_steps_var)
training_skip_steps_entry.grid(row=5, column=1, padx=5, pady=5)
#ToolTip(training_skip_steps_entry, "Down-select training times, e.g., times = times[::skip_steps].")

ttk.Label(frame, text="Trial Space Splitting Type:").grid(row=6, column=0, sticky=tk.W)
trial_space_splitting_type_combo = ttk.Combobox(frame, textvariable=trial_space_splitting_type_var, values=['split', 'combined'])
trial_space_splitting_type_combo.grid(row=6, column=1, padx=5, pady=5)
#ToolTip(trial_space_splitting_type_combo, "If we want separate DOFs for x,y,z components.")

ttk.Label(frame, text="Acceleration Computation Type:").grid(row=7, column=0, sticky=tk.W)
acceleration_computation_type_combo = ttk.Combobox(frame, textvariable=acceleration_computation_type_var, values=['finite-difference', 'acceleration-snapshots'])
acceleration_computation_type_combo.grid(row=7, column=1, padx=5, pady=5)
#ToolTip(acceleration_computation_type_combo, "How acceleration values are computed.")


ttk.Label(frame, text="Forcing:").grid(row=8, column=0, sticky=tk.W)
forcing_combo = ttk.Combobox(frame, textvariable=forcing_var, values=['True', 'False'])
forcing_combo.grid(row=8, column=1, padx=5, pady=5)
#ToolTip(forcing_combo, "Select whether to apply forcing.")

ttk.Label(frame, text="Truncation Type:").grid(row=9, column=0, sticky=tk.W)
truncation_type_combo = ttk.Combobox(frame, textvariable=truncation_type_var, values=['energy', 'size'])
truncation_type_combo.grid(row=9, column=1, padx=5, pady=5)
#ToolTip(truncation_type_combo, "Select the truncation type.")

ttk.Label(frame, text="Truncation Value:").grid(row=10, column=0, sticky=tk.W)
truncation_value_entry = ttk.Entry(frame, textvariable=truncation_value_var)
truncation_value_entry.grid(row=10, column=1, padx=5, pady=5)
#ToolTip(truncation_value_entry, "Enter the truncation value as a float.")

ttk.Label(frame, text="Boundary Truncation Type:").grid(row=11, column=0, sticky=tk.W)
boundary_truncation_type_combo = ttk.Combobox(frame, textvariable=boundary_truncation_type_var, values=['energy', 'size'])
boundary_truncation_type_combo.grid(row=11, column=1, padx=5, pady=5)
#ToolTip(boundary_truncation_type_combo, "Select the boundary truncation type.")

ttk.Label(frame, text="Boundary Truncation Value:").grid(row=12, column=0, sticky=tk.W)
boundary_truncation_value_entry = ttk.Entry(frame, textvariable=boundary_truncation_value_var)
boundary_truncation_value_entry.grid(row=12, column=1, padx=5, pady=5)
#ToolTip(boundary_truncation_value_entry, "Enter the boundary truncation value as a float.")

ttk.Label(frame, text="Regularization Parameter(s) (comma-separated):").grid(row=13, column=0, sticky=tk.W)
regularization_parameter_entry = ttk.Entry(frame, textvariable=regularization_parameter_var)
regularization_parameter_entry.grid(row=13, column=1, padx=5, pady=5)
#ToolTip(regularization_parameter_entry, "Enter regularization parameters as a comma-separated list.")

ttk.Label(frame, text="Model Name:\n(Enter the name of the model)").grid(row=14, column=0, sticky=tk.W)
model_name_entry = ttk.Entry(frame, textvariable=model_name_var)
model_name_entry.grid(row=14, column=1, padx=5, pady=5)
#ToolTip(model_name_entry, "Enter the name of the model.")


# Add a label for additional settings
ttk.Label(frame, text="Additional settings for neural-network models").grid(row=15, column=0, columnspan=2, pady=5)

# New input fields for additional options
ttk.Label(frame, text="Number of Epochs:").grid(row=16, column=0, sticky=tk.W)
num_epochs_var = tk.StringVar(value='25000')  # Default value
num_epochs_entry = ttk.Entry(frame, textvariable=num_epochs_var)
num_epochs_entry.grid(row=16, column=1, padx=5, pady=5)

ttk.Label(frame, text="Batch Size:").grid(row=17, column=0, sticky=tk.W)
batch_size_var = tk.StringVar(value='500')  # Default value
batch_size_entry = ttk.Entry(frame, textvariable=batch_size_var)
batch_size_entry.grid(row=17, column=1, padx=5, pady=5)

ttk.Label(frame, text="Learning Rate:").grid(row=18, column=0, sticky=tk.W)
learning_rate_var = tk.StringVar(value='1.e-3')  # Default value
learning_rate_entry = ttk.Entry(frame, textvariable=learning_rate_var)
learning_rate_entry.grid(row=18, column=1, padx=5, pady=5)

ttk.Label(frame, text="l2 weight regularization:").grid(row=19, column=0, sticky=tk.W)
weight_decay_var = tk.StringVar(value='1.e-8')  # Default value
weight_decay_entry = ttk.Entry(frame, textvariable=weight_decay_var)
weight_decay_entry.grid(row=19, column=1, padx=5, pady=5)

ttk.Label(frame, text="Learning Rate Decay:").grid(row=20, column=0, sticky=tk.W)
learning_rate_decay_var = tk.StringVar(value='0.9999')  # Default value
learning_rate_decay_entry = ttk.Entry(frame, textvariable=learning_rate_decay_var)
learning_rate_decay_entry.grid(row=20, column=1, padx=5, pady=5)

row_no = 21
ttk.Label(frame, text="Ensemble size:").grid(row=row_no, column=0, sticky=tk.W)
ensemble_size_var = tk.StringVar(value='5')  # Default value
ensemble_size_entry = ttk.Entry(frame, textvariable=ensemble_size_var)
ensemble_size_entry.grid(row=row_no, column=1, padx=5, pady=5)

row_no += 1

ttk.Label(frame, text="Resume Training:").grid(row=row_no, column=0, sticky=tk.W)
resume_var = tk.StringVar(value='False')  # Default value
resume_combo = ttk.Combobox(frame, textvariable=resume_var, values=['True', 'False'])
resume_combo.grid(row=row_no, column=1, padx=5, pady=5)

row_no += 1
ttk.Label(frame, text="Output path:").grid(row=row_no, column=0, sticky=tk.W)
output_path_var = tk.StringVar(value='ml-models')  # Default value
output_path_entry = ttk.Entry(frame, textvariable=output_path_var)
output_path_entry.grid(row=row_no, column=1, padx=5, pady=5)

row_no += 1
# Create buttons to generate the Python file
generate_button = ttk.Button(frame, text="Generate Python File", command=lambda: generate_python_file(collect_settings(), output_filename_var.get()))
generate_button.grid(row=row_no, column=0, columnspan=2, pady=10)

row_no += 1
run_button = ttk.Button(frame, text="Generate Python File and Train Model", command=lambda: run_python_file())
run_button.grid(row=row_no, column=0, columnspan=2, pady=10)

# Start the GUI event loop

root.mainloop()

