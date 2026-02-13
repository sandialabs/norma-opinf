import tkinter as tk
import nnopinf
import nnopinf.training
from tkinter import ttk, messagebox
import os
import subprocess  # For running the generated Python file

# Visual theme
BG_COLOR = "#f7f4ef"
CARD_COLOR = "#ffffff"
TEXT_COLOR = "#1e1f26"
SUBTEXT_COLOR = "#5b5f6a"
ACCENT_COLOR = "#d45500"
ACCENT_HOVER = "#c14900"

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
      settings['model-structure'] = model_structure_var.get()
      settings['n-hidden-layers'] = int(n_hidden_layers_var.get())
      neurons_per_layer = n_neurons_per_layer_var.get().strip()
      if neurons_per_layer.lower() in ("", "auto"):
        settings['n-neurons-per-layer'] = "auto"
      else:
        settings['n-neurons-per-layer'] = int(neurons_per_layer)
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
root.configure(bg=BG_COLOR)

style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background=BG_COLOR)
style.configure("Card.TFrame", background=CARD_COLOR)
style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Avenir", 11))
style.configure("Header.TLabel", background=BG_COLOR, foreground=ACCENT_COLOR, font=("Avenir", 20, "bold"))
style.configure("Subheader.TLabel", background=BG_COLOR, foreground=SUBTEXT_COLOR, font=("Avenir", 11))
style.configure("Section.TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=("Avenir", 12, "bold"))
style.configure("TEntry", fieldbackground=CARD_COLOR, foreground=TEXT_COLOR)
style.configure("TCombobox", fieldbackground=CARD_COLOR, foreground=TEXT_COLOR)
style.configure(
    "Accent.TButton",
    background=ACCENT_COLOR,
    foreground="white",
    borderwidth=0,
    padding=6,
    font=("Avenir", 11, "bold"),
)
style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])

root.lift()
root.attributes('-topmost',True)

# Header
header_frame = ttk.Frame(root, padding="16 14 16 6")
header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
ttk.Label(header_frame, text="OpInf Generator", style="Header.TLabel").grid(row=0, column=0, sticky=tk.W)
ttk.Label(
    header_frame,
    text="Configure and run ROM/OpInf builds with neural-network options.",
    style="Subheader.TLabel",
).grid(row=1, column=0, sticky=tk.W)

# Create a frame for the dropdowns
frame = ttk.Frame(root, padding="12 10 12 16", style="Card.TFrame")
frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

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
output_entry = ttk.Entry(frame, textvariable=output_filename_var, width=32)
output_entry.grid(row=0, column=1, padx=5, pady=5)
#ToolTip(output_entry, "Enter the name of the output Python file.")
ttk.Label(frame, text="FOM YAML File:").grid(row=1, column=0, sticky=tk.W)
fom_yaml_entry = ttk.Entry(frame, textvariable=fom_yaml_var, width=32)
fom_yaml_entry.grid(row=1, column=1, padx=5, pady=5)
#ToolTip(fom_yaml_entry, "Specify the YAML file for FOM settings.")

ttk.Label(frame, text="Training Data Directories (comma-separated):").grid(row=2, column=0, sticky=tk.W)
training_data_entry = ttk.Entry(frame, textvariable=training_data_var, width=32)
training_data_entry.grid(row=2, column=1, padx=5, pady=5)
#ToolTip(training_data_entry, "Enter the directories for training data.")

ttk.Label(frame, text="Model Type:").grid(row=3, column=0, sticky=tk.W)
model_type_combo = ttk.Combobox(
    frame,
    textvariable=model_type_var,
    values=['linear', 'quadratic', 'cubic', 'neural-network'],
    width=30,
)
model_type_combo.grid(row=3, column=1, padx=5, pady=5)
#ToolTip(model_type_combo, "Select the type of model to use.")

ttk.Label(frame, text="Stop Training Time:").grid(row=4, column=0, sticky=tk.W)
stop_training_time_entry = ttk.Entry(frame, textvariable=stop_training_time_var, width=32)
stop_training_time_entry.grid(row=4, column=1, padx=5, pady=5)
#ToolTip(stop_training_time_entry, "Specify the stop training time.")

ttk.Label(frame, text="Training Skip Steps:").grid(row=5, column=0, sticky=tk.W)
training_skip_steps_entry = ttk.Entry(frame, textvariable=training_skip_steps_var, width=32)
training_skip_steps_entry.grid(row=5, column=1, padx=5, pady=5)
#ToolTip(training_skip_steps_entry, "Down-select training times, e.g., times = times[::skip_steps].")

ttk.Label(frame, text="Trial Space Splitting Type:").grid(row=6, column=0, sticky=tk.W)
trial_space_splitting_type_combo = ttk.Combobox(
    frame, textvariable=trial_space_splitting_type_var, values=['split', 'combined'], width=30
)
trial_space_splitting_type_combo.grid(row=6, column=1, padx=5, pady=5)
#ToolTip(trial_space_splitting_type_combo, "If we want separate DOFs for x,y,z components.")

ttk.Label(frame, text="Acceleration Computation Type:").grid(row=7, column=0, sticky=tk.W)
acceleration_computation_type_combo = ttk.Combobox(
    frame,
    textvariable=acceleration_computation_type_var,
    values=['finite-difference', 'acceleration-snapshots'],
    width=30,
)
acceleration_computation_type_combo.grid(row=7, column=1, padx=5, pady=5)
#ToolTip(acceleration_computation_type_combo, "How acceleration values are computed.")


ttk.Label(frame, text="Forcing:").grid(row=8, column=0, sticky=tk.W)
forcing_combo = ttk.Combobox(frame, textvariable=forcing_var, values=['True', 'False'], width=30)
forcing_combo.grid(row=8, column=1, padx=5, pady=5)
#ToolTip(forcing_combo, "Select whether to apply forcing.")

ttk.Label(frame, text="Truncation Type:").grid(row=9, column=0, sticky=tk.W)
truncation_type_combo = ttk.Combobox(frame, textvariable=truncation_type_var, values=['energy', 'size'], width=30)
truncation_type_combo.grid(row=9, column=1, padx=5, pady=5)
#ToolTip(truncation_type_combo, "Select the truncation type.")

ttk.Label(frame, text="Truncation Value:").grid(row=10, column=0, sticky=tk.W)
truncation_value_entry = ttk.Entry(frame, textvariable=truncation_value_var, width=32)
truncation_value_entry.grid(row=10, column=1, padx=5, pady=5)
#ToolTip(truncation_value_entry, "Enter the truncation value as a float.")

ttk.Label(frame, text="Boundary Truncation Type:").grid(row=11, column=0, sticky=tk.W)
boundary_truncation_type_combo = ttk.Combobox(
    frame, textvariable=boundary_truncation_type_var, values=['energy', 'size'], width=30
)
boundary_truncation_type_combo.grid(row=11, column=1, padx=5, pady=5)
#ToolTip(boundary_truncation_type_combo, "Select the boundary truncation type.")

ttk.Label(frame, text="Boundary Truncation Value:").grid(row=12, column=0, sticky=tk.W)
boundary_truncation_value_entry = ttk.Entry(frame, textvariable=boundary_truncation_value_var, width=32)
boundary_truncation_value_entry.grid(row=12, column=1, padx=5, pady=5)
#ToolTip(boundary_truncation_value_entry, "Enter the boundary truncation value as a float.")

ttk.Label(frame, text="Regularization Parameter(s) (comma-separated):").grid(row=13, column=0, sticky=tk.W)
regularization_parameter_entry = ttk.Entry(frame, textvariable=regularization_parameter_var, width=32)
regularization_parameter_entry.grid(row=13, column=1, padx=5, pady=5)
#ToolTip(regularization_parameter_entry, "Enter regularization parameters as a comma-separated list.")

ttk.Label(frame, text="Model Name:\n(Enter the name of the model)").grid(row=14, column=0, sticky=tk.W)
model_name_entry = ttk.Entry(frame, textvariable=model_name_var, width=32)
model_name_entry.grid(row=14, column=1, padx=5, pady=5)
#ToolTip(model_name_entry, "Enter the name of the model.")


# Add a label for additional settings
ttk.Separator(frame).grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
ttk.Label(frame, text="Neural Network Settings", style="Section.TLabel").grid(
    row=16, column=0, columnspan=2, pady=5
)

# New input fields for additional options
ttk.Label(frame, text="Model Structure:").grid(row=17, column=0, sticky=tk.W)
model_structure_var = tk.StringVar(value='PsdLagrangianOperator')
model_structure_combo = ttk.Combobox(
    frame,
    textvariable=model_structure_var,
    values=['PsdLagrangianOperator', 'SpdOperator'],
    width=30,
)
model_structure_combo.grid(row=17, column=1, padx=5, pady=5)

ttk.Label(frame, text="Hidden Layers:").grid(row=18, column=0, sticky=tk.W)
n_hidden_layers_var = tk.StringVar(value='2')
n_hidden_layers_entry = ttk.Entry(frame, textvariable=n_hidden_layers_var, width=32)
n_hidden_layers_entry.grid(row=18, column=1, padx=5, pady=5)

ttk.Label(frame, text="Neurons per Layer (int or auto):").grid(row=19, column=0, sticky=tk.W)
n_neurons_per_layer_var = tk.StringVar(value='auto')
n_neurons_per_layer_entry = ttk.Entry(frame, textvariable=n_neurons_per_layer_var, width=32)
n_neurons_per_layer_entry.grid(row=19, column=1, padx=5, pady=5)

ttk.Label(frame, text="Number of Epochs:").grid(row=20, column=0, sticky=tk.W)
num_epochs_var = tk.StringVar(value='25000')  # Default value
num_epochs_entry = ttk.Entry(frame, textvariable=num_epochs_var, width=32)
num_epochs_entry.grid(row=20, column=1, padx=5, pady=5)

ttk.Label(frame, text="Batch Size:").grid(row=21, column=0, sticky=tk.W)
batch_size_var = tk.StringVar(value='500')  # Default value
batch_size_entry = ttk.Entry(frame, textvariable=batch_size_var, width=32)
batch_size_entry.grid(row=21, column=1, padx=5, pady=5)

ttk.Label(frame, text="Learning Rate:").grid(row=22, column=0, sticky=tk.W)
learning_rate_var = tk.StringVar(value='1.e-3')  # Default value
learning_rate_entry = ttk.Entry(frame, textvariable=learning_rate_var, width=32)
learning_rate_entry.grid(row=22, column=1, padx=5, pady=5)

ttk.Label(frame, text="l2 weight regularization:").grid(row=23, column=0, sticky=tk.W)
weight_decay_var = tk.StringVar(value='1.e-8')  # Default value
weight_decay_entry = ttk.Entry(frame, textvariable=weight_decay_var, width=32)
weight_decay_entry.grid(row=23, column=1, padx=5, pady=5)

ttk.Label(frame, text="Learning Rate Decay:").grid(row=24, column=0, sticky=tk.W)
learning_rate_decay_var = tk.StringVar(value='0.9999')  # Default value
learning_rate_decay_entry = ttk.Entry(frame, textvariable=learning_rate_decay_var, width=32)
learning_rate_decay_entry.grid(row=24, column=1, padx=5, pady=5)

row_no = 25
ttk.Label(frame, text="Ensemble size:").grid(row=row_no, column=0, sticky=tk.W)
ensemble_size_var = tk.StringVar(value='5')  # Default value
ensemble_size_entry = ttk.Entry(frame, textvariable=ensemble_size_var, width=32)
ensemble_size_entry.grid(row=row_no, column=1, padx=5, pady=5)

row_no += 1

ttk.Label(frame, text="Resume Training:").grid(row=row_no, column=0, sticky=tk.W)
resume_var = tk.StringVar(value='False')  # Default value
resume_combo = ttk.Combobox(frame, textvariable=resume_var, values=['True', 'False'], width=30)
resume_combo.grid(row=row_no, column=1, padx=5, pady=5)

row_no += 1
ttk.Label(frame, text="Output path:").grid(row=row_no, column=0, sticky=tk.W)
output_path_var = tk.StringVar(value='ml-models')  # Default value
output_path_entry = ttk.Entry(frame, textvariable=output_path_var, width=32)
output_path_entry.grid(row=row_no, column=1, padx=5, pady=5)

row_no += 1
# Create buttons to generate the Python file
generate_button = ttk.Button(
    frame,
    text="Generate Python File",
    style="Accent.TButton",
    command=lambda: generate_python_file(collect_settings(), output_filename_var.get()),
)
generate_button.grid(row=row_no, column=0, columnspan=2, pady=10)

row_no += 1
run_button = ttk.Button(
    frame,
    text="Generate Python File and Train Model",
    command=lambda: run_python_file(),
)
run_button.grid(row=row_no, column=0, columnspan=2, pady=10)

# Start the GUI event loop

root.mainloop()
