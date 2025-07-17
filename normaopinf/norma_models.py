import romtools
import subprocess
import os
import normaopinf.parser
import yaml
from normaopinf.parser import *


class NormaMultiDomainModel:

    def __init__(self,input_file_name,subdomain_input_files,environment,norma_parameter_space, files_to_copy = [], files_to_soft_link = []):
        self.files_to_copy_ = files_to_copy
        self.files_to_soft_link_ = files_to_soft_link
        self.input_file_name_ = input_file_name
        self.environment_ = environment
        self.norma_parameter_space_ = norma_parameter_space
        self.subdomain_input_files_ = subdomain_input_files


    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        '''
        This function is called from the base directory and is
        responsible for populating the run directory located at run_directory.

        Examples would be setuping up input files, linking mesh files.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary contatining parameter names and sample values

        '''
        # Copy 
        for filename in self.files_to_copy_:
          os.system('cp -r ' + filename +  ' ' + run_directory + '/.')

        # Soft link
        for filename in self.files_to_soft_link_:
          os.system('ln -s ' + filename +  ' ' + run_directory + '/.')

        yaml.add_representer(quoted, quoted_presenter)

        # Update parameters
        input_file_name = run_directory + '/' + self.input_file_name_
        norma_yaml = normaopinf.parser.open_yaml(input_file_name)

        subdomain_yamls = []
        for subdomain_input_file in self.subdomain_input_files_:
            yaml_file = normaopinf.parser.open_yaml(run_directory + '/' + subdomain_input_file)
            subdomain_yamls.append(yaml_file)
        norma_yaml,subdomain_yamls = self.norma_parameter_space_.update_norma_yaml(norma_yaml,subdomain_yamls,parameter_sample)
        normaopinf.parser.save_yaml(norma_yaml,input_file_name)

        for i,subdomain_input_file in enumerate(self.subdomain_input_files_):
            normaopinf.parser.save_yaml(subdomain_yamls[i],run_directory + '/' + subdomain_input_file)


    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        '''
        This function is called from the base directory. It needs to execute our
        model.  If the model runs succesfully, return 0.  If fails, return 1.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary contatining parameter names and sample values

        '''
        current_dir = os.getcwd()
        os.chdir(run_directory)


        norma_executable = self.environment_['norma-executable']
        #with open('norma.output', 'w') as outfile:
        os.system(norma_executable + ' ' + self.input_file_name_ + ' > norma.output')
                #p1 = subprocess.Popen([norma_executable,self.input_file_name_],stdout=outfile,stderr=outfile, env=os.environ,shell=True)
                #code = p1.wait() # so that these run one at a time
                #if code != 0:
                #    message = f"ERROR: The executable did not terminate successfully."

        os.chdir(current_dir)
        return 0





class NormaModel:

    def __init__(self,input_file_name,environment,norma_parameter_space, files_to_copy = [], files_to_soft_link = []):
        self.files_to_copy_ = files_to_copy
        self.files_to_soft_link_ = files_to_soft_link
        self.input_file_name_ = input_file_name
        self.environment_ = environment
        self.norma_parameter_space_ = norma_parameter_space


    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        '''
        This function is called from the base directory and is
        responsible for populating the run directory located at run_directory.

        Examples would be setuping up input files, linking mesh files.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary contatining parameter names and sample values

        '''
        # Copy 
        for filename in self.files_to_copy_:
          os.system('cp -r ' + filename +  ' ' + run_directory + '/.')

        # Soft link
        for filename in self.files_to_soft_link_:
          os.system('ln -s ' + filename +  ' ' + run_directory + '/.')

        yaml.add_representer(quoted, quoted_presenter)

        # Update parameters
        input_file_name = run_directory + '/' + self.input_file_name_
        norma_yaml = normaopinf.parser.open_yaml(input_file_name)
        norma_yaml = self.norma_parameter_space_.update_norma_yaml(norma_yaml,parameter_sample)
        normaopinf.parser.save_yaml(norma_yaml,input_file_name)


    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        '''
        This function is called from the base directory. It needs to execute our
        model.  If the model runs succesfully, return 0.  If fails, return 1.

        Args:
          run_directory (str): Absolute path to run_directory.
          parameter_sample: Dictionary contatining parameter names and sample values

        '''
        current_dir = os.getcwd()
        os.chdir(run_directory)


        norma_executable = self.environment_['norma-executable']
        #with open('norma.output', 'w') as outfile:
        os.system(norma_executable + ' ' + self.input_file_name_ + ' > norma.output')
                #p1 = subprocess.Popen([norma_executable,self.input_file_name_],stdout=outfile,stderr=outfile, env=os.environ,shell=True)
                #code = p1.wait() # so that these run one at a time
                #if code != 0:
                #    message = f"ERROR: The executable did not terminate successfully."

        os.chdir(current_dir)
        return 0

