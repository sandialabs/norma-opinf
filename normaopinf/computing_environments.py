def create_serial_environment(norma_executable):
  '''
  Generate a serial computing environment 

  Args:
    norma_executable: string for path to norma executable

  Returns:
    dictionary: dictionary with information on how to execute norma

  '''

  environment = {}
  environment['norma-executable'] = norma_executable
  return environment
