import os
import sys
from argparse import Namespace
from importlib import import_module
import traceback

def list_solution_files(code_path):
  """
  Check prefix and file type
  """
  solution_files = []
  for x in os.listdir(code_path):
     if x[:4] == "sol_" and x[-3:] == ".py":
        solution_files.append(x)
  solution_files = sorted(solution_files)
  return solution_files


def run_tests(code_path, tests, answers):
  solution_files = list_solution_files(code_path)
  scores = {}
  for solution_file in  solution_files:

    print('testing {}'.format(solution_file))
    module_name = solution_file.replace('.py', '')
    test_answers = answers[module_name]
    args_ = {
              'code_path': code_path,
              'file_name': solution_file,
              'module_name': module_name,
           }
    args = Namespace(**args_)

    scores_ = run_tests_(args, tests, test_answers)
    scores[module_name] = scores_
  return scores


def run_tests_(args, tests, test_answers):
  #Add directory where code to be tested is to the path
  sys.path.append(args.code_path)
  module = import_module(args.module_name)
  #run module
  test_data  = module.main(test=True)
  #run tests
  num_correct, num_total = 0,0
  tests_run = []
  scores = []
  for test_name, test in tests.items():
    try:
        x,y = test(args.module_name, test_data, test_answers)
    except Exception as e:
        print("Error on test '{}'! Error was:\n '{}'".format(test_name, e))
        traceback.print_exc()
        x,y = 0,1
    if y!= 0:
      tests_run.append(test_name)
      name = "{} - {}".format(args.module_name, test_name)
      scores.append({'score': x, 'max_score': y, 'name': name})

    num_correct += x
    if x - y != 0:
      print("{} failed".format(test_name))
    num_total += y

  #Log results
  if num_total != 0:
    print("Number of tests correct: {} / {}".format(num_correct, num_total))
    print("Tests run: {}".format(tests_run))
  return scores

