import argparse
import pandas as pd
import numpy as np

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def main(args):
  # read data
  surprise_data = get_data(args.training_data)

  # split data
  train_set, test_set = split_data(surprise_data)

  # train model
  model = train_model(args.n_factors, args.lr_all, args.reg_all, train_set)

  # evaluate model
  eval_model(model, test_set)

def get_data(path):
  """
  Read in ratings data for training. Data should have format: movie_id, user_id
  
  Args:
    path: Path to file for ratings data

  Return: A surprise Dataset object of the ratings data
  """
  print('READING DATA...')
  df = pd.read_csv(path)
  reader = Reader(rating_scale=(0.5, 5.0))
  surprise_data = Dataset.load_from_df(df, reader)

  return surprise_data

def split_data(surprise_data):
  """
  Split ratings data into 80/20 train/test split
  
  Args:
    surprise_data: surprise Dataset object of ratings data

  Return: 2 surprise Dataset objects for training and test sets
  """
  print('SPLITTING DATA...')
  train_set, test_set = train_test_split(surprise_data, test_size=0.2, random_state=77)
  
  return train_set, test_set

def train_model(n_factors, lr_all, reg_all, train_set):
  """
  Train SVD model
  
  Args:
    n_factors: The number of factors.
    lr_all: Learning rate for all parameters
    reg_all: Regularization rate for all parameters
    train_set: surprise Dataset object for training set

  Return: Object for fitted SVD model
  """
  print('TRAINING MODEL...')
  print(f'\tUsing parameters: n_factors={n_factors}; lr_all={lr_all}; reg_all={reg_all}')
  svd_model = SVD(
    n_factors=n_factors, 
    n_epochs=20, # default 
    lr_all=lr_all, 
    reg_all=reg_all,
    random_state=77
  ).fit(train_set)

  return svd_model

def eval_model(svd_model, test_set):
  """
  Evaluate RMSE on test set for fitted SVD model
  
  Args: 
    svd_model: Object for fitted SVD model
    test_set: surprise Dataset object for test set
  """
  predictions = svd_model.test(test_set)
  # RMSE as the primary metric to evaluate (minimize)
  rmse = accuracy.rmse(predictions)
  print(rmse) 

def parse_args():
  """
  Parse cmd arguments
  """
  # setup arg parser
  parser = argparse.ArgumentParser()

  # add arguments
  parser.add_argument("--training_data", dest='training_data', type=str)
  parser.add_argument("--n_factors", dest='n_factors', type=int, default=20)
  parser.add_argument("--lr_all", dest='lr_all', type=float, default=0.007)
  parser.add_argument("--reg_all", dest='reg_all', type=float, default=0.02)

  # parse args
  args = parser.parse_args()
  # return args

  return args

# run script
if __name__ == "__main__":
  # add space in logs
  print("\n\n")
  print("*" * 60)

  # parse args
  args = parse_args()

  # run main function
  main(args)

  # add space in logs
  print("*" * 60)
  print("\n\n")
