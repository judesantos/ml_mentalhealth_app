
"""
This module provides the model training pipeline for the
XGBoost model using the dataset prepared by the preparation pipeline.

The module contains functions to train the model, tune the hyperparameters,
test and save the model. Hyperparameter tuning is done using
Bayesian optimization with xgboost as the model.

The test data is split into training and validation sets,
then the model is trained using the training set and evaluated using
the validation set. The model with the lowest log-loss is selected as
the best model and evaluated using the test set.
"""

import os

import time
from datetime import datetime
import humanfriendly
import pickle as pkl
from loguru import logger

import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


from app.ml.config import model_settings as settings
from app.ml.model.pipeline.preparation import get_mental_health_data
from app.ml.model.pipeline.preparation import MentalHealthData

"""
  Xgboost model class helper functions and variables
  xgboost requires the class labels to start from 0
"""

# Mapping class description of the actual target label.
# _target_class_mapping = {1: '0 Days',
#                         2: '1-13 Days', 3: '14+ Days', 9: 'Unsure'}
#
# xgboost class label mapping to description
# _alt_target_class_mapping = {0: '0 Days',
#                              1: '1-13 Days', 2: '14+ Days', 3: 'Unsure'}


def _target_label_mapping(y=None):
    """ Convert target dataset labels to xgboost """
    # _MENT14D_ to xgboost label mapping
    label_mapping = {1: 0, 2: 1, 3: 2, 9: 3}
    # Convert to xgboost labels
    return np.vectorize(label_mapping.get)(y)


def build_model():
    """
    Train model given a dataset, then save the model

    The function loads the preprocessed dataset,
    splits the dataset into training and testing sets,
    trains the model using the training set, evaluates the model
    using the test set, and saves the model.

    Returns:
      float: model score
    """

    logger.info('Building model...')

    # Load the preprocessed dataset
    mh: MentalHealthData = get_mental_health_data()
    df = mh.get_data()
    # Identify X and y
    X, y = df.drop(columns=[mh.target]), df[mh.target]

    # Split the dataset into train and validation/tests sets
    # Set aside 60% for training, 20% for validation, and 20% for testing
    X_train, x_temp, y_train, y_temp = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.4
    )

    logger.debug(f'Model features: {X_train.columns}')

    # split once more for the validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        stratify=y_temp,
        test_size=0.5
    )

    _y_train = _target_label_mapping(y=y_train)
    _y_test = _target_label_mapping(y=y_test)
    _y_val = _target_label_mapping(y=y_val)

    # Compute class weights and train the model with it.
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(_y_train),
        y=_y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    sample_weight = np.array([class_weights_dict[class_label]
                             for class_label in _y_train])

    # Hyper parameter tuning - use validation data
    h_params = _hyper_parameter_tuning(
        X_train,
        _y_train,
        x_val,
        _y_val,
        sample_weight
    )
    if h_params is None:
        logger.error('Error tuning hyperparameters. Model not trained.')
        return None

    logger.info('Training model...')
    # Train the model using the best hyperparameters

    xgb_model = _create_and_train_model(
        X_train,
        _y_train,
        x_test,
        _y_test,
        h_params,
        sample_weight
    )

    if xgb_model is None:
        logger.error('Error training model. Model not trained.')
        return None

    # Save the model
    _save_model(xgb_model)


def _hyper_parameter_tuning(X_train, y_train, x_test, y_test, sample_weight):
    """
    This function tunes the hyperparameters for the model using
    the training and validation data.

    The function uses Bayesian optimization with select hyperparameters
    and a predefined range of values using xgboost as the model.
    Creates a new mode at each iteration and evaluates the model using the
    test parameters and  measures the log-loss. The model with the
    lowest log-loss is selected as the best model.

    Args:
      X_train: (array) training set features
      y_train: (array) training set target
      x_test: (array) validation set features
      y_test: (array) validation set target
      sample_weight: (array) class weights

    Returns:
      dict: best hyperparameters
    """

    start = time.perf_counter()
    logger.info('Start: Hyperparameter tuning....')

    try:
        # Create the tuning model using DMatrix for XGBoost
        _x_train = xgb.DMatrix(
            X_train,
            label=y_train,
            enable_categorical=True,
            weight=sample_weight
        )

        _x_test = xgb.DMatrix(
            x_test,
            label=y_test,
            enable_categorical=True
        )

        # Define Bayesian optimization callback function
        # and train at each iteration
        def xgb_eval(max_depth, learning_rate, num_boost_round, subsample,
                     colsample_bytree, gamma, reg_alpha, reg_lambda):
            params = {
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',
                'num_class': 4,
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'gamma': gamma,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }

            # Train model with current hyperparameters
            model = xgb.train(
                params,
                _x_train,
                num_boost_round=int(num_boost_round),
                # evals=[(_x_test, 'eval')],
                verbose_eval=False
            )

            # Predict probabilities
            y_pred_probs = model.predict(_x_test)
            # Compute log-loss
            return -log_loss(y_test, y_pred_probs)

        # Bounds for hyperparameters
        # TODO - configurable hyperparameters
        param_bounds = {
            # n_estimators is num_boost_round for XGBoostClassifier
            'num_boost_round': [100, 300],
            'max_depth': [3, 10],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.6, 1.0],
            'colsample_bytree': [0.6, 1.0],
            'gamma': [0, 5],
            'reg_alpha': [0, 1],
            'reg_lambda': [1, 5],
        }

        # Bayesian optimization
        optimizer = BayesianOptimization(
            f=xgb_eval,
            pbounds=param_bounds,
            verbose=False
        )
        # Run the optimization tasks then extract optimized results
        optimizer.maximize(init_points=5, n_iter=25)

        elapsed = _get_elapsed(start, time.perf_counter())
        logger.info(f'End: Hyperparameter tuning - elapsed(mins): {elapsed}')

        # Tuning is done, get the best parameters

        best_params = optimizer.max['params']
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['num_boost_round'] = int(best_params['num_boost_round'])
        best_params['learning_rate'] = float(best_params['learning_rate'])
        best_params['subsample'] = float(best_params['subsample'])
        best_params['colsample_bytree'] = float(
            best_params['colsample_bytree'])
        best_params['gamma'] = int(best_params['gamma'])
        best_params['reg_alpha'] = float(best_params['reg_alpha'])
        best_params['reg_lambda'] = int(best_params['reg_lambda'])

        logger.info("Best Parameters:", best_params)

        return best_params
    except Exception as e:
        logger.error(f'Error tuning hyperparameters: {e}')
        return None


def _create_and_train_model(
        X_train, y_train, x_test, y_test, h_params, sample_weight):
    """
    This function trains the xgboost model using the optimized
    hyperparameters. The model is a classifier with categorical and
    continuous features. The target variable is a multi-class
    classification with 4 classes. The dataset is highly imbalanced
    leaning towards the '0 Days' class and so class weights
    are computed and used in training.

    The model parameter chosen is to minimize the log-
    loss and optimize recall for the minority classes.

    Args:
      X_train: (array) training set features
      y_train: (array) training set target
      x_test: (array) test set features
      y_test: (array) test set target
      h_params: (dict) hyperparameters
      sample_weight: (array) class weights

    Returns:
      object: trained model
    """

    start = time.perf_counter()
    logger.info('Start: Training model...')

    # Train model with best parameters
    params = {
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': 4,
        'max_depth': h_params['max_depth'],
        'learning_rate': h_params['learning_rate'],
        'subsample': h_params['subsample'],
        'colsample_bytree': h_params['colsample_bytree'],
        'gamma': h_params['gamma'],
        'reg_alpha': h_params['reg_alpha'],
        'reg_lambda': h_params['reg_lambda'],
    }
    num_boost_round = h_params['num_boost_round']

    try:

        # Create the tuning model using DMatrix for XGBoost
        _x_train = xgb.DMatrix(
            X_train,
            label=y_train,
            enable_categorical=True,
            weight=sample_weight
        )

        _x_test = xgb.DMatrix(
            x_test,
            label=y_test,
            enable_categorical=True,
        )

        # Train model
        model_xgb = xgb.train(
            params,
            _x_train,
            # evals=[(_x_test, 'eval')],
            num_boost_round=int(num_boost_round),
        )

        # Predict
        y_pred_probs = model_xgb.predict(_x_test)
        y_pred = y_pred_probs.argmax(axis=1)

        # Evaluate
        final_log_loss = log_loss(y_test, y_pred_probs)
        logger.info(f'Final Log-Loss: {final_log_loss:.3f}')

        elapsed = _get_elapsed(start, time.perf_counter())
        logger.info(f'End: Training model - elapsed(mins): {elapsed}')

        # Log model metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        logger.info(
            f'Model metrics: Accuracy={accuracy:.3f}, '
            f'Precision: {precision:.3f}, '
            f'Recall: {recall:.3f}, F1: {f1:.3f}'
        )

        return model_xgb

    except Exception as e:
        logger.error(f'Error training model: {e}')
        return None


def _save_model(model):
    """
    Save model to disk. The function saves the trained model
    to the specified path using pickle.

    The model filename will be in the format: 'model_name_YYYYMMDDHHMMSS.pkl'
    - with the postfixed build timestamp, to serve as its unique version
    identifier.

    The environment variable responsible for disseminating the correct
    model file to deploy for the inference  service will dynamically
    update. This operation updates the in-memory setting, as well as
    the environment file on disk.

    Args:
      model: (object) trained model
    """

    # Begin: Saving model to disk

    # 1. Get the current deployed model name

    model_path = settings.model_path
    model_fname = settings.model_name

    # 2. Prepare the timestamp suffix for the model filename

    date_suffix = datetime.now().strftime('%Y%m%d%H%M%S')

    # 3. Split the current filename to remove the old timestamp and

    # reuse the base filename
    base, ext = os.path.splitext(model_fname)

    # Find the last underscore in the base
    if "_" in base:
        base_name, _ = base.rsplit("_", 1)
        model_fname = base_name

    # 4. Build the new model filename

    model_fname = f'{model_fname}_{date_suffix}.pkl'
    # Set the new model path and filename
    model_pname = f'{model_path}/{model_fname}'

    # 5. Save model into the deployment path

    logger.debug(f'Saving model to {model_pname}')
    with open(model_pname, 'wb') as f:
        # Serialize the model
        pkl.dump(model, f)

    # 6. Update the deployed model name in the settings

    settings.update({'MODEL_NAME': model_fname})


def _get_elapsed(start, end):
    return humanfriendly.format_timespan(end - start)
