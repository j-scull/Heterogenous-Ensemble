"""
Defines a HeterogenousEnsembleClassifier and a StackedHeterogenousEnsembleClassifier class that work with the scikit-learn API.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.multiclass import unique_labels
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.datasets import load_iris


#--------------------------------------------------------------------------------------------------------------------------------------------


class HeterogenousEnsembleClassifier(BaseEstimator, ClassifierMixin):
    
    """
    An ensemble classifier that uses heterogeneous models at the base layer. Base models are different due to different hyper-parameters used.

    Parameters
    ----------
    base_estimator: scikit-learn estimator 
        The model type to be used at the base layer of the ensemble model.

    hp_range_map: dictionary
        A dictinary of hyperparamters and the ranges of values that will be used from them
        
    n_estimators: int
        How many models to use in the ensemble
        
    bootstrap: boolean
        Wheter or not to use bootstrap sampling when training base estimators
    
    Attributes
    ----------
    classes_ : array of shape = [n_classes] 
        The classes labels.


    Notes
    -----
    The default values for most base learners are used, unless hyperparameter ranges are specified

    See also
    --------
    

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = HeterogenousEnsembleClassifier(tree.DecisionTreeClassifier(), {'max_depth':[5, 10, 15], })
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
    
    def __init__(self, base_estimator = svm.SVC(), n_estimators = 10, hp_range_map = None, bootstrap = True, random_state=None, verbosity = 0):

        """
        Setup a HeterogenousEnsembleClassifier .
        Parameters
        ----------
        base_estimator: The model type to be used at the base layer of the ensemble model.
        n_estimators: How many models to use in the ensemble
        hp_range_map: A dictinary of hyperparamters and the ranges of values that will be used from them
        bootstrap: Wheter or not to use bootstrap sampling wehn training base estimators
        random_state: Either int ofr np.random.RandomState. Set as an int for reproduceable results across all calls to fit
        verbose: If set to 1, gives calls to fit gives details of ensemble construction during calls to fit
        
        Returns
        -------
        The estimator
        """     

        # Initialise random state if set
        self.random_state = random_state
        
        # Initialise class variabels
        self.base_estimator = base_estimator
        self.hp_range_map = hp_range_map
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.verbosity = verbosity
        
        
    # The fit function to train a classifier
    def fit(self, X, y):
        """
        Builds a Bagging ensemble of estimators from the training data.
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        y: The class labels of the dataset.
        
        Returns
        -------
        The fitted classifier.

        """
        
        # Check that X and y have the correct shape - converts to np.ndarray
        X, y = check_X_y(X, y)

        # Check that y consists of single non-continuous labels
        if type_of_target(y) not in ['multiclass', 'binary']: 
            raise ValueError('Unknown label type: HeterogenousEnsembleClasifier only supports classification tasks')

        # Store the number of descriptive features
        self.n_features_in_ = X.shape[1]

        # Get distinct classes in the data
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
                
        # set the random_state for reproducable results
        random_state = check_random_state(self.random_state)
            
        # build the ensemble using a ranom selection of hyperparameters
        self.estimators_ = []
        for i in range(self.n_estimators):
            model = clone(self.base_estimator)
            # defaults to empty if no hp_range is set
            param_selection = {}     
            if self.hp_range_map:
                for param, values in self.hp_range_map.items():
                    param_selection[param] = random_state.choice(values)
            model.set_params(**param_selection, random_state=random_state)
            self.estimators_.append(model)
        
        # Train each model in the ensemble 
        if self.verbosity: print('Building ensemble:')
        for model in self.estimators_:
            
            # if bootstrap=True, use bootstrap aggregation with 100% sampling
            X_sample, y_sample = resample(X, y, replace=self.bootstrap, n_samples=X.shape[0], random_state=random_state)
            model.fit(X_sample, y_sample)
            if self.verbosity: print(model)
            
        return self


    def predict(self, X):
        """
        Predict class label for X.
        
        The predicted class of an input sample is computed as the class with the highest mean predicted probability. 
        If base estimators do not implement a predict_proba method, then it resorts to voting.
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        
        Returns
        -------
        The predicted class labels for X .
        """
        # Check if the model has been fitted
        check_is_fitted(self, 'estimators_')
        
        # Check X is valid
        X = check_array(X)
        
        # Check X has the correct number of features
        if not X.shape[1] == self.n_features_in_:
            raise ValueError('Input mismatch: array has {} features, expected {} features'.format(
                              X.shape[1], self.n_features_in_))
            
        probabilities = self._ensemble_probabilities(X)
        
        # average probabilities together
        average_probabilites = np.array([np.sum(output, axis=0) / self.n_estimators for output in probabilities])

        # return the class with the highest probability 
        return np.array([self.classes_[np.argmax(output)] for output in average_probabilites])
        
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as the mean predicted class probabilities 
        of the base estimators in the ensemble. If base estimators do not implement a predict_proba method, then it 
        resorts to voting and the predicted class probabilities of an input sample represents the proportion of estimators 
        predicting each class.
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        
        Returns
        -------
        The predicted class labels for X.
        """
        # Check if the model has been fitted
        check_is_fitted(self, 'estimators_')
        
        # Check X is valid
        X = check_array(X)
        
        # Check X has the correct number of features
        if not X.shape[1] == self.n_features_in_:
            raise ValueError('Input mismatch: array has {} features, expected {} features'.format(
                              X.shape[1], self.n_features_in_))

        probabilities = self._ensemble_probabilities(X)
        
        return np.array([np.sum(output, axis=0) / self.n_estimators for output in probabilities])

    
    #---------------------------------utility functions-----------------------------------------
    
    def _ensemble_probabilities(self, X):
        """
        Gets the total probability scores given by each estimator in the ensemble for each class.
        Doesn't aggregate probabilities.
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        
        Returns
        -------
        The class probabilities of the input samples.
        An 3D numpy array of shape(X.shape[0], n_estimators, n_classes)
        
        * This function is used repeatedly by both classes in this notebook
        """
        # check if the base estimator implements predict_proba
        try:
            implements_predict_proba = callable(getattr(self.estimators_[0], 'predict_proba'))
        except AttributeError:
            implements_predict_proba = False
        # check for probabiltiy attribute i.e. for SVC
        probability_is_false = hasattr(self.estimators_[0], 'probability') and self.estimators_[0].probability == False
    
        probabilities = []
    
        if implements_predict_proba and not probability_is_false:
            
            # can use the base estimators' predict_proba
            for row in X:
                ensemble_output = []
                # predict for each model in the ensemble
                for model in self.estimators_:
                    model_output = model.predict_proba([row])[0]
                    ensemble_output.append(model_output)
                probabilities.append(ensemble_output)
            
        else:
            
            # use majority voting
            for row in X:
                ensemble_output = []
                for model in self.estimators_:
                    model_output =  model.predict([row])[0]
                    ensemble_output.append([1 if cls == model_output else 0 for cls in self.classes_])
                probabilities.append(ensemble_output)

        return np.array(probabilities)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



class StackedHeterogenousEnsembleClassifier(BaseEstimator, ClassifierMixin):
    
    """
    An ensemble classifier that uses heterogeneous models at the base layer. Base models are different due to different hyper-parameters used. Aggrefgattion is perfomred using a stack layer model.

    Parameters
    ----------
    base_estimator: scikit-learn estimator 
        The model type to be used at the base layer of the ensemble model.

    hp_range_map: dictionary
        A dictinary of hyperparamters and the ranges of values that will be used from them
        
    n_estimators: int
        The number of models to use in the ensemble
        
    bootstrap: boolean
        Whether or not to use bootstrap sampling wehn training base estimators
    
    stack_layer_estimator: scikit-learn estimator 
        Estimator type of the stack  layer model
        
    base_stack_data_ratio: float
        The ratio with which to split the data for straing the base and stack layers.
        
    Attributes
    ----------
    classes_ : array of shape = [n_classes] 
        The classes labels.

    Notes
    -----
    The default values for most base learners are used, unless hyperparameter ranges are specified

    See also
    --------
    

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = StackedHeterogenousEnsembleClassifier(tree.DecisionTreeClassifier(), {'max_depth':[5, 10, 15], })
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
    
    def __init__(self, base_estimator = svm.SVC(), n_estimators = 10, hp_range_map = None, bootstrap = True, stack_layer_estimator = svm.SVC(), base_stack_data_ratio = 0.7, random_state=None, verbosity = 0):

        """Setup a StackedHeterogenousEnsembleClassifier classifier .
        Parameters
        ----------
        base_estimator: The model type to be used at the base layer of the ensemble model.
        hp_range_map: A dictinary of hyperparamters and the ranges of values that will be used from them
        n_estimators: How many models to use in the ensemble
        bootstrap: Wheter or not to use bootstrap sampling wehn training base estimators
        stack_layer_estimator: Estimator type of the stack  layer model
        base_stack_data_ratio: The ratio with which to split the data for straing the base and stack layers.
        
        Returns
        -------
        The estimator
        """     

        # Initialise ranomd state if set
        self.random_state = random_state
        
        # Initialise class variabels
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.hp_range_map = hp_range_map
        self.bootstrap = bootstrap
        self.stack_layer_estimator = stack_layer_estimator
        self.base_stack_data_ratio = base_stack_data_ratio
        self.verbosity = verbosity
        

         
    def fit(self, X, y):
        """
        Builds a Bagging ensemble of estimators with a final layer stacked estimator from the training data.
        Trains the ensemble and the stacked layer using differnt subsets of the data - the ratio of the subset split
        is given by the base_stack_data_ratio.
        The stack layer estimator is fitted using the predict_proba output of the ensemble layer
        
        * In the documentaion, stacked estimators are trained using cross validation, here we only use a single split
        
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        y: The class labels of the dataset.
        
        Returns
        -------
        The fitted classifier.

        """
        # Check that X and y have the correct shape - converts to np.ndarray
        X, y = check_X_y(X, y)

        # Check that y consists of single non-continuous labels
        if type_of_target(y) not in ['multiclass', 'binary']: 
            raise ValueError('Unknown label type: HeterogenousEnsembleClasifier only supports classification tasks')

        # Store the number of descriptive features
        self.n_features_in_ = X.shape[1]

        # Get distinct classes in the data
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
        
        # set the random_state for reproducable results
        random_state = check_random_state(self.random_state)
        
        # split the input data into training and validation sets using the base_stack_data_ratio
        X_base, X_stack, y_base, y_stack = train_test_split(X, y, 
                                                        random_state = check_random_state(self.random_state), 
                                                        shuffle=True, 
                                                        stratify = y, 
                                                        train_size = self.base_stack_data_ratio)
        
        # train the ensemble with the base data
        self.ensemble_ = HeterogenousEnsembleClassifier(self.base_estimator, 
                                                        self.n_estimators, 
                                                        self.hp_range_map,
                                                        self.bootstrap,
                                                        self.random_state, 
                                                        self.verbosity)
        self.ensemble_.fit(X_base, y_base)
    
        # get the ensemble's output
        ensemble_output = self.ensemble_._ensemble_probabilities(X_stack)
        ensemble_output = np.array([np.concatenate(output) for output in ensemble_output])
        
        # fit the stack layer with the ensemble output
        self.stack_layer_estimator.fit(ensemble_output, y_stack)
    
        return self
    


    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        """
        Predict class label for X.
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        
        Returns
        -------
        The predicted class labels for X .
        """
        # Perform checks
        check_is_fitted(self, 'ensemble_')
        X = check_array(X)
        if not X.shape[1] == self.n_features_in_:
            raise ValueError('Input mismatch: array has {} features, expected {} features'.format(
                              X.shape[1], self.n_features_in_))
        
        # get the ensemble's output
        ensemble_output = self.ensemble_._ensemble_probabilities(X)
        ensemble_output = np.array([np.concatenate(output) for output in ensemble_output])
        
        # return the stack layer preiction
        return self.stack_layer_estimator.predict(ensemble_output)

    

    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        """
        Predict class probabilities for X using final_estimator_.predict_proba.
        
        Parameters
        ----------
        X: The descriptive features of the dataset.
        
        Returns
        -------
        The class probabilities of the input samples.
        """
        # Perform checks
        check_is_fitted(self, 'ensemble_')
        X = check_array(X)
        if not X.shape[1] == self.n_features_in_:
            raise ValueError('Input mismatch: array has {} features, expected {} features'.format(
                               X.shape[1], self.n_features_in_))
        
        # get the ensemble's output 
        ensemble_output = self.ensemble_._ensemble_probabilities(X)
        ensemble_output = np.array([np.concatenate(output) for output in ensemble_output])
        
        # return the stack layer probabilities
        return self.stack_layer_estimator.predict_proba(ensemble_output)