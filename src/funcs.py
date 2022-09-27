import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score


def plot_hist(data, title, label, vlines=None):
    
    '''Plot histogram of continuous distribution
    
    Parameters
    ----------
    data : array
        Continous data to be visualized
    title : str
        Title of histogram
    label : str 
        x-axis label, description of data being visualized
    vlines : list, optional
        x values to plot vertical lines
        
    '''
    
    plt.figure(figsize=(10,6))
    heights = plt.hist(data)[0]
    plt.title(title, fontsize=18)
    plt.xlabel(label, fontsize=14)
    
    if vlines:
        for v_line in vlines:
            plt.vlines(v_line, 0, int(max(heights) * .75), colors='r', lw=5)
            
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show();
    
    
    

def plot_churn(data):
    
    '''Create detailed visual of churn distribution
    
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing churn column to visualize
    
    '''
    
    idx, vals = data['Churn'].value_counts().index.astype(str), data['Churn'].value_counts().values
    pcts = data['Churn'].value_counts(normalize=True).values

    plt.figure(figsize=(10,6))
    plt.bar(idx, vals)
    plt.title('Distribution of Employee Churn', fontsize=18)
    plt.xlabel('Employee Churn', fontsize=14)
    plt.ylabel('Number Employees', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim(0, vals[0] + 250)

    plt.text(x=-.075, y=vals[0]+25, s=f"{round(pcts[0], 2) * 100}%", fontsize=14)
    plt.text(x=.925, y=vals[1]+25, s=f"{round(pcts[1], 2) * 100}%", fontsize=14)

    plt.show();
    
    


def model_churn(model, X_train, X_test, y_train, y_test):
    
    '''Fit classification model and display metrics for churn prediction
    
    Function accepts a classification model along with train and test data.
    Fits the model on training data and outputs accuracy, precision, and
    recall score for both training and test data. Displays confusion
    matrix for each as well.
    
    Parameters
    ----------
    model : scikit-learn classification model
        classification model making predictions
    X_train : pandas DataFrame
        Training features
    X_test : pandas DataFrame
        Testing features
    y_train : pandas Series
        Training target
    y_test : pandas Series
        Testing target
    
    '''
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    print('Training Accuracy: ', accuracy_score(y_train, train_preds))
    print('Testing Accuracy: ', accuracy_score(y_test, test_preds))
    print('-------------------')
    print('Training Precision: ', precision_score(y_train, train_preds))
    print('Testing Precision: ', precision_score(y_test, test_preds))
    print('-------------------')
    print('Training Recall: ', recall_score(y_train, train_preds))
    print('Testing Recall: ', recall_score(y_test, test_preds))
    print('-------------------')

    fig, ax = plt.subplots(ncols=2, figsize=(10,6))
    ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, ax=ax[0])
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax[1])

    
def plot_grid(X):
    
    '''Plot 3x3 grid of images
    
    Parameters
    ----------
    X : numpy array
        array of images to visualize
    
    '''
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5,5))
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(X[i*3+j], cmap=plt.get_cmap('gray'))    
    