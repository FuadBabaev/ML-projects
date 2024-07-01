import numpy as np


def calc_recall(true_labels, pred_labels, k, exclude_self=False, return_mistakes=False):
    '''
    счиатет recall@k для приближенного поиска соседей
    
    true_labels: np.array (n_samples, k)
    pred_labels: np.array (n_samples, k)
    
    exclude_self: bool
        Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
    return_mistakes: bool
        Возвращать ли ошибки
    
    returns:
        recall@k
        mistakes: np.array (n_samples, ) с количеством ошибок
    '''
    n = true_labels.shape[0]
    n_success = []
    shift = int(exclude_self)
    
    for i in range(n):
        n_success.append(np.intersect1d(true_labels[i, shift:k+shift], pred_labels[i, shift:k+shift]).shape[0])
        
    recall = sum(n_success) / n / k
    if return_mistakes:
        mistakes = k - np.array(n_success)
        return recall, mistakes
    return recall


def plot_ann_performance(*args, **kwargs):
    '''
    some docstring :)
    '''
    pass

    
def analyze_ann_method(*args, **kwargs):
    '''
    some docstring :)
    '''
    pass



# Для FASHION MNIST
def knn_predict_classification(neighbor_ids, tr_labels, n_classes, distances=None, weights='uniform'):
    '''
    по расстояниям и айдишникам получает ответ для задачи классификации
    
    distances: (n_samples, k) - расстояния до соседей
    neighbor_ids: (n_samples, k) - айдишники соседей
    tr_labels: (n_samples,) - метки трейна
    n_classes: кол-во классов
    
    returns:
        labels: (n_samples,) - предсказанные метки
    '''
    
    n, k = neighbor_ids.shape

    labels = np.take(tr_labels, neighbor_ids)
    labels = np.add(labels, np.arange(n).reshape(-1, 1) * n_classes, out=labels)

    if weights == 'uniform':
        w = np.ones(n * k)
    elif weights == 'distance' and distances is not None:
        w = 1. / (distances.ravel() + 1e-10)
    else:
        raise NotImplementedError()
        
    labels = np.bincount(labels.ravel(), weights=w, minlength=n * n_classes)
    labels = labels.reshape(n, n_classes).argmax(axis=1).ravel()
    return labels


# Для крабов!
def get_k_neighbors(distances, k):
    '''
    считает по матрице попарных расстояний метки k ближайших соседей
    
    distances: (n_queries, n_samples)
    k: кол-во соседей
    
    returns:
        labels: (n_queries, k) - метки соседей
    '''
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    lowest_distances = np.take_along_axis(distances, indices, axis=1)
    neighbors_idx = lowest_distances.argsort(axis=1)
    indices = np.take_along_axis(indices, neighbors_idx, axis=1) # sorted
    sorted_distances = np.take_along_axis(distances, indices, axis=1)
    return sorted_distances, indices


# Для крабов! Пишите сами...
def knn_predict_regression(labels, y, weights='uniform', distances=None):
    '''
    по расстояниям и айдишникам получает ответ для задачи регрессии
    '''
    pass
