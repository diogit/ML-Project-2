# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn import cluster, mixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
#from scipy.misc import imread
from matplotlib.pyplot import imread

matplotlib.style.use('seaborn-deep')

RADIUS = 6371

def lat_lon_to_3d(lon,lat):
    x = RADIUS * np.cos(lat*np.pi/180) * np.cos(lon*np.pi/180)
    y = RADIUS * np.cos(lat*np.pi/180) * np.sin(lon*np.pi/180)
    z = RADIUS * np.sin(lat*np.pi/180)
    return x,y,z

def read_data(filename='tp2_data.csv'):
    #Read data and convert Latitude/Longitude into x/y/z coordinates
    data = pd.read_csv(filename)
    lon = data.longitude
    lat = data.latitude
    faults = data.fault
    x,y,z = lat_lon_to_3d(lon,lat)
    #Original data plot
    data = np.zeros((lon.shape[0],3))
    data[:,0] = x
    data[:,1] = y
    data[:,2] = z
    return data, lon, lat, faults
    
def compute_read_indexes(data,labels,faults):
    sf = sc = tp = tn = 0.0
    for ix in range(len(labels)-1):
        same_fault = faults[ix] == faults[ix+1:]
        same_cluster = labels[ix] == labels[ix+1:]
        sf += np.sum(same_fault)
        sc += np.sum(same_cluster)
        tp += np.sum(np.logical_and(same_fault,same_cluster))
        tn += np.sum(np.logical_and(np.logical_not(same_fault), np.logical_not(same_cluster)))
    total = len(labels)*(len(labels)-1)/2
    precision = tp/sc
    recall = tp/sf
    rand = (tp+tn)/total
    F1 = precision*recall*2/(precision+recall)
    return precision, recall, rand, F1, adjusted_rand_score(labels, faults), silhouette_score(data, labels)

def plot_indexes(filename,title,xlabel,ylabel,xrange,scores):
    plt.figure(figsize=(20,10))
    plt.grid(b=True, which='major', axis='both')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xrange,scores['precision'],'ro-',label='Precision',linewidth=1,markersize=7)
    plt.plot(xrange,scores['recall'],'y^-',label='Recall',linewidth=1,markersize=7)
    plt.plot(xrange,scores['rand'],'gd-',label='Rand',linewidth=1,markersize=7)
    plt.plot(xrange,scores['f1'],'cs-',label='F1',linewidth=1,markersize=7)
    plt.plot(xrange,scores['ari'],'bP-',label='Adjusted Rand',linewidth=1,markersize=7)
    plt.plot(xrange,scores['silhouette'],'mD-',label='Silhouette',linewidth=1,markersize=7)
    plt.legend(loc='best')
    plt.savefig(filename,dpi=300)

def plot_classes(labels,lon,lat,filename, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5),frameon=False)    
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(10,5),frameon=False)    
    plt.subplot(111)
    plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off')
    plt.savefig(filename,dpi=300)

def write_results(results, results_filename='Best Score Results.txt'):
    f = open(results_filename,'a')
    f.write(results)
    f.close()

def kmeans(data,lon,lat,faults,min_k=20,max_k=100,interval_k=5,remove_noise=True):
    filename = 'KMeans Scores'
    title = 'KMeans Scores for k varying from '+str(min_k)+' to '+str(max_k)+' in intervals of '+str(interval_k)
    if remove_noise == True:
        data = data[faults!=-1]
        faults = faults[faults!=-1].reset_index(drop=True)
        filename += ' Without Noise'
        title += ' without noise'
    scores = {'precision':[],'recall':[],'rand':[],'f1':[],'ari':[],'silhouette':[]}
    best_precision=best_recall=best_rand=best_F1=best_ARI=best_silhouette=0,0
    for k in range(min_k,max_k+1,interval_k):
        labels = KMeans(n_clusters=k).fit_predict(data)
        precision, recall, rand, F1, adjusted_rand_score, silhouette_score = compute_read_indexes(data,labels, faults)
        if(precision > best_precision[1]):
            best_precision = k, precision
        scores['precision'].append(precision)
        if(recall > best_recall[1]):
            best_recall = k, recall
        scores['recall'].append(recall)
        if(rand > best_rand[1]):
            best_rand = k, rand
        scores['rand'].append(rand)
        if(F1 > best_F1[1]):
            best_F1 = k, F1
        scores['f1'].append(F1)
        if(adjusted_rand_score > best_ARI[1]):
            best_ARI = k, adjusted_rand_score
        scores['ari'].append(adjusted_rand_score)
        if(silhouette_score > best_silhouette[1]):
            best_silhouette = k, silhouette_score
        scores['silhouette'].append(silhouette_score)
    write_results('\nBest '+filename+':'
          '\n\tBest Precision: '+str(best_precision[1])+' with k='+str(best_precision[0])+
          '\n\tBest Recall: '+str(best_recall[1])+' with k='+str(best_recall[0])+
          '\n\tBest Rand: '+str(best_rand[1])+' with k='+str(best_rand[0])+
          '\n\tBest F1: '+str(best_F1[1])+' with k='+str(best_F1[0])+
          '\n\tBest Adjusted Rand Index: '+str(best_ARI[1])+' with k='+str(best_ARI[0])+
          '\n\tBest Silhouette Score: '+str(best_silhouette[1])+' with k='+str(best_silhouette[0]))
    plot_indexes(filename=filename,
                 title=title,
                 xlabel='K - Number of Centroids/Clusters',
                 ylabel='Scores',
                 xrange=np.linspace(min_k,max_k,len(scores['precision'])),
                 scores=scores)

def epsilon_selection(data, faults):
    #Use K-Nearest-Neighbours to compute each points 4th neighbour.
    reg = KNeighborsClassifier(4)
    reg.fit(data, np.zeros(data[:,0].shape))
    dists, _ = reg.kneighbors(data, return_distance=True) #Compute KNeighbours
    dists = dists[:,3] #Select distance to 4th neighbour
    #Sort distances
    dists = np.sort(dists)
    #Revert array
    y = dists[::-1]
    #Calculate the threshold point by computing the percentage of noise
    noise_percentage = round(len(faults[faults==-1])/len(faults),2)
    #1st [0] = 1st array of tuple, 2nd [0] value of array
    threshold_point = np.where(y == y[int(len(y)*noise_percentage)])[0][0]
    plt.figure(figsize=(20,10))
    plt.grid(b=True, which='major', axis='both')
    plt.title('Sorted 4-dist Graph for the Dataset')
    plt.xlabel('Seismic event')
    plt.ylabel('Distance to its 4th nearest neighbour')
    #Plot distances (same as plt.plot(range(len(dists)),dists))
    plt.plot(y)
    #Plot threshold line that separates noise from clusters
    plt.axvline(x=threshold_point,color='g')
    #Plot graph intersection
    plt.plot(threshold_point,y[threshold_point],'rx',
     label='Threshold point - the first point in the first “valley” of the sorted k-dist graph',
     markersize=10)
    plt.legend(bbox_to_anchor=(0.22,0.17))
    plt.savefig('Epsilon Selection Graph',dpi=300)
    write_results('\nPercentage of noise: '+str(noise_percentage*100)+'%')
    write_results('\nEpsilon selection: '+str(y[threshold_point]))
    #Epsilon
    return y[threshold_point]

def dbscan(data,lon,lat,faults,min_eps=100,max_eps=1000,interval_eps=50):
    scores = {'precision':[],'recall':[],'rand':[],'f1':[],'ari':[],'silhouette':[]}
    best_precision=best_recall=best_rand=best_F1=best_ARI=best_silhouette=0,0
    #Check scores for (100km,1000km) with intervals of 50km
    for eps in range(min_eps,max_eps+1,50):
        labels = cluster.DBSCAN(eps=eps).fit_predict(data)
        precision, recall, rand, F1, adjusted_rand_score, silhouette_score = compute_read_indexes(data,labels, faults)
        if(precision > best_precision[1]):
            best_precision = eps, precision
        scores['precision'].append(precision)
        if(recall > best_recall[1]):
            best_recall = eps, recall
        scores['recall'].append(recall)
        if(rand > best_rand[1]):
            best_rand = eps, rand
        scores['rand'].append(rand)
        if(F1 > best_F1[1]):
            best_F1 = eps, F1
        scores['f1'].append(F1)
        if(adjusted_rand_score > best_ARI[1]):
            best_ARI = eps, adjusted_rand_score
        scores['ari'].append(adjusted_rand_score)
        if(silhouette_score > best_silhouette[1]):
            best_silhouette = eps, silhouette_score
        scores['silhouette'].append(silhouette_score)
    write_results('\nBest DBSCAN Scores:'
          '\n\tBest Precision: '+str(best_precision[1])+' with eps='+str(best_precision[0])+
          '\n\tBest Recall: '+str(best_recall[1])+' with eps='+str(best_recall[0])+
          '\n\tBest Rand: '+str(best_rand[1])+' with eps='+str(best_rand[0])+
          '\n\tBest F1: '+str(best_F1[1])+' with eps='+str(best_F1[0])+
          '\n\tBest Adjusted Rand Index: '+str(best_ARI[1])+' with eps='+str(best_ARI[0])+
          '\n\tBest Silhouette Score: '+str(best_silhouette[1])+' with eps='+str(best_silhouette[0]))
    plot_indexes(filename='DBSCAN Scores',
             title='DBSCAN algorithm with eps varying from '+str(min_eps)+' to '+str(max_eps)+' in intervals of '+str(interval_eps),
             xlabel='Epsilon value',
             ylabel='Scores',
             xrange=np.linspace(min_eps,max_eps,len(scores['precision'])),
             scores=scores)

def gaussianMM(data,lon,lat,faults,min_c=20,max_c=100,interval_c=5,remove_noise=False):
    filename = 'Gaussian Mixture Model Scores'
    title = 'Gaussian Mixture Models with c varying from '+str(min_c)+' to '+str(max_c)+' in intervals of '+str(interval_c)
    if remove_noise == True:
        data = data[faults!=-1]
        faults = faults[faults!=-1].reset_index(drop=True)
        filename += ' Without Noise'
        title += ' without noise'
    scores = {'precision':[],'recall':[],'rand':[],'f1':[],'ari':[],'silhouette':[]}
    best_precision=best_recall=best_rand=best_F1=best_ARI=best_silhouette=0,0
    for c in range(min_c,max_c+1,interval_c):
        gaussian = mixture.GaussianMixture(n_components=c).fit(data)
        labels = gaussian.predict(data)
        precision, recall, rand, F1, adjusted_rand_score, silhouette_score = compute_read_indexes(data,labels, faults)
        if(precision > best_precision[1]):
            best_precision = c, precision
        scores['precision'].append(precision)
        if(recall > best_recall[1]):
            best_recall = c, recall
        scores['recall'].append(recall)
        if(rand > best_rand[1]):
            best_rand = c, rand
        scores['rand'].append(rand)
        if(F1 > best_F1[1]):
            best_F1 = c, F1
        scores['f1'].append(F1)
        if(adjusted_rand_score > best_ARI[1]):
            best_ARI = c, adjusted_rand_score
        scores['ari'].append(adjusted_rand_score)
        if(silhouette_score > best_silhouette[1]):
            best_silhouette = c, silhouette_score
        scores['silhouette'].append(silhouette_score)
    write_results('\nBest '+filename+':'
          '\n\tBest Precision: '+str(best_precision[1])+' with c='+str(best_precision[0])+
          '\n\tBest Recall: '+str(best_recall[1])+' with c='+str(best_recall[0])+
          '\n\tBest Rand: '+str(best_rand[1])+' with c='+str(best_rand[0])+
          '\n\tBest F1: '+str(best_F1[1])+' with c='+str(best_F1[0])+
          '\n\tBest Adjusted Rand Index: '+str(best_ARI[1])+' with c='+str(best_ARI[0])+
          '\n\tBest Silhouette Score: '+str(best_silhouette[1])+' with c='+str(best_silhouette[0]))
    plot_indexes(filename=filename,
         title=title,
         xlabel='C - Number of gaussian components',
         ylabel='Scores',
         xrange=np.linspace(min_c,max_c,len(scores['precision'])),
         scores=scores)

def execute():
    #Create this file + erase content if it already exists
    f = open('Best Score Results.txt','w')
    f.close()
    #Read data and convert Latitude/Longitude into x/y/z coordinates
    data, lon, lat, faults = read_data()
    kmeans(data,lon,lat,faults,remove_noise=False)
    kmeans(data,lon,lat,faults)
    
    dbscan(data,lon,lat,faults)
    epsilon_selection(data,faults)
    
    gaussianMM(data,lon,lat,faults,remove_noise=False)
    gaussianMM(data,lon,lat,faults)

    labels = KMeans(n_clusters=20).fit_predict(data)
    plot_classes(labels,lon,lat,'K-Means Cluster, k=20')
    
    labels = cluster.DBSCAN(eps=150).fit_predict(data)
    plot_classes(labels,lon,lat,'DBSCAN Cluster, eps=150')
    labels = cluster.DBSCAN(eps=148).fit_predict(data)
    plot_classes(labels,lon,lat,'DBSCAN Cluster, eps=148')

    gaussian = mixture.GaussianMixture(n_components=20).fit(data)
    labels = gaussian.predict(data)
    plot_classes(labels,lon,lat,'GMM Cluster, c=20')   

execute()