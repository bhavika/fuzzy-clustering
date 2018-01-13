### Fuzzy C Means Clustering 

Source: https://en.wikipedia.org/wiki/Fuzzy_clustering
 
A tutorial: https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html


Create some data:

    mu = [1, 1.5, 2]
    sigma = [0.1, 0.1, 0.2]
    n = 500
    p = [0.25, 0.5, 0.25]

    sigma2 = [0.3**0.5, 0.4 ** 0.5, 0.3 ** 0.5]

    dataset_1 = create_data(mu, sigma, n, p, random_seed=7)
    dataset_2 = create_data(mu, sigma2, n, p, random_seed=7)
   
Run Fuzzy C-Means Clustering:

    results_1 = FCM(dataset_1)
    labels1, centroids1 = results_1[0], results_1[1]

    print("Accuracy metrics for Data 1 - Homogeneity, Completeness, V-Measure {}".format(homogeneity_completeness_v_measure(list(dataset_1['label']), labels1)))
