import pandas as pd
import re


def preprocess_text(text):
    # Remove any word that starts with the symbol @
    text = re.sub(r'@\w+', '', text)
    
    # Remove any URL
    text = re.sub(r'http\S+', '', text)
    
    # Convert every word to lowercase
    text = text.lower()
    
    # Remove any hashtag symbols
    text = re.sub(r'#', '', text)
    
    return text

def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (intersection / union)

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    
    for idx, tweet in data.iterrows():
        tweet_set = set(tweet['text'].split())
        min_distance = float('inf')
        nearest_cluster = -1
        
        for i, centroid in enumerate(centroids):
            distance = jaccard_distance(tweet_set, centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = i
                
        clusters[nearest_cluster].append(idx)
        
    return clusters


def update_centroids(data, clusters):
    new_centroids = []
    
    for cluster in clusters:
        cluster_sets = [set(data.loc[idx]['text'].split()) for idx in cluster]
        centroid = set.union(*cluster_sets)
        
        min_sum_distance = float('inf')
        best_candidate = set()
        
        for tweet_set in cluster_sets:
            sum_distance = sum(jaccard_distance(tweet_set, other_set) for other_set in cluster_sets)
            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                best_candidate = tweet_set
                
        new_centroids.append(best_candidate)
        
    return new_centroids


def k_means_clustering(data, k, max_iterations=100):
    centroids = [set(row['text'].split()) for _, row in data.sample(k).iterrows()]
    
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters)
        
        if new_centroids == centroids:
            break
            
        centroids = new_centroids
        
    return clusters, centroids
    
def sum_of_squared_errors(data, clusters, centroids):
    total_error = 0
    
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            tweet_set = set(data.loc[idx]['text'].split())
            total_error += jaccard_distance(tweet_set, centroids[i])**2
            
    return total_error


if __name__ == "__main__":

    # Read the CSV file
    data = pd.read_csv("Health-Tweets/usnewshealth.txt", sep='|', header=None, names=["tweet_id", "timestamp", "text"])

    # Remove the tweet id and timestamp
    data = data.drop(columns=["tweet_id", "timestamp"])

    # Apply the preprocessing function to each row
    data['text'] = data['text'].apply(preprocess_text)

    K_values = [2, 3, 5, 7, 10]
    results = []

    for k in K_values:
        clusters, centroids = k_means_clustering(data, k)
        error = sum_of_squared_errors(data, clusters, centroids)
        results.append((k, error))

    # Report the results
    for k, error in results:
        clusters, centroids = k_means_clustering(data, k)
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        print(f"K: {k}, Sum of squared errors: {error:.4f}")
        print("Size of each cluster:")
        
        for i, size in enumerate(cluster_sizes, start=1):
            print(f"{i}: {size} tweets")
        
        print()

