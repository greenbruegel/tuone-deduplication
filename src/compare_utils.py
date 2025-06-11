import recordlinkage
from recordlinkage.base import BaseCompareFeature
import jellyfish

class CompareAnyStringListMatch(BaseCompareFeature):
    """
    Compare feature that matches if any string in a list matches any string in another list
    using string similarity metrics.
    """
    
    def __init__(self, left_on, right_on, threshold=0.8, method='jarowinkler', *args, **kwargs):
        """
        Initialize the comparison feature.
        
        Parameters
        ----------
        left_on : str
            The name of the column in the left DataFrame
        right_on : str
            The name of the column in the right DataFrame
        threshold : float
            The similarity threshold (0 to 1)
        method : str
            The string similarity method to use ('jarowinkler', 'levenshtein', etc.)
        """
        super().__init__(left_on, right_on, *args, **kwargs)
        self.threshold = threshold
        self.method = method

    def _compute_vectorized(self, s1, s2):
        """
        Compute the similarity between two series of string lists.
        
        Parameters
        ----------
        s1 : pandas.Series
            Series containing lists of strings from the left DataFrame
        s2 : pandas.Series
            Series containing lists of strings from the right DataFrame
            
        Returns
        -------
        pandas.Series
            Series containing the maximum similarity score between any pair of strings
        """
        def get_similarity(str1, str2):
            if not isinstance(str1, list):
                str1 = [str1]
            if not isinstance(str2, list):
                str2 = [str2]
                
            # Convert all items to strings and handle None/NaN
            str1 = [str(x).lower() if x is not None else '' for x in str1]
            str2 = [str(x).lower() if x is not None else '' for x in str2]
            
            # If either list is empty, return 0
            if not str1 or not str2:
                return 0.0
                
            # Calculate similarity between all pairs
            max_sim = 0.0
            for s1 in str1:
                for s2 in str2:
                    if self.method == 'jarowinkler':
                        sim = jellyfish.jaro_winkler_similarity(s1, s2)
                    elif self.method == 'levenshtein':
                        # Normalize Levenshtein distance to 0-1 range
                        max_len = max(len(s1), len(s2))
                        if max_len == 0:
                            sim = 1.0
                        else:
                            sim = 1 - (jellyfish.levenshtein_distance(s1, s2) / max_len)
                    else:
                        raise ValueError(f"Unknown method: {self.method}")
                    
                    max_sim = max(max_sim, sim)
            
            return max_sim

        # Apply the similarity function to each pair of values
        return s1.combine(s2, get_similarity) 