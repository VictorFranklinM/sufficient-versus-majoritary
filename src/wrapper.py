import numpy as np

class TreeFeatureThresholdMap:
    def __init__(self, tree_clf):
        self.tree = tree_clf.tree_
        self.feature_threshold_map = []
        self._extract_unique_nodes()

    def _extract_unique_nodes(self):
        features = self.tree.feature
        thresholds = self.tree.threshold

        for node_idx in range(self.tree.node_count):
            # Ignores the leaves
            if self.tree.children_left[node_idx] == self.tree.children_right[node_idx]:
                continue

            feat = features[node_idx]
            thres = thresholds[node_idx]
            key = (feat, round(thres, 5))

            # Uses only keys that didn't appear yet.
            if key not in self.feature_threshold_map:
                self.feature_threshold_map.append(key)

    def get_mapping(self):
        return self.feature_threshold_map

    def __repr__(self):
        return f"<TreeFeatureThresholdMap: {len(self.feature_threshold_map)} unique pairs>"

class ForestFeatureThresholdMap:
    def __init__(self, forest_clf, feature_names=None):
        self.forest = forest_clf
        self.feature_names = feature_names
        self.unique_pairs = []
        self.mapping = {}
        self.human_readable = []
        self._extract_from_forest()

    def _extract_from_forest(self):
        # Extracts unique pairs (feature, threshold) from all the trees.
        for t_clf in self.forest.estimators_:
            wrapper = TreeFeatureThresholdMap(t_clf)
            self.unique_pairs.extend(wrapper.get_mapping())

        count = 1
        for pair in self.unique_pairs:
            # Converts into native python types.
            feat_idx = int(pair[0]) if not isinstance(pair[0], (int, np.integer)) else pair[0]
            thres = float(pair[1]) if not isinstance(pair[1], (float, np.floating)) else pair[1]

            if (feat_idx, thres) not in self.mapping:
                self.mapping[(feat_idx, thres)] = count

                # If the feature has a name, uses the name instead of the feature.
                feat_name = self.feature_names[feat_idx] if self.feature_names is not None else f"feature_{feat_idx}"
                self.human_readable.append(f"{feat_name} <= {thres}")
                count += 1

    def get_mapping(self):
        # Returns the inner mapping (indexes and thresholds).
        return self.mapping

    def get_human_readable(self):
        # Returns readable condition list (ex: 'age <= 45.5').
        return self.human_readable

    def get_tuples(self):
        # Returns list [(feature_name, threshold)] with no text.
        tuples = []
        for (feat_idx, thres) in self.mapping.keys():
            feat_name = self.feature_names[feat_idx] if self.feature_names is not None else f"feature_{feat_idx}"
            tuples.append((feat_name, thres))
        return tuples
    
    def evaluate_instance(self, x):
        # Calculates the forest valuation for each instance x.
        # for each (feature_i, threshold_t) on the mapping:
        #     - if x[i] <= t, adds d[key]
        #     - else, adds -d[key]
        # Returns the valuation list.
        result = []
        for (feat_idx, thres), key_val in self.mapping.items():
            feat_value = x.iloc[feat_idx] if hasattr(x, "iloc") else x[feat_idx]
            if feat_value <= thres:
                result.append(key_val)
            else:
                result.append(-key_val)
        return result

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"<ForestFeatureThresholdMap: {len(self)} unique pairs>"