from acepy.index import FeatureIndexCollection

fea_ind1 = FeatureIndexCollection([(0, 1), (0, 2), (0, (3, 4)), (1, (0, 1))], feature_size=5)

print(fea_ind1)
fea_ind1.update((0, 0))
print(fea_ind1)
fea_ind1.update([(1, 2), (1, (3, 4))])
print(fea_ind1)
fea_ind1.difference_update([(0, [3, 4, 2])])
print(fea_ind1)